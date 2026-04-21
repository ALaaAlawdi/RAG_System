from .setup import get_vector_store, get_openai_embedding_model, read_pdf, read_txt
from ..core.logger import setup_logger
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Optional
import hashlib

logger = setup_logger(__name__)

COLLECTION_NAME = "base_center"
UPLOADS_DIR = "uploads"
OLLAMA_MODEL = "hf.co/AlaaAlawdi/llama_finetune"

_llm_openai = ChatOpenAI(model="gpt-4o")
_llm_ollama = ChatOllama(model=OLLAMA_MODEL, temperature=0)


def _get_llm(provider: str):
    if provider == "finetune":
        logger.debug(f"[LLM] Using Ollama model: {OLLAMA_MODEL}")
        return _llm_ollama
    logger.debug("[LLM] Using OpenAI gpt-4o")
    return _llm_openai


class KnowledgeCenter:
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or COLLECTION_NAME
        logger.debug(
            f"KnowledgeCenter initialized collection='{self.collection_name}'"
        )

    
    async def add_document(self, document_path: str) -> tuple[bool, str, int]:
        logger.info(
            f"[ADD] Starting ingestion for: {document_path}"
        )
        try:
            logger.debug("[ADD] Connecting to vector store...")
            vector_store: Chroma = await get_vector_store(self.collection_name)
            logger.debug(
                f"[ADD] Docs before upload: {vector_store._collection.count()}"
            )

            if document_path.endswith(".txt"):
                documents = await read_txt(document_path)
            else:
                documents = await read_pdf(document_path)

            if not documents:
                logger.error(
                    "[ADD] No documents extracted — file may be empty or unreadable"
                )
                return False, "An error occurred while reading the document, please try again.", 0

            chunk_ids = [
                hashlib.sha256(doc.page_content.encode()).hexdigest()
                for doc in documents
            ]
            existing_ids = set(vector_store._collection.get(ids=chunk_ids)["ids"])
            new_docs = [doc for doc, cid in zip(documents, chunk_ids) if cid not in existing_ids]
            new_ids = [cid for cid in chunk_ids if cid not in existing_ids]
            skipped = len(documents) - len(new_docs)

            logger.info(
                f"[ADD] {len(new_docs)} new chunks to store, {skipped} duplicates skipped"
            )

            if new_docs:
                await vector_store.aadd_documents(new_docs, ids=new_ids)

            total = vector_store._collection.count()
            logger.info(
                f"[ADD] Done. Collection '{self.collection_name}' now has {total} total docs."
            )
            msg = f"Successfully uploaded {len(new_docs)} chunks to {self.collection_name} ({skipped} duplicates skipped)"
            return True, msg, len(new_docs)
        except Exception as e:
            logger.error(f"[ADD] Failed: {e}", exc_info=True)
            return False, str(e), 0

    async def get_answer(self, query: str, similarity_ratio: int, file_path: Optional[str]):
        logger.info(
            f"[SEARCH] Query: '{query[:80]}' | k={similarity_ratio} | filter={file_path!r}"
        )
        try:
            
            if file_path and not file_path.startswith(UPLOADS_DIR):
                logger.warning(
                    f"[SEARCH] Rejected invalid file_path: {file_path!r}"
                )
                return []

            vector_store: Chroma = await get_vector_store(self.collection_name)
            total = vector_store._collection.count()
            logger.info(
                f"[SEARCH] Collection '{self.collection_name}' has {total} docs"
            )

            if total == 0:
                logger.warning("[SEARCH] Collection is empty — nothing to search")
                return []

            logger.debug(
                f"[SEARCH] Generating query embedding via OpenAI..."
            )
            embedding_model = await get_openai_embedding_model()
            query_embedding = await embedding_model.aembed_query(query)
            logger.debug(
                f"[SEARCH] Query embedding generated (dim={len(query_embedding)})"
            )

            where = {"source": {"$in": [file_path]}} if file_path else None
            n = min(similarity_ratio, total)
            logger.debug(
                f"[SEARCH] Querying ChromaDB: n_results={n}, where={where}"
            )

            raw = vector_store._collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=where,
            )

            docs = [
                {"text": text, "metadata": meta}
                for text, meta in zip(raw["documents"][0], raw["metadatas"][0])
            ]
            logger.info(f"[SEARCH] Returned {len(docs)} results")
            if docs:
                logger.debug(
                    f"[SEARCH] Top result: {docs[0]['metadata'].get('source')} | {docs[0]['text'][:80]!r}"
                )
            return docs
        except Exception as e:
            logger.error(f"[SEARCH] Failed: {e}", exc_info=True)
            return None

    async def chat(self, query: str, k: int = 5, file_path: Optional[str] = None, provider: str = "openai"):
        logger.info(f"[CHAT] Incoming query: '{query[:80]}'")
        try:
            docs = await self.get_answer(query, k, file_path)

            if docs is None:
                logger.error("[CHAT] get_answer returned None — search failed")
                return None

            if not docs:
                logger.warning("[CHAT] No relevant documents found for query")
                return {"answer": "No relevant documents found.", "sources": []}

            context = "\n\n".join([d["text"] for d in docs])
            logger.debug(f"[CHAT] Context: {len(context)} chars across {len(docs)} chunks")

            prompt = (
                "You are a helpful assistant who is good at analyzing source information and answering questions.\n"
                "Use the following source documents to answer the user's questions.\n"
                "If you don't know the answer, just say that you don't know.\n"
                "Use three sentences maximum and keep the answer concise.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )

            logger.debug("[CHAT] Sending prompt to LLM...")
            response = await _get_llm(provider).ainvoke(prompt)
            logger.info(f"[CHAT] LLM response received ({len(response.content)} chars)")
            return {"answer": response.content, "sources": docs}
        except Exception as e:
            logger.error(f"[CHAT] Unexpected error: {e}", exc_info=True)
            return None