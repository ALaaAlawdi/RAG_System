import hashlib
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .setup import get_vector_store, get_openai_embedding_model
from ..core.config import settings
from ..core.logger import setup_logger
from ..core.llm import get_llm
from ..core.prompts import RAG_SYSTEM_PROMPT

logger = setup_logger(__name__)

COLLECTION_NAME = "base_center"
UPLOADS_DIR = settings.UPLOADS_DIR


async def _vectorize(documents: list[Document], collection_name: str) -> tuple[bool, str, int]:
    vector_store: Chroma = await get_vector_store(collection_name)

    chunk_ids = [hashlib.sha256(doc.page_content.encode()).hexdigest() for doc in documents]
    existing_ids = set(vector_store._collection.get(ids=chunk_ids)["ids"])
    new_docs = [doc for doc, cid in zip(documents, chunk_ids) if cid not in existing_ids]
    new_ids = [cid for cid in chunk_ids if cid not in existing_ids]
    skipped = len(documents) - len(new_docs)

    logger.info(f"[ADD] {len(new_docs)} new chunks, {skipped} duplicates skipped")

    if new_docs:
        await vector_store.aadd_documents(new_docs, ids=new_ids)

    total = vector_store._collection.count()
    logger.info(f"[ADD] Collection '{collection_name}' now has {total} docs")
    return True, f"Added {len(new_docs)} chunks ({skipped} duplicates skipped)", len(new_docs)


class KnowledgeCenter:
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or COLLECTION_NAME

    async def add_document(self, documents: list[Document]) -> tuple[bool, str, int]:
        if not documents:
            return False, "No content extracted from the document.", 0
        return await _vectorize(documents, self.collection_name)

    async def get_answer(self, query: str, k: int, file_path: Optional[str]):
        logger.info(f"[SEARCH] Query: '{query[:80]}' | k={k} | filter={file_path!r}")
        try:
            vector_store: Chroma = await get_vector_store(self.collection_name)
            total = vector_store._collection.count()

            if total == 0:
                logger.warning("[SEARCH] Collection is empty")
                return []

            embedding_model = await get_openai_embedding_model()
            query_embedding = await embedding_model.aembed_query(query)

            where = {"source": {"$in": [file_path]}} if file_path else None
            n = min(k, total)

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
            return docs
        except Exception as e:
            logger.error(f"[SEARCH] Failed: {e}", exc_info=True)
            return None

    async def chat(self, query: str, k: int = 5, file_path: Optional[str] = None, provider: str = "openai"):
        logger.info(f"[CHAT] Query: '{query[:80]}'")
        try:
            docs = await self.get_answer(query, k, file_path)

            if docs is None:
                return None

            if not docs:
                return {"answer": "No relevant documents found.", "sources": []}

            context = "\n\n".join(d["text"] for d in docs)
            prompt = RAG_SYSTEM_PROMPT(context=context, question=query)

            response = await get_llm(provider).ainvoke(prompt)
            logger.info(f"[CHAT] Response: {len(response.content)} chars")
            return {"answer": response.content, "sources": docs}
        except Exception as e:
            logger.error(f"[CHAT] Failed: {e}", exc_info=True)
            return None