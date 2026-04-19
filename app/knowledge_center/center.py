from .setup import get_vector_store, read_pdf, read_txt
from ..core.logger import setup_logger
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from typing import Optional
from uuid import uuid4

logger = setup_logger(__name__)

COLLECTION_NAME = "base_center"


class KnowledgeCenter:
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or COLLECTION_NAME

    async def add_document(self, document_path: str) -> tuple[bool, str]:
        try:
            vector_store: Chroma = await get_vector_store(self.collection_name)
            if document_path.endswith(".txt"):
                documents = await read_txt(document_path)
            else:
                documents = await read_pdf(document_path)

            if not documents:
                return False, "An error occurred while reading the document, please try again."

            uuids = [str(uuid4()) for _ in range(len(documents))]
            await vector_store.aadd_documents(documents, ids=uuids)
            return True, f"Successfully uploaded {len(documents)} chunks to {self.collection_name}"
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return False, str(e)

    async def get_answer(
        self, query: str, similarity_ratio: int, file_path: Optional[str]
    ):
        try:
            vector_store: Chroma = await get_vector_store(self.collection_name)
            if file_path:
                filter_dict = {"source": {"$in": [file_path]}}
                results = await vector_store.asimilarity_search(
                    query=query, k=similarity_ratio, filter=filter_dict
                )
            else:
                results = await vector_store.asimilarity_search(
                    query=query, k=similarity_ratio
                )
            return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return None

    async def chat(self, query: str, k: int = 4, file_path: Optional[str] = None):
        docs = await self.get_answer(query, k, file_path)
        if not docs:
            return {"answer": "No relevant documents found.", "sources": []}

        context = "\n\n".join([d["text"] for d in docs])
        prompt = (
            "You are a helpful assistant. Answer the question based solely on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = await llm.ainvoke(prompt)
        logger.info("Chat query completed successfully")
        return {"answer": response.content, "sources": docs}
