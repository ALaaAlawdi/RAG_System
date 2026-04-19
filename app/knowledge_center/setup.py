from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_core.documents import Document
import PyPDF2
from ..core.logger import setup_logger
from typing import Optional, List

logger = setup_logger(__name__)


async def get_openai_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large")


async def get_persistent_client():
    return chromadb.PersistentClient(path="./chroma_db")


async def get_vector_store(collection_name: Optional[str]) -> Chroma:
    embedding_model = await get_openai_embedding_model()
    client = await get_persistent_client()
    return Chroma(
        client=client,
        collection_name=collection_name if collection_name else "base_center",
        embedding_function=embedding_model,
    )


async def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )


async def get_text_chunks(text: str) -> Optional[list[str]]:
    splitter = await get_text_splitter()
    return splitter.split_text(text)


async def read_pdf(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                chunks = await get_text_chunks(raw_text)
                for i, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path,
                                "page": page_num,
                                "chunk": i,
                            },
                        )
                    )
        return documents
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


async def read_txt(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = await get_text_chunks(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "page": 1,
                        "chunk": i,
                    },
                )
            )
        return documents
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []
