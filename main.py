import os
from app.core.config import settings  # noqa: F401 — loads OPENAI_API_KEY into env first
from fastapi import FastAPI
from app.routers.rag import router

os.makedirs("uploads", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

app = FastAPI(title="RAG System", version="1.0.0")
app.include_router(router, prefix="/api/v1")
