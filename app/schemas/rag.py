from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    query: str
    k: int = 4
    file_path: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int
