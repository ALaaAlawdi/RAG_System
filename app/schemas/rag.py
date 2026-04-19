from pydantic import BaseModel
from typing import Optional



class SourceItem(BaseModel):
    text: str
    metadata: dict


class AskRequest(BaseModel):
    query: str
    k: int = 4
    file_path: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int
