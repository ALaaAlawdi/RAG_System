from pydantic import BaseModel
from typing import Optional
from app.api.v1.enums import LLMProvider
from app.api.v1.constants import DEFAULT_K


class SourceItem(BaseModel):
    text: str
    metadata: dict


class AskRequest(BaseModel):
    query: str
    k: int = DEFAULT_K
    file_path: Optional[str] = None
    provider: LLMProvider = LLMProvider.openai


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
