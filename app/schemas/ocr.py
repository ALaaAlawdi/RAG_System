from pydantic import BaseModel
from typing import Optional, Literal


class OCRSubmitResponse(BaseModel):
    success: bool
    request_check_url: str
    error: Optional[str] = None


class OCRStatusResponse(BaseModel):
    status: Literal["pending", "processing", "complete", "error"]
    markdown: Optional[str] = None
    error: Optional[str] = None

    class Config:
        extra = "allow"


class OCRResponse(BaseModel):
    success: bool
    provider: str
    markdown: Optional[str] = None
    message: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
