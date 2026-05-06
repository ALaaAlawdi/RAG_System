from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int
