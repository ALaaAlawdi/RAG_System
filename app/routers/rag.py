import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..knowledge_center.center import KnowledgeCenter
from ..schemas.rag import AskRequest, AskResponse, UploadResponse
from ..core.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

UPLOADS_DIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, file.filename)

    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")

    kc = KnowledgeCenter()
    success, message = await kc.add_document(save_path)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    chunks_added = int(message.split()[2]) if success else 0
    return UploadResponse(success=True, message=message, chunks_added=chunks_added)


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    kc = KnowledgeCenter()
    result = await kc.chat(query=body.query, k=body.k, file_path=body.file_path)
    return AskResponse(answer=result["answer"], sources=result["sources"])
