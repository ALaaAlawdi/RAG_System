import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..knowledge_center.center import KnowledgeCenter, UPLOADS_DIR
from ..schemas.rag import AskRequest, AskResponse, UploadResponse
from ..core.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # Fix #10: 20 MB limit


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):

    original_name = file.filename or "file"
    stem, ext = os.path.splitext(original_name)
    ext = ext.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    content = await file.read()
    
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(UPLOADS_DIR, unique_name)

    logger.info(
        f"Saving '{original_name}' as '{unique_name}'"
    )

    try:
        with open(save_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")

    kc = KnowledgeCenter()

    success, message, chunks_added = await kc.add_document(save_path)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return UploadResponse(success=True, message=message, chunks_added=chunks_added)


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if body.file_path and not body.file_path.startswith(UPLOADS_DIR):
        raise HTTPException(
            status_code=400, detail=f"file_path must start with '{UPLOADS_DIR}/'."
        )

    kc = KnowledgeCenter()
    result = await kc.chat(query=body.query, k=body.k, file_path=body.file_path, provider=body.provider)

    if result is None:
        raise HTTPException(
            status_code=500, detail="An error occurred while processing your query."
        )

    return AskResponse(answer=result["answer"], sources=result["sources"])
