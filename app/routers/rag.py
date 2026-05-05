import os
import uuid
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from ..knowledge_center.center import KnowledgeCenter, UPLOADS_DIR
from ..schemas.rag import AskRequest, AskResponse, UploadResponse
from ..schemas.ocr import OCRResponse
from ..services.ocr_client import OCRClient, OCRAPIError, OCRTimeoutError
from ..core.config import settings, ALLOWED_EXTENSIONS
from ..core.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):

    original_name = file.filename or "file"
    stem, ext = os.path.splitext(original_name)
    ext = ext.lower()   

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail="Only PDF and TXT files are supported."
        )
    
    content = await file.read()
    
    if len(content) > settings.UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size is {settings.UPLOAD_MAX_BYTES // (1024*1024)} MB."
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


@router.post("/ocr", response_model=OCRResponse)
async def ocr_document(
    file: UploadFile = File(...),
    output_format: str = Form(default="markdown"),
    langs: str = Form(default="ar,en"),
):  
    original_name = file.filename or "file"
    _, ext = os.path.splitext(original_name)

    if ext.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported for OCR.")

    content = await file.read()
    if len(content) > settings.UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.UPLOAD_MAX_BYTES // (1024 * 1024)} MB.",
        )

    stem, _ = os.path.splitext(original_name)
    unique_stem = f"{stem}_{uuid.uuid4().hex[:8]}"

    ocr_inputs_dir  = os.path.join(settings.UPLOADS_DIR, "ocr")
    ocr_outputs_dir = os.path.join(settings.OUTPUTS_DIR, "ocr")

    os.makedirs(ocr_inputs_dir,  exist_ok=True)
    os.makedirs(ocr_outputs_dir, exist_ok=True)

    input_path  = os.path.join(ocr_inputs_dir,  f"{unique_stem}.pdf")
    output_path = os.path.join(ocr_outputs_dir, f"{unique_stem}.md")

    try:
        with open(input_path, "wb") as f:
            f.write(content)
        logger.info(f"[OCR] Saved input to '{input_path}'")
    except Exception as e:
        logger.error(f"[OCR] Failed to save input file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    logger.info(f"[OCR] Received '{original_name}' ({len(content)} bytes)")

    try:
        client = OCRClient()
        result = await client.process_document(
            file_bytes=content,
            filename=original_name,
            output_format=output_format,
            langs=langs,
        )

        markdown_text = result.markdown or ""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            logger.info(f"[OCR] Saved output to '{output_path}'")
        except Exception as e:
            logger.error(f"[OCR] Failed to save output file: {e}")
            raise HTTPException(status_code=500, detail="OCR succeeded but failed to save output.")

        return OCRResponse(
            success=True,
            provider=settings.OCR_PROVIDER,
            markdown=markdown_text,
            message="OCR completed successfully.",
            input_path=input_path,
            output_path=output_path,
        )
    except OCRAPIError as e:
        logger.error(f"[OCR] API error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except OCRTimeoutError as e:
        logger.warning(f"[OCR] Timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"[OCR] HTTP error: {e}")
        raise HTTPException(status_code=502, detail="Upstream OCR service error.")