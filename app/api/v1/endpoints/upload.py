from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from ....knowledge_center.center import KnowledgeCenter
from ....knowledge_center.document_reader import read_document, read_url
from ....schemas.upload import UploadResponse
from ....core.config import settings, ALLOWED_EXTENSIONS
from ....core.logger import setup_logger
from ....services.ocr_client import OCRAPIError, OCRTimeoutError
from ..utils import save_upload
from ...dependencies import get_knowledge_center

logger = setup_logger(__name__)
router = APIRouter()


@router.post("/", response_model=UploadResponse)
async def upload(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    kc: KnowledgeCenter = Depends(get_knowledge_center),
):
    if file is None and not url:
        raise HTTPException(status_code=400, detail="Provide a file or a URL.")

    try:
        if url:
            logger.info(f"[UPLOAD] Ingesting URL: {url}")
            documents = await read_url(url)
        else:
            original_name = file.filename or "file"
            _, ext = original_name.rsplit(".", 1) if "." in original_name else (original_name, "")
            ext = f".{ext}".lower()

            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )

            content = await file.read()

            if len(content) > settings.UPLOAD_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {settings.UPLOAD_MAX_BYTES // (1024 * 1024)} MB.",
                )

            save_path = save_upload(content, original_name, settings.UPLOADS_DIR)
            logger.info(f"[UPLOAD] Saved '{original_name}' to '{save_path}'")
            documents = await read_document(save_path)
        
        success, message, chunks_added = await kc.add_document(documents)
        if not success:
            raise HTTPException(status_code=500, detail=message)

        return UploadResponse(success=True, message=message, chunks_added=chunks_added)

    except HTTPException:
        raise
    except OCRAPIError as e:
        logger.error(f"[UPLOAD] OCR API error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except OCRTimeoutError as e:
        logger.warning(f"[UPLOAD] OCR timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.error(f"[UPLOAD] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
