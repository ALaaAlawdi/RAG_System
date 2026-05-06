import asyncio
import httpx
from typing import Optional

from ..core.config import settings
from ..core.logger import setup_logger
from ..schemas.ocr import OCRSubmitResponse, OCRStatusResponse
from ..api.v1.constants import (
    OCR_OUTPUT_FORMAT,
    OCR_LANGS,
    OCR_SKIP_CACHE,
    OCR_TABLE_ROW_BBOXES,
    OCR_DISABLE_IMAGE_CAPTIONS,
    OCR_EXTRACT_LINKS,
    OCR_PAGINATE,
    OCR_KEEP_PAGE_HEADER,
    OCR_KEEP_PAGE_FOOTER,
    OCR_NEW_BLOCK_TYPES,
    OCR_POLL_INTERVAL_SECONDS,
    OCR_MAX_POLL_ATTEMPTS,
)

logger = setup_logger(__name__)


class OCRAPIError(Exception):
    pass


class OCRTimeoutError(Exception):
    pass


class OCRClient:
    def __init__(self) -> None:
        self._headers = {"X-Api-Key": settings.DATALAB_API_KEY}

    def _build_payload(self, output_format: str, langs: str) -> dict:
        def b(val: bool) -> str:
            return str(val).lower()

        return {
            "output_format":          output_format,
            "langs":                  langs,
            "skip_cache":             b(OCR_SKIP_CACHE),
            "table_row_bboxes":       b(OCR_TABLE_ROW_BBOXES),
            "disable_image_captions": b(OCR_DISABLE_IMAGE_CAPTIONS),
            "extract_links":          b(OCR_EXTRACT_LINKS),
            "paginate":               b(OCR_PAGINATE),
            "keep_page_header":       b(OCR_KEEP_PAGE_HEADER),
            "keep_page_footer":       b(OCR_KEEP_PAGE_FOOTER),
            "new_block_types":        b(OCR_NEW_BLOCK_TYPES),
        }

    async def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        output_format: Optional[str] = None,
        langs: Optional[str] = None,
    ) -> OCRStatusResponse:
        resolved_format = output_format or OCR_OUTPUT_FORMAT
        resolved_langs  = langs         or OCR_LANGS
        url             = settings.ocr_url

        logger.info(f"[OCR] Provider='{settings.OCR_PROVIDER}', URL='{url}'")
        submit = await self._submit(file_bytes, filename, resolved_format, resolved_langs, url)
        return await self._poll(submit.request_check_url)

    async def _submit(
        self,
        file_bytes: bytes,
        filename: str,
        output_format: str,
        langs: str,
        url: str,
    ) -> OCRSubmitResponse:
        logger.info(f"[OCR] Submitting '{filename}' ({len(file_bytes)} bytes)")
        payload = self._build_payload(output_format, langs)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self._headers,
                data=payload,
                files={"file": (filename, file_bytes, "application/pdf")},
                timeout=60.0,
            )
            response.raise_for_status()

        data = OCRSubmitResponse.model_validate(response.json())
        if not data.success:
            logger.error(f"[OCR] Submit failed: {data.error}")
            raise OCRAPIError(f"Datalab submit error: {data.error}")

        logger.info(f"[OCR] Submitted. Polling: {data.request_check_url}")
        return data

    async def _poll(self, check_url: str) -> OCRStatusResponse:
        async with httpx.AsyncClient() as client:
            for attempt in range(1, OCR_MAX_POLL_ATTEMPTS + 1):
                logger.debug(f"[OCR] Poll {attempt}/{OCR_MAX_POLL_ATTEMPTS}")
                response = await client.get(check_url, headers=self._headers, timeout=30.0)
                response.raise_for_status()

                data = OCRStatusResponse.model_validate(response.json())

                if data.status == "complete":
                    logger.info("[OCR] Processing complete")
                    return data

                if data.status == "error":
                    logger.error(f"[OCR] Error: {data.error}")
                    raise OCRAPIError(f"Datalab processing error: {data.error}")

                logger.debug(f"[OCR] Status='{data.status}', sleeping {OCR_POLL_INTERVAL_SECONDS}s")
                await asyncio.sleep(OCR_POLL_INTERVAL_SECONDS)

        raise OCRTimeoutError(
            f"OCR polling timed out after {OCR_MAX_POLL_ATTEMPTS} attempts "
            f"({OCR_MAX_POLL_ATTEMPTS * OCR_POLL_INTERVAL_SECONDS:.0f}s total)"
        )
