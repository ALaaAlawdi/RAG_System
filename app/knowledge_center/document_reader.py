import os
from pathlib import Path
from langchain_core.documents import Document

from .setup import read_txt, read_docx, read_markdown, fetch_url, get_text_chunks
from ..core.config import settings
from ..core.logger import setup_logger
from ..services.ocr_client import OCRClient

logger = setup_logger(__name__)


async def _ocr_and_chunk(pdf_path: str) -> list[Document]:
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    filename = os.path.basename(pdf_path)
    client = OCRClient()
    result = await client.process_document(file_bytes=file_bytes, filename=filename)

    markdown_text = result.markdown or ""
    output_path = os.path.join(settings.OCR_OUTPUTS_DIR, f"{Path(pdf_path).stem}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    logger.info(f"[OCR] Saved markdown to '{output_path}'")

    chunks = await get_text_chunks(markdown_text)
    return [
        Document(page_content=chunk, metadata={"source": pdf_path, "page": 1, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]


async def read_document(path: str) -> list[Document]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return await _ocr_and_chunk(path)
    elif ext == ".txt":
        return await read_txt(path)
    elif ext == ".docx":
        return await read_docx(path)
    elif ext == ".md":
        return await read_markdown(path)
    raise ValueError(f"Unsupported file type: {ext}")


async def read_url(url: str) -> list[Document]:
    return await fetch_url(url)