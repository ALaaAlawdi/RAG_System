DEFAULT_K: int = 4
MAX_QUERY_LENGTH: int = 2000

# OCR marker parameters
OCR_OUTPUT_FORMAT: str = "markdown"
OCR_LANGS: str = "ar,en"
OCR_SKIP_CACHE: bool = False
OCR_TABLE_ROW_BBOXES: bool = True
OCR_DISABLE_IMAGE_CAPTIONS: bool = True
OCR_EXTRACT_LINKS: bool = True
OCR_PAGINATE: bool = False
OCR_KEEP_PAGE_HEADER: bool = False
OCR_KEEP_PAGE_FOOTER: bool = False
OCR_NEW_BLOCK_TYPES: bool = False

# OCR polling
OCR_POLL_INTERVAL_SECONDS: float = 3.0
OCR_MAX_POLL_ATTEMPTS: int = 40
