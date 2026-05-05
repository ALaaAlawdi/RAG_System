import os
from pydantic_settings import BaseSettings
from pydantic import Field

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    # File storage
    UPLOAD_MAX_BYTES: int = Field(default=20 * 1024 * 1024, env="UPLOAD_MAX_BYTES")
    UPLOADS_DIR:      str = Field(default="uploads",         env="UPLOADS_DIR")
    OUTPUTS_DIR:      str = Field(default="outputs",         env="OUTPUTS_DIR")

    # OCR: secrets
    DATALAB_API_KEY: str = Field(..., env="DATALAB_API_KEY")
    
    # OCR: provider URLs
    OCR_LOCAL_URL: str = Field(default="http://localhost:8000/v1",              env="OCR_LOCAL_URL")
    OCR_CLOUD_URL: str = Field(default="https://www.datalab.to/api/v1/marker",  env="OCR_CLOUD_URL")
    OCR_PROVIDER:  str = Field(default="cloud",                                 env="OCR_PROVIDER")

    # OCR: marker parameters
    OCR_OUTPUT_FORMAT:          str  = Field(default="markdown", env="OCR_OUTPUT_FORMAT")
    OCR_LANGS:                  str  = Field(default="ar,en",    env="OCR_LANGS")
    OCR_SKIP_CACHE:             bool = Field(default=False,       env="OCR_SKIP_CACHE")
    OCR_TABLE_ROW_BBOXES:       bool = Field(default=True,        env="OCR_TABLE_ROW_BBOXES")
    OCR_DISABLE_IMAGE_CAPTIONS: bool = Field(default=True,        env="OCR_DISABLE_IMAGE_CAPTIONS")
    OCR_EXTRACT_LINKS:          bool = Field(default=True,        env="OCR_EXTRACT_LINKS")
    OCR_PAGINATE:               bool = Field(default=False,       env="OCR_PAGINATE")
    OCR_KEEP_PAGE_HEADER:       bool = Field(default=False,       env="OCR_KEEP_PAGE_HEADER")
    OCR_KEEP_PAGE_FOOTER:       bool = Field(default=False,       env="OCR_KEEP_PAGE_FOOTER")
    OCR_NEW_BLOCK_TYPES:        bool = Field(default=False,       env="OCR_NEW_BLOCK_TYPES")

    # OCR: polling behaviour
    OCR_POLL_INTERVAL_SECONDS: float = Field(default=3.0, env="OCR_POLL_INTERVAL_SECONDS")
    OCR_MAX_POLL_ATTEMPTS:     int   = Field(default=40,  env="OCR_MAX_POLL_ATTEMPTS")

    @property
    def ocr_url(self) -> str:
        return self.OCR_LOCAL_URL if self.OCR_PROVIDER == "local" else self.OCR_CLOUD_URL

    class Config:
        env_file = ".env"


settings = Settings()

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

