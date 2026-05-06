import os
from pydantic_settings import BaseSettings
from pydantic import Field

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".docx", ".md"})

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

    @property
    def ocr_url(self) -> str:
        return self.OCR_LOCAL_URL if self.OCR_PROVIDER == "local" else self.OCR_CLOUD_URL

    @property
    def OCR_INPUTS_DIR(self) -> str:
        return os.path.join(self.UPLOADS_DIR, "ocr")

    @property
    def OCR_OUTPUTS_DIR(self) -> str:
        return os.path.join(self.OUTPUTS_DIR, "ocr")

    class Config:
        env_file = ".env"


settings = Settings()

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.OCR_INPUTS_DIR, exist_ok=True)
os.makedirs(settings.OCR_OUTPUTS_DIR, exist_ok=True)