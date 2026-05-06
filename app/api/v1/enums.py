from enum import Enum


class LLMProvider(str, Enum):
    openai = "openai"
    finetune = "finetune"


class OCRProvider(str, Enum):
    cloud = "cloud"
    local = "local"


class SupportedFileType(str, Enum):
    pdf = ".pdf"
    txt = ".txt"
    docx = ".docx"
    md = ".md"
