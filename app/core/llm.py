from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from .logger import setup_logger

logger = setup_logger(__name__)

_OLLAMA_MODEL = "hf.co/AlaaAlawdi/llama_finetune"

_llm_openai = ChatOpenAI(model="gpt-4o")
_llm_ollama = ChatOllama(model=_OLLAMA_MODEL, temperature=0)


def get_llm(provider: str):
    if provider == "finetune":
        logger.debug(f"[LLM] Using Ollama model: {_OLLAMA_MODEL}")
        return _llm_ollama
    logger.debug("[LLM] Using OpenAI gpt-4o")
    return _llm_openai
