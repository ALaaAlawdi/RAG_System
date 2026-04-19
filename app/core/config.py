import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"


settings = Settings()

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

