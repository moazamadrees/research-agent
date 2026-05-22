from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-5-20250929"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "research_papers"

    confidence_threshold: float = 0.65
    max_retrieval_docs: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
