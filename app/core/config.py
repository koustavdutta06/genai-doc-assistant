from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "enterprise_docs"

    upload_dir: str = "./data/uploads"
    allowed_extensions: set[str] = {".pdf", ".txt", ".csv", ".xlsx"}
    max_upload_mb: int = 25

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_results: int = 4

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
