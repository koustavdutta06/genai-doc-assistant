from functools import lru_cache

import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import get_settings


@lru_cache
def get_chat_model() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
    )


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )


def check_ollama_health() -> bool:
    settings = get_settings()
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False
