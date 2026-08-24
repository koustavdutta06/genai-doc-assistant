from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.llm_client import get_embeddings


def _clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
    )


def add_documents(chunks: list[Document]) -> int:
    if not chunks:
        return 0

    store = get_vector_store()
    ids = []
    for chunk in chunks:
        chunk.metadata = _clean_metadata(chunk.metadata)
        ids.append(chunk.metadata["chunk_id"])

    store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def similarity_search_with_score(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    settings = get_settings()
    store = get_vector_store()
    return store.similarity_search_with_score(query, k=k or settings.top_k_results)


def list_sources() -> list[str]:
    store = get_vector_store()
    data = store.get(include=["metadatas"])
    sources = {metadata.get("source") for metadata in data.get("metadatas", []) if metadata.get("source")}
    return sorted(sources)


def delete_by_source(filename: str) -> None:
    store = get_vector_store()
    store.delete(where={"source": filename})
