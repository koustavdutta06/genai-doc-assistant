from langchain_core.documents import Document

from app.services.chunking_service import chunk_documents


def test_chunk_documents_assigns_stable_ids():
    docs = [
        Document(page_content="A" * 2500, metadata={"source": "report.pdf", "page": 0}),
        Document(page_content="short row text", metadata={"source": "data.csv", "row": 3}),
    ]

    chunks = chunk_documents(docs)

    assert len(chunks) > len(docs)
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith(chunk.metadata["source"])

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_documents_handles_empty_input():
    assert chunk_documents([]) == []
