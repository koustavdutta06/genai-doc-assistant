from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        pieces = splitter.split_documents([doc])
        for index, piece in enumerate(pieces):
            source = piece.metadata.get("source", "unknown")
            anchor = piece.metadata.get("page", piece.metadata.get("row", 0))
            piece.metadata["chunk_index"] = index
            piece.metadata["chunk_id"] = f"{source}-{anchor}-{index}"
            chunks.append(piece)

    return chunks
