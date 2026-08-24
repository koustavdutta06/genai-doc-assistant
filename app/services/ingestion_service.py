from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.schemas import UploadFileResult
from app.services.chunking_service import chunk_documents
from app.services.vector_store_service import add_documents


def _row_to_text(row: pd.Series) -> str:
    return " | ".join(f"{col}: {val}" for col, val in row.items())


def _parse_pdf(path: Path) -> list[Document]:
    return PyPDFLoader(str(path)).load()


def _parse_txt(path: Path, filename: str) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": filename, "doc_type": "txt"})]


def _parse_csv(path: Path, filename: str) -> list[Document]:
    df = pd.read_csv(path)
    return [
        Document(
            page_content=_row_to_text(row),
            metadata={"source": filename, "doc_type": "csv", "row": idx},
        )
        for idx, row in df.iterrows()
    ]


def _parse_xlsx(path: Path, filename: str) -> list[Document]:
    sheets = pd.read_excel(path, sheet_name=None)
    documents = []
    for sheet_name, df in sheets.items():
        for idx, row in df.iterrows():
            documents.append(
                Document(
                    page_content=_row_to_text(row),
                    metadata={"source": filename, "doc_type": "xlsx", "sheet": sheet_name, "row": idx},
                )
            )
    return documents


_PARSERS = {
    ".txt": lambda path, filename: _parse_txt(path, filename),
    ".csv": lambda path, filename: _parse_csv(path, filename),
    ".xlsx": lambda path, filename: _parse_xlsx(path, filename),
    ".pdf": lambda path, filename: _parse_pdf(path),
}


def ingest_document(filename: str, content: bytes) -> UploadFileResult:
    settings = get_settings()
    extension = Path(filename).suffix.lower()

    if extension not in settings.allowed_extensions:
        return UploadFileResult(filename=filename, status="rejected", error=f"Unsupported file type: {extension}")

    if len(content) > settings.max_upload_mb * 1024 * 1024:
        return UploadFileResult(filename=filename, status="rejected", error="File exceeds max upload size")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / filename
    dest_path.write_bytes(content)

    try:
        documents = _PARSERS[extension](dest_path, filename)
        if not documents:
            return UploadFileResult(filename=filename, status="failed", error="No content extracted")

        chunks = chunk_documents(documents)
        chunks_added = add_documents(chunks)
        return UploadFileResult(filename=filename, status="indexed", chunks_added=chunks_added)
    except Exception as exc:  # noqa: BLE001 - surfaced per-file to the caller, not swallowed
        return UploadFileResult(filename=filename, status="failed", error=str(exc))
