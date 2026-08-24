from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source: str
    doc_type: str
    page: int | None = None
    row: int | None = None
    sheet: str | None = None
    chunk_index: int
    snippet: str
    score: float


class UploadFileResult(BaseModel):
    filename: str
    status: str
    chunks_added: int = 0
    error: str | None = None


class UploadDocumentsResponse(BaseModel):
    results: list[UploadFileResult]


class AskQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = None


class AskQuestionResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    is_grounded: bool
    warnings: list[str] = []


class HealthCheckResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chroma_ready: bool
