from pathlib import Path

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agents.orchestrator import run_agent
from app.core.config import get_settings
from app.core.llm_client import check_ollama_health
from app.core.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    HealthCheckResponse,
    UploadDocumentsResponse,
    UploadFileResult,
)
from app.services.ingestion_service import ingest_document

router = APIRouter()


@router.get("/health-check", response_model=HealthCheckResponse)
def health_check() -> HealthCheckResponse:
    settings = get_settings()
    chroma_ready = Path(settings.chroma_persist_dir).exists()
    return HealthCheckResponse(
        status="ok",
        ollama_reachable=check_ollama_health(),
        chroma_ready=chroma_ready,
    )


@router.post("/upload-documents", response_model=UploadDocumentsResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadDocumentsResponse:
    results: list[UploadFileResult] = []
    for file in files:
        content = await file.read()
        results.append(ingest_document(file.filename, content))
    return UploadDocumentsResponse(results=results)


@router.post("/ask-question", response_model=AskQuestionResponse)
def ask_question(request: AskQuestionRequest) -> AskQuestionResponse:
    try:
        return run_agent(request.question, top_k=request.top_k)
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama. Make sure it is running (`ollama serve`).") from exc
