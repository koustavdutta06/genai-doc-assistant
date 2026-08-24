# Enterprise GenAI Doc Assistant

A Generative AI + RAG + Agentic AI capstone project. Upload enterprise documents (PDF, TXT, CSV, XLSX), ask natural-language questions, and get grounded answers with citations, produced by a LangGraph agent pipeline running entirely on local models via Ollama.

## Architecture

```
Upload (PDF/TXT/CSV/XLSX)
      |
Parse & extract text (per-format loaders)
      |
Chunk (RecursiveCharacterTextSplitter)
      |
Embed (Ollama: nomic-embed-text)
      |
Store (ChromaDB, persistent)   +   raw file kept in data/uploads/ (for exact structured queries)
      |
Question -> [safety_check] -> [plan] -+-> [retrieve]       (semantic search)      -+-> [reason] -> [validate] -> answer + sources
                  |                    |                                          |        |
              reject unsafe            +-> [tabular_query]  (exact pandas filter/  +   ungrounded -> retry reason (max 1x)
              input -> refusal             count/sum/mean over the source CSV/XLSX)     else -> abstain
```

The agent graph (`app/agents/orchestrator.py`) is a LangGraph `StateGraph`:

| Node | File | Responsibility |
|---|---|---|
| `safety_check` | `app/utils/guardrails.py` | Rejects empty/oversized input, prompt-injection patterns, unsafe keywords |
| `plan` | `app/agents/planner.py` | Refines the question into a search query, and decides whether to route to semantic search or the structured table tool (deterministic regex/schema matching backs up the LLM's own routing decision, since a small local model is unreliable at this alone) |
| `retrieve` | `app/agents/retriever_agent.py` | Similarity search against ChromaDB (exposed as a LangChain `@tool`) — used for narrative/document text |
| `tabular_query` | `app/agents/tabular_agent.py`, `app/services/tabular_query_service.py` | Exact filter/count/sum/mean over the actual uploaded CSV/XLSX via `pandas.DataFrame.query()` — used instead of semantic search when the question needs precise filtering or aggregation over tabular data, which embedding similarity can't do reliably |
| `reason` | `app/agents/reasoner.py` | Generates a grounded, cited answer from retrieved context |
| `validate` | `app/agents/validator.py` | Deterministically corrects the answer if an exact computed aggregate value isn't present in it (small models sometimes recompute instead of citing the given number); otherwise checks lexical grounding against retrieved chunks, retries `reason` once if ungrounded, then falls back to an explicit abstention |

**Why two retrieval tools?** Semantic vector search retrieves by meaning, not by evaluating conditions — a query like "salary > 100000" has no reliable relationship to embedding distance, so plain RAG over tabular data silently returns wrong answers (it will confidently retrieve some rows that don't match and miss others that do). The `tabular_query` tool sidesteps this by running the filter/aggregate directly against the real data with pandas, guaranteeing correctness for that class of question, while semantic search still handles narrative/document text.

## Model choices

- **Chat/reasoning**: `llama3.2:3b` via Ollama. Originally planned as `llama3.1:8b`, but this machine's GPU (GTX 1650, 4GB VRAM) can't fit an 8B model — `llama3.2:3b` (~2GB) runs comfortably. Swap via `OLLAMA_CHAT_MODEL` in `.env` if you have more VRAM.
- **Embeddings**: `nomic-embed-text` via Ollama (274MB, 768-dim).

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
   Make sure Ollama is running (`ollama serve`, or the tray app) before starting the backend.

2. Create a virtualenv and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\pip install -r requirement.txt
   ```

3. Copy `.env.example` to `.env` and adjust if needed.

4. Run the backend:
   ```
   venv\Scripts\python -m uvicorn app.main:app --reload
   ```

5. In a second terminal, run the UI:
   ```
   venv\Scripts\python -m streamlit run streamlit_app/app.py --server.baseUrlPath doc-assistant
   ```

6. Open http://localhost:8501/doc-assistant, upload a document in the sidebar, then ask a question in the chat.

## Docker deployment

The FastAPI backend and Streamlit UI run in containers; **Ollama stays native on the host** (GPU passthrough into Docker on Windows requires WSL2 + the NVIDIA Container Toolkit, which is unnecessary complexity when Ollama already runs natively with GPU support).

1. Install Ollama and pull the models on the **host** as in Setup step 1 above, and make sure it's running.
2. From the project root:
   ```
   docker compose up --build
   ```
3. Open http://localhost:8501/doc-assistant. The backend is at http://localhost:8000 (`/docs` for Swagger).

Notes:
- `data/` is bind-mounted into the backend container, so uploads and the Chroma DB persist on the host across `docker compose down`/`up`.
- The backend reaches host-side Ollama via `http://host.docker.internal:11434` (works out of the box with Docker Desktop on Windows/Mac). If it can't connect, Ollama may only be bound to `127.0.0.1` — set `OLLAMA_HOST=0.0.0.0` in the Ollama service's environment and restart it.
- The frontend reaches the backend via the Docker Compose network (`http://backend:8000`), not `localhost`.
- To change models, edit the `environment:` block for `backend` in `docker-compose.yml` (`OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`) — same variable names as `.env`.

## API

- `GET /health-check` — reports backend status, Ollama reachability, Chroma readiness.
- `POST /upload-documents` — multipart file upload (`files`), supports PDF/TXT/CSV/XLSX. Returns per-file indexing status and chunk counts.
- `POST /ask-question` — `{"question": "...", "top_k": 4}` → grounded answer, source citations, groundedness flag, warnings.

Interactive docs at `/docs` (Swagger UI) once the backend is running.

## Testing

```
venv\Scripts\python -m pytest tests/
```

Covers `chunking_service` (pure function, no Ollama required). The agentic pipeline and ingestion were verified manually end-to-end (upload -> index -> grounded Q&A -> abstention on out-of-scope questions -> refusal on prompt-injection input).

## Known limitations

- Guardrails (`app/utils/guardrails.py`) use lexical overlap heuristics and a keyword/regex blocklist, not a dedicated safety model — adequate for a capstone demo, not production-hardened.
- Single ChromaDB collection shared across all uploaded documents; no per-user isolation.
- No authentication on the API — intended for local/demo use.
