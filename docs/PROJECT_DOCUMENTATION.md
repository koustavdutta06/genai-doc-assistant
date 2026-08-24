# Enterprise GenAI Doc Assistant — Project Documentation

## 1. Overview

The Enterprise GenAI Doc Assistant is a Generative AI + RAG + Agentic AI system that lets users upload enterprise documents (PDF, TXT, CSV, XLSX), ask natural-language questions about them, and receive grounded, cited answers. It is built around an explicit multi-step agent — implemented as a LangGraph state machine — that plans, retrieves, reasons, and validates before returning an answer, running entirely on local models via Ollama.

**Goals demonstrated by this project:**
- End-to-end document ingestion across four formats
- Semantic retrieval via a persistent vector database (ChromaDB)
- Agentic orchestration: planning, tool selection, reasoning, and self-validation
- Reliability mechanisms: input-safety guardrails, groundedness checking, deterministic correction of LLM arithmetic errors
- A REST API (FastAPI) and a web UI (Streamlit), containerized for deployment

## 2. High-Level Architecture

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

```mermaid
flowchart TD
    U[User uploads document] --> P[Parse: PyPDFLoader / pandas / raw text]
    P --> C[Chunk: RecursiveCharacterTextSplitter]
    C --> E[Embed: Ollama nomic-embed-text]
    E --> V[(ChromaDB\npersistent vector store)]
    P -.raw file kept.-> RAW[(data/uploads/\noriginal CSV & XLSX)]

    Q[User question] --> SC{safety_check}
    SC -- unsafe --> REF[Refusal response]
    SC -- safe --> PL[plan: refine query,\ndecide route]
    PL -- narrative question --> RET[retrieve: semantic\nsearch over ChromaDB]
    PL -- tabular filter / aggregate --> TQ[tabular_query: exact\npandas filter/count/sum/mean]
    RET --> RE[reason: generate\ncited answer]
    TQ --> RE
    RE --> VA{validate}
    VA -- aggregate value missing\nfrom answer --> FIX[deterministic correction]
    VA -- ungrounded, retries left --> RE
    VA -- grounded / retries exhausted --> ANS[Answer + sources + warnings]
    FIX --> ANS
```

## 3. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| REST API | FastAPI + Uvicorn | Async, typed, auto-generated Swagger docs |
| Web UI | Streamlit | Fast to build a chat + upload interface without a separate frontend stack |
| Agent orchestration | LangGraph (`StateGraph`) | Explicit, inspectable state machine rather than an opaque agent loop — each step (plan/retrieve/reason/validate) is a visible node with typed state |
| LLM (chat/reasoning) | Ollama, `llama3.2:3b` | Runs fully locally, no API cost/key. Sized to fit this machine's 4GB-VRAM GPU (see §9, Design Decisions) |
| Embeddings | Ollama, `nomic-embed-text` | Local embedding model, 768-dim, small footprint |
| Vector store | ChromaDB (`PersistentClient`) | Local, persistent, simple metadata filtering |
| Document parsing | `langchain_community.PyPDFLoader` (PDF), `pandas` (CSV/XLSX), stdlib (TXT) | Standard, well-supported loaders per format |
| Structured data queries | `pandas.DataFrame.query()` | Exact, deterministic filtering/aggregation over tabular data — see §6 |
| Config | `pydantic-settings` | Typed, env-driven configuration (`.env`) |
| Testing | `pytest` | Unit coverage for pure-function logic (chunking) |
| Deployment | Docker + Docker Compose | Reproducible container deployment for the app tier |

## 4. Component Breakdown

```
app/
  main.py                   FastAPI app: lifespan startup (creates data dirs), mounts the router
  core/
    config.py                pydantic-settings: all env-driven configuration, cached via get_settings()
    schemas.py                Pydantic request/response models for the API
    llm_client.py             Cached singletons for the Ollama chat model & embeddings client; health check
  services/
    ingestion_service.py      Per-format parsing (PDF/TXT/CSV/XLSX) -> chunk -> embed -> store; entrypoint for uploads
    chunking_service.py       Pure text-splitting logic (no I/O), independently unit-testable
    vector_store_service.py   ChromaDB wrapper: add/query/list/delete against the persistent collection
    tabular_query_service.py  Loads the raw uploaded CSV/XLSX and runs an exact pandas filter/aggregate query
  agents/
    orchestrator.py           Builds and compiles the LangGraph StateGraph; the sole entrypoint `run_agent()`
    planner.py                Refines the question into a search query; decides semantic vs. structured routing
    retriever_agent.py        Semantic search over ChromaDB, exposed as a LangChain @tool
    tabular_agent.py          Structured filter/count/sum/mean query, exposed as a LangChain @tool
    reasoner.py                Builds the grounded-answer prompt and calls the LLM
    validator.py               Deterministic aggregate correction + lexical groundedness check + retry/abstain logic
  utils/
    guardrails.py              Pure functions: input-safety checks, groundedness heuristic
  api/routes/routes.py         FastAPI endpoints: /health-check, /upload-documents, /ask-question
streamlit_app/app.py          Chat + upload UI, calls the FastAPI backend over HTTP
```

## 5. The Agentic Workflow (LangGraph)

The orchestrator (`app/agents/orchestrator.py`) defines a typed `AgentState` (a `TypedDict`) that flows through the graph, accumulating fields as each node runs. It is compiled once (`@lru_cache`) and re-invoked per question.

**Nodes:**

| Node | File | Responsibility |
|---|---|---|
| `safety_check` | `utils/guardrails.py` | Rejects empty/oversized input, prompt-injection patterns (e.g. "ignore all previous instructions"), and a basic unsafe-keyword list |
| `plan` | `agents/planner.py` | Refines the question into a search query, and decides whether to route to semantic search or the structured table tool. The LLM proposes a plan via structured output, but a small local model is unreliable at writing correct query syntax or setting routing flags — so deterministic regex/schema matching (comparison operators, categorical value lookup against the actual table, aggregate keyword detection) overrides or backs up the LLM's own fields for the common cases |
| `retrieve` | `agents/retriever_agent.py` | Semantic similarity search against ChromaDB — used for narrative/document text |
| `tabular_query` | `agents/tabular_agent.py`, `services/tabular_query_service.py` | Exact filter/count/sum/mean over the real uploaded CSV/XLSX via `pandas.DataFrame.query()` — used instead of semantic search when the question needs precise filtering or aggregation |
| `reason` | `agents/reasoner.py` | Builds a numbered-context prompt (citation markers `[1]`, `[2]`, …) and calls the LLM for a grounded answer |
| `validate` | `agents/validator.py` | Two checks, in order: (1) if the tabular tool computed an exact aggregate, verify that value literally appears in the answer — correct it deterministically if not; (2) otherwise check lexical groundedness against retrieved chunks, retry `reason` once if ungrounded, then fall back to an explicit abstention |

**Edges:**

```
START -> safety_check -> (unsafe: END with refusal)
                       -> (safe: plan)
plan -> (tabular routing signal: tabular_query) -> reason
plan -> (narrative question: retrieve) -> reason
reason -> validate -> (ungrounded & retries left: back to reason)
                    -> (grounded, or aggregate corrected, or retries exhausted: END)
```

`run_agent(question, top_k)` is the single function the API layer calls; it invokes the graph and maps the final state into the `AskQuestionResponse` returned to the client.

## 6. Why Two Retrieval Tools?

Semantic vector search retrieves by *meaning*, not by evaluating conditions. A question like *"employees whose salary is more than 100000"* has no reliable relationship to embedding distance — the retriever will confidently return some rows that don't satisfy the condition and miss others that do, because "closeness in embedding space" isn't the same as "satisfies a numeric filter." This was discovered directly during testing: with only semantic search, the same salary question returned an incomplete and partially wrong list.

The fix is architectural, not a prompt tweak: the agent has a second tool, `tabular_query`, that runs the filter/aggregate directly against the real data with pandas — guaranteeing correctness for that class of question — while semantic search continues to handle narrative/unstructured document text. The planner decides which tool to use per question.

## 7. Reliability & Guardrails

- **Input safety** (`safety_check` node): rejects empty/oversized questions, common prompt-injection phrasings, and a basic unsafe-content keyword list before any retrieval happens.
- **Groundedness check** (`validate` node): after the LLM answers, the answer's sentences are checked for lexical word-overlap against the retrieved context; an explicit "I don't have enough information" abstention is always treated as valid (not a failure) rather than penalized.
- **Bounded retry**: if the answer is judged ungrounded, the reasoner is re-invoked once with stricter instructions before the system falls back to an explicit abstention — it never silently returns an unverified answer after exhausting retries.
- **Deterministic aggregate correction**: small local LLMs occasionally recompute a count/sum/average instead of citing the exact value the tabular tool already calculated, and get the arithmetic wrong. Rather than trusting the model, the validator checks whether the exact computed value is present in the answer text and overwrites the answer with a correct, deterministically-generated statement if not.
- **Structured-query sandboxing**: query expressions are restricted to pandas' `DataFrame.query()` boolean-expression grammar (not arbitrary Python), and a regex blocklist rejects tokens like `__`, `import`, `exec(`, `os.`, `subprocess` as defense-in-depth before any expression is evaluated.
- **Temperature = 0**: the chat model runs at zero sampling temperature — for a fact-lookup assistant that must cite sources, deterministic output is strictly preferable to creative variation, and it measurably improved answer consistency during testing (see §9).

## 8. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health-check` | GET | Reports backend status, Ollama reachability, and Chroma readiness |
| `/upload-documents` | POST (multipart `files`) | Parses, chunks, embeds, and indexes one or more PDF/TXT/CSV/XLSX files; returns per-file status and chunk counts |
| `/ask-question` | POST (`{"question": "...", "top_k": 4}`) | Runs the full agent pipeline; returns `{answer, sources, is_grounded, warnings}` |

Interactive Swagger docs are available at `/docs` once the backend is running.

## 9. Design Decisions Log

Key pivots made during development, and why:

- **LLM provider — Anthropic API considered, then Ollama chosen.** The user has a Claude subscription but not separate Anthropic API billing; Ollama (local, free) was used instead, with `langchain-ollama` for both chat and embeddings.
- **Chat model — `llama3.1:8b` planned, `llama3.2:3b` used.** The original plan picked `llama3.1:8b` partly for reliable native tool-calling. In practice this machine's GPU (GTX 1650, 4GB VRAM) OOM'd loading an 8B model. Since the implementation invokes tools manually (not via the LLM's own function-calling), model size mattered more than tool-calling support — `llama3.2:3b` (~2GB) was substituted and fits comfortably.
- **Agent framework — LangChain + LangGraph**, chosen over hand-rolled orchestration for an explicit, inspectable state graph that maps cleanly onto separate planner/retriever/reasoner/validator files.
- **Temperature 0.1 → 0.0.** Initial testing showed the small model's structured-output routing decisions and short-list answers were inconsistent run-to-run at temperature 0.1 (e.g., sometimes dropping a correct row from a 4-item list). Setting temperature to 0 made both routing and answer generation fully deterministic across repeated runs with identical input.
- **Single-tool RAG → dual-tool agent.** Manual testing surfaced that plain semantic search gives wrong answers to filter/aggregate questions over tabular data (see §6) — this drove adding the `tabular_query` tool and the planner's routing logic, including a deterministic layer to compensate for the small model's unreliable structured-output generation (flaky query expressions, occasional SQL syntax instead of pandas, mis-set routing flags).
- **Deterministic aggregate correction.** Even with the tabular tool computing an exact value, the LLM sometimes ignored it and recomputed incorrectly. Rather than relying further on prompting, the validator now checks and corrects this class of error directly.

## 10. Deployment

**Local (native):** see `README.md` → Setup. Requires Ollama running natively with the two models pulled, a Python virtualenv, and `.env` copied from `.env.example`.

**Docker:** see `README.md` → Docker deployment. The FastAPI backend and Streamlit UI run as containers (`docker-compose.yml`); Ollama stays native on the host (GPU passthrough into Docker on Windows would require WSL2 + the NVIDIA Container Toolkit, unnecessary when Ollama already has native GPU support). The backend reaches host Ollama via `http://host.docker.internal:11434`; `data/` is bind-mounted so uploads and the vector DB persist across container restarts.

## 11. Testing

Automated: `pytest tests/` covers `chunking_service` (pure function, no Ollama dependency).

Manual end-to-end verification performed during development:
- Upload → index → grounded Q&A with correct citations (PDF/TXT/CSV)
- Abstention on out-of-scope questions (no hallucinated answer)
- Refusal on prompt-injection input
- Structured filter questions (e.g. "salary more than 100000") — verified against hand-computed expected results
- Structured aggregate questions (count / average) — verified against hand-computed expected results
- Full pipeline re-verified against the containerized (Docker) deployment, confirming identical results to the native run

## 12. Known Limitations

- Guardrails use lexical-overlap heuristics and a keyword/regex blocklist, not a dedicated safety model — adequate for a capstone demo, not production-hardened.
- Single ChromaDB collection shared across all uploaded documents; no per-user or per-session isolation.
- No authentication on the API — intended for local/demo use.
- The tabular tool's deterministic query-building (regex-based comparison/categorical/aggregate detection) covers common single-condition questions well; more complex multi-clause questions (e.g. combining a comparison, a category filter, and an aggregate in one question) rely more heavily on the LLM's own structured output and are less consistently reliable.
- A 3B-parameter local model, even at temperature 0, has a lower ceiling on complex multi-step reasoning than a larger hosted model — this is a deliberate trade-off for zero API cost and full local execution.
