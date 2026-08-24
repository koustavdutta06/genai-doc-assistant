from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_client import get_chat_model
from app.core.schemas import SourceCitation

_REASONER_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question using ONLY the "
    "numbered context blocks below. Cite the blocks you used with bracketed numbers like [1], [2]. "
    "If a context block states an exact computed result (e.g. 'EXACT COUNT(...) = 4' or "
    "'EXACT MEAN(...) = 123456.78'), use that number directly rather than recomputing or recounting it yourself. "
    "When a question asks to list matching records, list every one of them from the context blocks provided — "
    "do not skip any and do not include ones that were not given to you. "
    "If the context does not contain enough information to answer, say exactly: "
    "\"I don't have enough information in the provided documents to answer this question.\" "
    "Do not use outside knowledge. Respond with ONLY your final answer — never repeat, quote, or restate the "
    "numbered context blocks themselves in your response."
)

_RETRY_SUFFIX = (
    "\n\nYour previous answer was not sufficiently grounded in the context. "
    "Rewrite the answer using only facts explicitly present in the numbered context blocks, "
    "and cite every claim with a bracketed number."
)


def _build_context_block(docs: list[dict]) -> str:
    lines = []
    for idx, doc in enumerate(docs, start=1):
        location = doc.get("page") if doc.get("page") is not None else doc.get("row")
        lines.append(f"[{idx}] (source: {doc.get('source')}, loc: {location})\n{doc.get('content')}")
    return "\n\n".join(lines)


def _build_sources(docs: list[dict]) -> list[SourceCitation]:
    sources = []
    for idx, doc in enumerate(docs, start=1):
        content = doc.get("content", "")
        sources.append(
            SourceCitation(
                source=doc.get("source") or "unknown",
                doc_type=doc.get("doc_type") or "unknown",
                page=doc.get("page"),
                row=doc.get("row"),
                sheet=doc.get("sheet"),
                chunk_index=doc.get("chunk_index") or idx - 1,
                snippet=content[:280],
                score=float(doc.get("score", 0.0)),
            )
        )
    return sources


def reason_node(state: dict) -> dict:
    docs = state.get("retrieved_docs", [])
    context_block = _build_context_block(docs)

    system_prompt = _REASONER_PROMPT
    if state.get("retry_count", 0) > 0:
        system_prompt += _RETRY_SUFFIX

    user_content = f"Context:\n{context_block}\n\nQuestion: {state['question']}"

    model = get_chat_model()
    response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])

    return {
        "answer": response.content.strip(),
        "sources": _build_sources(docs),
    }
