from app.utils.guardrails import check_grounding

MAX_RETRIES = 1


def _number_variants(value: float) -> list[str]:
    variants = {str(value), f"{value:.2f}", f"{value:,.2f}"}
    if value == int(value):
        variants.update({str(int(value)), f"{int(value):,}"})
    return list(variants)


def _aggregate_correction(state: dict) -> dict | None:
    """Small local models sometimes recompute an aggregate instead of using the exact value the
    tabular tool already calculated (and get it wrong). Deterministically override the answer if
    the exact computed value isn't present in it, rather than trusting the LLM's arithmetic."""
    tabular_result = state.get("tabular_result")
    if not tabular_result or tabular_result.get("aggregate_op") in (None, "none"):
        return None

    value = tabular_result.get("aggregate_value")
    if value is None:
        return None

    answer = state.get("answer", "")
    if any(variant in answer for variant in _number_variants(value)):
        return None

    op = tabular_result["aggregate_op"]
    source = state.get("tabular_source")
    if op == "count":
        text = f"The exact count is {int(value)} (out of {tabular_result.get('total_rows')} total rows in {source})."
    else:
        column = tabular_result.get("aggregate_column")
        text = (
            f"The exact {op} of {column} over the {tabular_result.get('row_count')} matched rows "
            f"in {source} is {value:,.2f}."
        )

    return {"answer": text, "is_grounded": True, "should_retry": False}


def validate_node(state: dict) -> dict:
    correction = _aggregate_correction(state)
    if correction:
        return correction

    docs = state.get("retrieved_docs", [])
    context_chunks = [doc.get("content", "") for doc in docs]
    is_grounded = check_grounding(state.get("answer", ""), context_chunks)

    if is_grounded:
        return {"is_grounded": True, "should_retry": False}

    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        return {"is_grounded": False, "retry_count": retry_count + 1, "should_retry": True}

    warnings = list(state.get("warnings", []))
    warnings.append("The answer could not be fully verified against the retrieved documents.")
    return {
        "is_grounded": False,
        "should_retry": False,
        "answer": "I don't have enough information in the provided documents to answer this question confidently.",
        "warnings": warnings,
    }


def should_retry(state: dict) -> str:
    return "retry" if state.get("should_retry") else "end"
