import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.llm_client import get_chat_model
from app.services.tabular_query_service import get_categorical_values, list_tabular_sources

_SQL_KEYWORD_PATTERN = re.compile(r"\b(select|from|where|insert|update|delete)\b", re.IGNORECASE)

_COMPARISON_PATTERNS = [
    (re.compile(r"(?:more than|greater than|over|above|at least)\s*\$?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE), ">"),
    (re.compile(r"(?:less than|under|below|at most)\s*\$?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE), "<"),
    (re.compile(r"(?:equal to|exactly)\s*\$?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE), "=="),
]

_AGGREGATE_OP_PATTERNS = [
    (re.compile(r"\b(?:average|mean)\b", re.IGNORECASE), "mean"),
    (re.compile(r"\b(?:total|sum)\b", re.IGNORECASE), "sum"),
    (re.compile(r"\b(?:how many|count|number of)\b", re.IGNORECASE), "count"),
]

_STOP_WORD_SPLIT = re.compile(r"\b(?:of|in|for|where|by)\b", re.IGNORECASE)

_FEW_SHOT = (
    "Example: table 'employees.csv' has columns ['name', 'department', 'annual_salary_usd']. "
    'Question: "who has a salary over 100000?" -> tabular_query_expr = "annual_salary_usd > 100000" '
    "(pandas boolean expression, NEVER SQL syntax like SELECT/FROM/WHERE)."
)


class QueryPlan(BaseModel):
    search_query: str = Field(description="Refined, keyword-rich semantic search query for narrative/document text retrieval.")
    use_tabular_query: bool = Field(
        description="True only if the question requires filtering, counting, or aggregating rows in one of the listed tables."
    )
    tabular_source: str | None = Field(
        default=None, description="Exact filename (from the list provided) to query, required if use_tabular_query is true."
    )
    tabular_query_expr: str = Field(
        default="",
        description='A pandas DataFrame.query() boolean expression using exact column names, e.g. "annual_salary_usd > 100000". Empty string selects all rows. Never SQL.',
    )
    tabular_aggregate_op: Literal["none", "count", "sum", "mean"] = Field(
        default="none", description="Set to count/sum/mean if the question asks for a total, count, or average."
    )
    tabular_aggregate_column: str | None = Field(
        default=None, description="Exact column name to aggregate, required when tabular_aggregate_op is sum or mean."
    )


def _guess_column(text: str, columns: list[str]) -> str | None:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    candidates = [col for col in columns if set(re.findall(r"[a-zA-Z]+", col.lower())) & words]
    return candidates[0] if len(candidates) == 1 else None


def _guess_comparison_filter(question: str, columns: list[str]) -> str | None:
    column = _guess_column(question, columns)
    if not column:
        return None
    for pattern, operator in _COMPARISON_PATTERNS:
        match = pattern.search(question)
        if match:
            value = match.group(1).replace(",", "")
            return f"{column} {operator} {value}"
    return None


def _guess_categorical_filter(question: str, source: str) -> str | None:
    for column, values in get_categorical_values(source).items():
        for value in values:
            if re.search(rf"\b{re.escape(value)}\b", question, re.IGNORECASE):
                escaped = value.replace("'", "\\'")
                return f"{column} == '{escaped}'"
    return None


def _guess_aggregate(question: str, columns: list[str]) -> tuple[str, str | None]:
    for pattern, op in _AGGREGATE_OP_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        if op == "count":
            return "count", None
        tail = _STOP_WORD_SPLIT.split(question[match.end() :], maxsplit=1)[0]
        column = _guess_column(tail, columns) or _guess_column(question, columns)
        return op, column
    return "none", None


def _build_system_prompt(tables: dict[str, list[str]]) -> str:
    if not tables:
        table_info = "No structured (CSV/XLSX) tables are currently available."
    else:
        table_info = "\n".join(f"- {name}: columns = {cols}" for name, cols in tables.items())

    return (
        "You plan how to answer questions against an enterprise knowledge base. There are two retrieval modes: "
        "(1) semantic search over document text, and (2) a structured table query tool for exact filtering, "
        "counting, or aggregating over CSV/XLSX tables.\n\n"
        f"Available structured tables:\n{table_info}\n\n"
        "Use the structured table query ONLY when the question involves a numeric or categorical filter, count, "
        "sum, or average over one of the tables listed above, and only reference columns that actually exist in "
        f"that table. Otherwise set use_tabular_query to false and just refine the question into a search query.\n\n{_FEW_SHOT}"
    )


def plan_node(state: dict) -> dict:
    question = state["question"]
    tables = list_tabular_sources()

    model = get_chat_model().with_structured_output(QueryPlan)
    plan: QueryPlan = model.invoke([SystemMessage(content=_build_system_prompt(tables)), HumanMessage(content=question)])

    tabular_source = plan.tabular_source if plan.tabular_source in tables else None
    if tabular_source is None and len(tables) == 1:
        tabular_source = next(iter(tables))

    query_expr = plan.tabular_query_expr or ""
    if _SQL_KEYWORD_PATTERN.search(query_expr):
        query_expr = ""
    aggregate_op = plan.tabular_aggregate_op
    aggregate_column = plan.tabular_aggregate_column

    categorical_hit = False
    if tabular_source:
        columns = tables[tabular_source]

        # The LLM's structured-output fields are sampled and can be flaky (empty/garbage expressions,
        # wrong aggregate op) on a small local model. Prefer deterministic regex/lookup-derived values
        # for the common cases (numeric comparison, categorical equality, simple aggregate); fall back
        # to the LLM's own fields when no deterministic signal is found.
        filters = [f for f in (_guess_comparison_filter(question, columns), _guess_categorical_filter(question, tabular_source)) if f]
        categorical_hit = bool(filters)
        if filters:
            query_expr = " and ".join(filters)

        det_agg_op, det_agg_column = _guess_aggregate(question, columns)
        if det_agg_op != "none":
            aggregate_op = det_agg_op
            aggregate_column = det_agg_column or aggregate_column

    # Require concrete grounding evidence (a matched filter, or a resolved sum/mean column) before
    # deterministically forcing the tabular route — a bare intent keyword like "how many" or
    # "average" isn't specific to the table and would misroute unrelated narrative questions
    # (e.g. "how many days of leave do employees get" has nothing to do with employees.csv).
    has_resolved_aggregate = aggregate_op in ("sum", "mean") and bool(aggregate_column)
    deterministic_signal = categorical_hit or has_resolved_aggregate
    use_tabular_query = bool(tabular_source) and (plan.use_tabular_query or deterministic_signal)

    return {
        "search_query": plan.search_query or question,
        "use_tabular_query": use_tabular_query,
        "tabular_source": tabular_source,
        "tabular_query_expr": query_expr,
        "tabular_aggregate_op": aggregate_op,
        "tabular_aggregate_column": aggregate_column,
        "retry_count": 0,
    }
