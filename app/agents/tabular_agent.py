from langchain_core.tools import tool

from app.services.tabular_query_service import run_tabular_query


@tool
def query_tabular_data(
    source: str,
    query_expr: str = "",
    aggregate_op: str = "none",
    aggregate_column: str | None = None,
) -> dict:
    """Run an exact filter/count/sum/mean query against an uploaded CSV or XLSX file."""
    return run_tabular_query(source, query_expr or None, aggregate_op, aggregate_column)


def _row_to_content(row: dict) -> str:
    return " | ".join(f"{k}: {v}" for k, v in row.items() if k != "_sheet")


def tabular_query_node(state: dict) -> dict:
    result = query_tabular_data.invoke(
        {
            "source": state["tabular_source"],
            "query_expr": state.get("tabular_query_expr") or "",
            "aggregate_op": state.get("tabular_aggregate_op", "none"),
            "aggregate_column": state.get("tabular_aggregate_column"),
        }
    )

    if result.get("error"):
        warnings = list(state.get("warnings", []))
        warnings.append(f"Structured query failed: {result['error']}")
        return {"retrieved_docs": [], "warnings": warnings, "tabular_result": None}

    source = state["tabular_source"]
    header = f"Structured query over {source}: {result['row_count']} of {result['total_rows']} rows matched."
    if result.get("aggregate_op") and result["aggregate_op"] != "none":
        header += f" EXACT {result['aggregate_op'].upper()}({result.get('aggregate_column')}) = {result.get('aggregate_value')}."

    docs = [
        {
            "content": header,
            "source": source,
            "doc_type": "tabular_summary",
            "page": None,
            "row": None,
            "sheet": None,
            "chunk_index": 0,
            "chunk_id": f"{source}-summary",
            "score": 1.0,
        }
    ]

    for position, (row, row_index) in enumerate(zip(result["rows"], result["row_indices"]), start=1):
        docs.append(
            {
                "content": _row_to_content(row),
                "source": source,
                "doc_type": "tabular_row",
                "page": None,
                "row": row_index,
                "sheet": row.get("_sheet"),
                "chunk_index": position,
                "chunk_id": f"{source}-match-{row_index}",
                "score": 1.0,
            }
        )

    return {"retrieved_docs": docs, "tabular_result": result}
