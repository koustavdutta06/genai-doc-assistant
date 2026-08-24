import re
from pathlib import Path

import pandas as pd

from app.core.config import get_settings

_FORBIDDEN_PATTERN = re.compile(r"__|import|exec\(|eval\(|os\.|sys\.|subprocess|open\(|lambda", re.IGNORECASE)


def list_tabular_sources() -> dict[str, list[str]]:
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    sources: dict[str, list[str]] = {}
    if not upload_dir.exists():
        return sources

    for path in upload_dir.iterdir():
        try:
            if path.suffix.lower() == ".csv":
                sources[path.name] = list(pd.read_csv(path, nrows=0).columns)
            elif path.suffix.lower() == ".xlsx":
                sheets = pd.read_excel(path, sheet_name=None, nrows=0)
                columns = sorted({col for df in sheets.values() for col in df.columns})
                sources[path.name] = columns
        except Exception:
            continue

    return sources


def get_categorical_values(filename: str, max_unique: int = 200) -> dict[str, list[str]]:
    try:
        df = _load_dataframe(filename)
    except (FileNotFoundError, ValueError):
        return {}

    values: dict[str, list[str]] = {}
    for column in df.columns:
        if df[column].dtype == object:
            uniques = df[column].dropna().unique()
            if 0 < len(uniques) <= max_unique:
                values[column] = [str(v) for v in uniques]
    return values


def _load_dataframe(filename: str) -> pd.DataFrame:
    settings = get_settings()
    path = Path(settings.upload_dir) / filename

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None)
        frames = []
        for sheet_name, sheet_df in sheets.items():
            sheet_df = sheet_df.copy()
            sheet_df["_sheet"] = sheet_name
            frames.append(sheet_df)
        return pd.concat(frames, ignore_index=True)

    raise ValueError(f"Unsupported tabular file: {filename}")


def run_tabular_query(
    source: str,
    query_expr: str | None,
    aggregate_op: str = "none",
    aggregate_column: str | None = None,
    max_rows: int = 30,
) -> dict:
    try:
        df = _load_dataframe(source)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc), "rows": [], "row_indices": [], "row_count": 0, "total_rows": 0}

    total_rows = len(df)
    filtered = df

    if query_expr:
        if _FORBIDDEN_PATTERN.search(query_expr):
            return {
                "error": f"Query expression '{query_expr}' contains disallowed tokens.",
                "rows": [],
                "row_indices": [],
                "row_count": 0,
                "total_rows": total_rows,
            }
        try:
            filtered = df.query(query_expr, engine="python")
        except Exception as exc:
            return {
                "error": f"Invalid query expression '{query_expr}': {exc}",
                "rows": [],
                "row_indices": [],
                "row_count": 0,
                "total_rows": total_rows,
            }

    result = {
        "error": None,
        "rows": filtered.head(max_rows).to_dict(orient="records"),
        "row_indices": list(filtered.head(max_rows).index),
        "row_count": len(filtered),
        "total_rows": total_rows,
    }

    if aggregate_op == "count":
        result["aggregate_op"] = "count"
        result["aggregate_column"] = aggregate_column
        result["aggregate_value"] = float(len(filtered))
    elif aggregate_op in ("sum", "mean") and aggregate_column:
        try:
            result["aggregate_op"] = aggregate_op
            result["aggregate_column"] = aggregate_column
            result["aggregate_value"] = float(getattr(filtered[aggregate_column], aggregate_op)())
        except Exception as exc:
            result["error"] = f"Could not compute {aggregate_op} on column '{aggregate_column}': {exc}"

    return result
