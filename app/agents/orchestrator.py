from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.planner import plan_node
from app.agents.reasoner import reason_node
from app.agents.retriever_agent import retrieve_node
from app.agents.tabular_agent import tabular_query_node
from app.agents.validator import should_retry, validate_node
from app.core.schemas import AskQuestionResponse, SourceCitation
from app.utils.guardrails import check_input_safety


class AgentState(TypedDict, total=False):
    question: str
    top_k: int | None
    is_safe: bool
    safety_reason: str | None
    search_query: str
    use_tabular_query: bool
    tabular_source: str | None
    tabular_query_expr: str
    tabular_aggregate_op: str
    tabular_aggregate_column: str | None
    tabular_result: dict | None
    retrieved_docs: list[dict]
    answer: str
    sources: list[SourceCitation]
    is_grounded: bool
    should_retry: bool
    warnings: list[str]
    retry_count: int


def safety_node(state: AgentState) -> dict:
    is_safe, reason = check_input_safety(state["question"])
    return {"is_safe": is_safe, "safety_reason": reason}


def _safety_router(state: AgentState) -> str:
    return "plan" if state.get("is_safe") else "end"


def _plan_router(state: AgentState) -> str:
    return "tabular_query" if state.get("use_tabular_query") else "retrieve"


@lru_cache
def _get_compiled_graph():
    graph = StateGraph(AgentState)
    graph.add_node("safety_check", safety_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tabular_query", tabular_query_node)
    graph.add_node("reason", reason_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "safety_check")
    graph.add_conditional_edges("safety_check", _safety_router, {"plan": "plan", "end": END})
    graph.add_conditional_edges("plan", _plan_router, {"tabular_query": "tabular_query", "retrieve": "retrieve"})
    graph.add_edge("tabular_query", "reason")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "validate")
    graph.add_conditional_edges("validate", should_retry, {"retry": "reason", "end": END})

    return graph.compile()


def run_agent(question: str, top_k: int | None = None) -> AskQuestionResponse:
    graph = _get_compiled_graph()
    initial_state: AgentState = {
        "question": question,
        "top_k": top_k,
        "warnings": [],
        "retry_count": 0,
    }
    result = graph.invoke(initial_state)

    if not result.get("is_safe", True):
        message = "I can't help with that request."
        if result.get("safety_reason"):
            message += f" {result['safety_reason']}"
        return AskQuestionResponse(answer=message, sources=[], is_grounded=False, warnings=["Blocked by input safety check."])

    return AskQuestionResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        is_grounded=result.get("is_grounded", False),
        warnings=result.get("warnings", []),
    )
