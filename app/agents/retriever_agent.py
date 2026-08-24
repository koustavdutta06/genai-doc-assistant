from langchain_core.tools import tool

from app.services.vector_store_service import similarity_search_with_score


@tool
def retrieve_documents(query: str, k: int = 4) -> list[dict]:
    """Retrieve the most relevant document chunks for a query from the vector store."""
    results = similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "doc_type": doc.metadata.get("doc_type"),
            "page": doc.metadata.get("page"),
            "row": doc.metadata.get("row"),
            "sheet": doc.metadata.get("sheet"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "score": score,
        }
        for doc, score in results
    ]


def retrieve_node(state: dict) -> dict:
    top_k = state.get("top_k") or 4
    docs = retrieve_documents.invoke({"query": state["search_query"], "k": top_k})
    return {"retrieved_docs": docs}
