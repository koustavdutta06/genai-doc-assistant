import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise GenAI Doc Assistant", layout="wide")


def check_health() -> dict | None:
    try:
        response = requests.get(f"{BACKEND_URL}/health-check", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


with st.sidebar:
    st.header("Documents")

    health = check_health()
    if health:
        status_icon = "🟢" if health.get("ollama_reachable") else "🟡"
        st.caption(f"{status_icon} Backend: {health.get('status')} | Ollama: {health.get('ollama_reachable')}")
    else:
        st.caption("🔴 Backend unreachable")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / CSV / XLSX",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
    )

    if st.button("Upload & Index", disabled=not uploaded_files):
        files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        with st.spinner("Indexing documents..."):
            try:
                response = requests.post(f"{BACKEND_URL}/upload-documents", files=files_payload, timeout=300)
                response.raise_for_status()
                results = response.json().get("results", [])
                for result in results:
                    if result["status"] == "indexed":
                        st.success(f"{result['filename']}: {result['chunks_added']} chunks indexed")
                    else:
                        st.error(f"{result['filename']}: {result.get('error', 'failed')}")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

st.title("Enterprise Knowledge Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    location = src.get("page") if src.get("page") is not None else src.get("row")
                    st.markdown(f"**{src['source']}** (loc: {location}, score: {src['score']:.3f})")
                    st.caption(src["snippet"])
        if message["role"] == "assistant" and message.get("warnings"):
            for warning in message["warnings"]:
                st.warning(warning)

if question := st.chat_input("Ask a question about your uploaded documents"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask-question", json={"question": question}, timeout=120
                )
                response.raise_for_status()
                data = response.json()
                st.markdown(data["answer"])
                if not data.get("is_grounded", True):
                    st.warning("This answer may not be fully grounded in the uploaded documents.")
                if data.get("sources"):
                    with st.expander("Sources"):
                        for src in data["sources"]:
                            location = src.get("page") if src.get("page") is not None else src.get("row")
                            st.markdown(f"**{src['source']}** (loc: {location}, score: {src['score']:.3f})")
                            st.caption(src["snippet"])
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "warnings": data.get("warnings", []),
                    }
                )
            except requests.RequestException as exc:
                error_message = f"Error contacting backend: {exc}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
