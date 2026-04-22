"""
Streamlit chat UI for the Utility AI Assistant.
Run with:  streamlit run ui.py
"""

import sys, os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, os.path.dirname(__file__))

import re
import streamlit as st
from app.router import route_query
from app.tools import TOOL_REGISTRY
from app.rag import retrieve_docs
from app.llm import generate_answer


def _md(text: str) -> None:
    """Render markdown with dollar signs escaped to prevent Streamlit LaTeX mode."""
    st.markdown(re.sub(r'(?<!\\)\$', r'\\$', text))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Utility AI Assistant", page_icon="⚡", layout="centered")

st.markdown("""
<style>
.confidence-HIGH  { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.confidence-MEDIUM{ background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.confidence-LOW   { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.source-tag       { background:#e2e3e5; color:#383d41; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
.meta-row         { margin-top:8px; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = {}   # { customer_id: [msg, ...] }
if "pending" not in st.session_state:
    st.session_state.pending = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚡ Utility AI Assistant")

    st.subheader("Customer")
    customer_id = st.selectbox(
        "Select customer",
        ["C001 — Alice Johnson", "C002 — Bob Martinez", "C003 — Carol Nguyen", "C004 — David Chen"],
        label_visibility="collapsed",
    ).split(" ")[0]  # extract just "C001" etc.

    st.divider()

    st.subheader("Try an example")
    examples = [
        ("📄 Bill summary",     "What is my total bill?",                      True),
        ("📊 Compare usage",    "How does my usage compare to last month?",     True),
        ("🔍 Explain charges",  "Break down my charges",                        True),
        ("⚡ Peak pricing",     "What is peak pricing and when does it apply?", False),
        ("💡 Reduce my bill",   "How can I reduce my electricity bill?",        False),
        ("❓ Why higher bill",  "Why is my bill higher than last month?",       True),
    ]

    for label, query_text, needs_cid in examples:
        if st.button(label, use_container_width=True):
            # Store as pending — will be processed on next rerun automatically
            st.session_state.pending = {
                "query": query_text,
                "customer_id": customer_id if needs_cid else None,
            }
            st.rerun()

    st.divider()
    st.caption("**Routing logic**")
    st.caption("Bill / usage → tool\nWhy / explain → RAG\nBoth present → tool + RAG")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚡ Utility AI Assistant")
st.caption(f"Active customer: **{customer_id}**  ·  Type a question or pick an example from the sidebar.")

# Shorthand for the current customer's message list
chat = st.session_state.messages.setdefault(customer_id, [])

# ── Core processing function ──────────────────────────────────────────────────

def process_query(query: str, cid: str | None, history: list = None):
    """Run the full pipeline and render the assistant response."""
    route = route_query(query)
    mode = route["mode"]
    resolved_cid = cid or route["customer_id"]

    tool_data = None
    rag_docs = None
    sources = []

    if mode in ("tool", "both"):
        tool_fn = TOOL_REGISTRY.get(route.get("tool_name", "get_bill"))
        if tool_fn and resolved_cid:
            tool_data = tool_fn(resolved_cid)
            sources.append("tool")
        else:
            mode = "rag"

    if mode in ("rag", "both"):
        rag_docs = retrieve_docs(query, top_k=2)
        if rag_docs:
            sources.append("rag")

    result = generate_answer(query=query, tool_data=tool_data, rag_docs=rag_docs, history=history)

    answer     = result["answer"]
    confidence = result["confidence"]
    source     = "+".join(sources) if sources else "none"

    # Render
    _md(answer)
    st.markdown(
        f'<div class="meta-row">'
        f'<span class="source-tag">source: {source}</span>'
        f'<span class="confidence-{confidence}">{confidence}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if tool_data and tool_data.get("status") == "ok":
        with st.expander("Tool data"):
            st.json(tool_data["data"])

    if rag_docs:
        with st.expander(f"{len(rag_docs)} policy section(s) retrieved"):
            for doc in rag_docs:
                st.markdown(f"**{doc['title']}** · score: {doc['score']}")
                snippet = doc["content"][:300] + ("…" if len(doc["content"]) > 300 else "")
                st.caption(re.sub(r'(?<!\\)\$', r'\\$', snippet))

    return {"role": "assistant", "content": answer, "source": source, "confidence": confidence}

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in chat:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _md(msg["content"])
            st.markdown(
                f'<div class="meta-row">'
                f'<span class="source-tag">source: {msg["source"]}</span>'
                f'<span class="confidence-{msg["confidence"]}">{msg["confidence"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])

# ── Handle sidebar button click (auto-submit) ─────────────────────────────────

if st.session_state.pending:
    pending = st.session_state.pending
    st.session_state.pending = None

    cid_for_pending = pending["customer_id"] or customer_id
    pending_chat = st.session_state.messages.setdefault(cid_for_pending, [])
    history = [{"role": m["role"], "content": m["content"]} for m in pending_chat]
    pending_chat.append({"role": "user", "content": pending["query"]})
    with st.chat_message("user"):
        st.markdown(pending["query"])

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = process_query(pending["query"], pending["customer_id"], history=history)
    pending_chat.append(response)
    st.rerun()

# ── Chat input (free typing) ──────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your bill…"):
    history = [{"role": m["role"], "content": m["content"]} for m in chat]
    chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = process_query(prompt, customer_id, history=history)
    chat.append(response)
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────

if not chat:
    st.info("Select a customer in the sidebar, then type a question or click an example.")
