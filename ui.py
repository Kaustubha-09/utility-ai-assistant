"""
Streamlit chat UI for the Utility AI Assistant.

Run with:  streamlit run ui.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from app.router import route_query
from app.tools import TOOL_REGISTRY
from app.rag import retrieve_docs
from app.llm import generate_answer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Utility AI Assistant",
    page_icon="⚡",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.confidence-HIGH  { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.confidence-MEDIUM{ background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.confidence-LOW   { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.source-tag       { background:#e2e3e5; color:#383d41; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin-right:6px; }
.meta-row         { margin-top:8px; display:flex; align-items:center; gap:6px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚡ Utility AI Assistant")
st.caption("Ask about your electricity bill, usage, charges, or billing policies.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Example queries")

    examples = [
        ("📄 Bill summary",       "What is my total bill?",                        "C001"),
        ("📊 Compare usage",      "How does my usage compare to last month?",       "C003"),
        ("🔍 Explain charges",    "Break down my charges for me",                   "C002"),
        ("⚡ Peak pricing",       "What is peak pricing and when does it apply?",   ""),
        ("💡 Reduce bill",        "How can I reduce my electricity bill?",          ""),
        ("❓ Mixed question",     "Why is my bill higher than last month?",         "C003"),
    ]

    for label, query, cid in examples:
        if st.button(label, use_container_width=True):
            st.session_state["prefill_query"] = query
            st.session_state["prefill_cid"] = cid

    st.divider()
    st.markdown("**Mock customers**")
    st.markdown("- `C001` Alice Johnson\n- `C002` Bob Martinez\n- `C003` Carol Nguyen\n- `C004` David Chen")

    st.divider()
    st.markdown("**How routing works**")
    st.markdown(
        "- Bill / usage keywords → **tool**\n"
        "- Why / explain / policy → **RAG**\n"
        "- Both present → **tool + RAG**"
    )

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            st.markdown(
                f'<div class="meta-row">'
                f'<span class="source-tag">source: {msg["source"]}</span>'
                f'<span class="confidence-{msg["confidence"]}">{msg["confidence"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])

# ── Input form ────────────────────────────────────────────────────────────────

prefill_query = st.session_state.pop("prefill_query", "")
prefill_cid   = st.session_state.pop("prefill_cid", "")

with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your question",
            value=prefill_query,
            placeholder="e.g. Why is my bill higher this month?",
            label_visibility="collapsed",
        )
    with col2:
        cid = st.text_input(
            "Customer ID",
            value=prefill_cid,
            placeholder="C001",
            label_visibility="collapsed",
        )
    submitted = st.form_submit_button("Ask ⚡", use_container_width=True)

# ── Process query ─────────────────────────────────────────────────────────────

if submitted and query.strip():
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # Route
            route = route_query(query)
            mode = route["mode"]
            customer_id = cid.strip().upper() or route["customer_id"]

            tool_data = None
            rag_docs   = None
            sources    = []

            if mode in ("tool", "both"):
                tool_fn = TOOL_REGISTRY.get(route.get("tool_name", "get_bill"))
                if tool_fn and customer_id:
                    tool_data = tool_fn(customer_id)
                    sources.append("tool")
                else:
                    mode = "rag"

            if mode in ("rag", "both"):
                rag_docs = retrieve_docs(query, top_k=2)
                if rag_docs:
                    sources.append("rag")

            result = generate_answer(query=query, tool_data=tool_data, rag_docs=rag_docs)

        answer     = result["answer"]
        confidence = result["confidence"]
        source     = "+".join(sources) if sources else "none"

        st.markdown(answer)
        st.markdown(
            f'<div class="meta-row">'
            f'<span class="source-tag">source: {source}</span>'
            f'<span class="confidence-{confidence}">{confidence}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Show tool data and RAG chunks in expanders
        if tool_data and tool_data.get("status") == "ok":
            with st.expander("Tool data retrieved"):
                st.json(tool_data["data"])

        if rag_docs:
            with st.expander(f"{len(rag_docs)} policy section(s) retrieved"):
                for doc in rag_docs:
                    st.markdown(f"**{doc['title']}** (score: {doc['score']})")
                    st.caption(doc["content"][:300] + ("…" if len(doc["content"]) > 300 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source,
        "confidence": confidence,
    })
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.info("Type a question above or pick an example from the sidebar to get started.")
