"""
Utility AI Assistant — FastAPI entry point

POST /query  →  route → tools + RAG → LLM → structured response

Run with:  uvicorn app.main:app --reload
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.router import route_query
from app.tools import TOOL_REGISTRY
from app.rag import retrieve_docs
from app.llm import generate_answer

app = FastAPI(
    title="Utility AI Assistant",
    description="Billing assistant combining MCP tools, RAG retrieval, and Claude",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None  # explicit override; otherwise extracted from query


class QueryResponse(BaseModel):
    answer: str
    source: str       # "tool" | "rag" | "tool+rag" | "none"
    confidence: str   # "HIGH" | "MEDIUM" | "LOW"
    debug: Optional[dict] = None


@app.get("/")
def health_check():
    return {"status": "ok", "service": "utility-ai-assistant"}


@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    route = route_query(query)
    mode = route["mode"]
    customer_id = request.customer_id or route["customer_id"]

    tool_data = None
    rag_docs = None
    sources_used = []

    if mode in ("tool", "both"):
        tool_fn = TOOL_REGISTRY.get(route.get("tool_name", "get_bill"))
        if tool_fn and customer_id:
            tool_data = tool_fn(customer_id)
            sources_used.append("tool")
        else:
            mode = "rag"  # no customer ID available — fall back to policy docs

    if mode in ("rag", "both"):
        rag_docs = retrieve_docs(query, top_k=2)
        if rag_docs:
            sources_used.append("rag")

    result = generate_answer(query=query, tool_data=tool_data, rag_docs=rag_docs)

    debug_info = {
        "route": route,
        "customer_id_used": customer_id,
        "tool_called": route.get("tool_name") if tool_data else None,
        "rag_chunks_retrieved": len(rag_docs) if rag_docs else 0,
    }

    return QueryResponse(
        answer=result["answer"],
        source="+".join(sources_used) if sources_used else "none",
        confidence=result["confidence"],
        debug=debug_info if os.getenv("DEBUG", "false").lower() == "true" else None,
    )
