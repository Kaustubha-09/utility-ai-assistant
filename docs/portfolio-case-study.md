# Utility AI Assistant — Portfolio Case Study

Skim time: 3 minutes.

## The brief

Build a single small codebase that demonstrates **four production-grade patterns** in the LLM-application stack — MCP-style tool calling, RAG retrieval, intent routing, grounded synthesis — and show how they compose under one request flow with explicit confidence scoring.

## The build

- **FastAPI** REST API + **Streamlit** chat UI, sharing one router + tools + RAG + LLM core.
- **3 MCP-style tools** (`get_bill`, `compare_usage`, `explain_charges`) reading mock customer data, returning typed JSON.
- **scikit-learn TF-IDF** retrieval over a 7-section policy corpus — no embedding API, no vector DB.
- **Keyword intent router** — auditable, near-zero-latency, falls back to RAG on no match.
- **Gemini 2.5 Flash** for grounded synthesis. **`CONFIDENCE: HIGH/MEDIUM/LOW`** label on every response.

## The engineering I'd defend

### 1. Auditable routing over LLM classification

Keyword routing runs in microseconds for free. The score is readable — when a query misroutes, I can see exactly why and adjust the keyword set. An LLM router is a black box that costs ~50–200ms and a fraction of a cent per query. At this scope, an LLM router would be cargo-culted. See [decisions.md, ADR-002](decisions.md#adr-002--keyword-router-instead-of-llm-classifier).

### 2. TF-IDF over embeddings

7 policy sections, a few hundred tokens total. Embeddings would mean a vector DB, an embedding API call per chunk at index time, another per query at retrieval time — total overkill. TF-IDF in scikit-learn runs locally in milliseconds. The retrieval contract is `top-k by cosine similarity`; if the corpus grew past a few hundred chunks, swap to FAISS without changing the contract. See [ADR-003](decisions.md#adr-003--tf-idf-retrieval-not-embeddings).

### 3. Grounded synthesis as a hallucination defense

The system prompt is explicit: *"For billing questions, base your answer ONLY on provided `[TOOL DATA]` and `[POLICY DOCS]`."* The LLM never sees raw customer data — only the structured JSON the tools produced. If a tool errored, the LLM gets the error message and degrades gracefully. The architectural separation between *what do we know* and *how do we explain it* is the defense against the classic *"LLM fabricates a peak-pricing rate"* failure mode. See [ADR-004](decisions.md#adr-004--grounded-synthesis-with-explicit-no-inventing-rule).

### 4. Explicit confidence as a first-class field

LLM instructed to end every response with `CONFIDENCE: HIGH/MEDIUM/LOW`. Parser extracts. UI renders color-coded badge. Result: `LOW` confidence is a reachable state — the "I don't know" path that lots of demos skip. See [ADR-005](decisions.md#adr-005--explicit-confidence-highmediumlow-label).

### 5. Mock mode exercises every path except the LLM call

Without `GEMINI_API_KEY`, `_mock_generate()` produces deterministic template responses keyed off tool output. Routing, tools, RAG, response shape, confidence labels — every code path reachable offline. The mock's confidence rules mirror what a real LLM would produce. See [ADR-006](decisions.md#adr-006--mock-mode-runs-every-code-path-except-the-llm-call).

### 6. Vendor migration in one file

`app/llm.py` was Anthropic Claude. Now it's Gemini 2.5 Flash. Routing, tools, RAG, response shape, confidence parsing — none changed. The migration was a one-file diff because the boundary was a single function (`generate_answer`). This is what boundary-driven architecture buys you. See [ADR-007](decisions.md#adr-007--single-appllmpy-is-the-llm-vendor-boundary).

## The honest part

- **Mock-data scope.** 4 customers, 7 policy sections — demonstrates the patterns, not a production billing system.
- **Confidence is self-reported, not calibrated.** `HIGH` from the model doesn't mean "right 90% of the time". Calibration is roadmap Phase 6.
- **No hallucination regression tests.** The grounding prompt is a soft constraint; output validation that rejects fabricated numbers is roadmap Phase 1.
- **No streaming, no caching, no rate limiting.** All operational gaps tracked in [limitations.md](limitations.md).

## What I'd do next

Roadmap Phase 1 (hallucination hardening) is the highest-leverage next step. Output validation that rejects any numeric token not in `[TOOL DATA]` would catch the failure modes that the soft "don't invent numbers" rule misses, and it's a few dozen lines of code.

## What this signals to a recruiter

- I can compose multiple LLM-application patterns under one contract without sprawling into a multi-service architecture.
- I make explicit, auditable choices on routing and retrieval — and document *why* each pattern was chosen over the obvious alternative (LLM router, embedding search).
- I understand grounded synthesis as a defense against the dominant LLM failure mode, not just a buzzword.
- I treat uncertainty as a first-class field — `LOW` confidence is reachable and renders distinctly in the UI.
- I migrated the LLM vendor in a single file and documented it; I understand what a boundary is worth.
