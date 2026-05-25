# Architecture Decision Records

Dated decisions. Append-only.

## ADR-001 · Four patterns in one small codebase

**Date:** 2026-04
**Status:** Accepted

This codebase intentionally co-locates MCP-style tool calling, RAG, intent routing, and grounded synthesis — instead of demonstrating each pattern in isolation.

**Why:** the value isn't in any single pattern; it's in the *contract* between them. The router decides which to use; tools produce structured data; RAG produces grounded context; the LLM synthesizes only from what the others provide. Showing the contract is the point.

---

## ADR-002 · Keyword router instead of LLM classifier

**Date:** 2026-04
**Status:** Accepted

`router.py` scores queries against fixed keyword sets, not an LLM.

**Why:** routing happens on every query. An LLM router costs ~50–200ms and a fraction of a cent each, where a keyword scorer runs in microseconds for free. More importantly: a keyword router is *auditable* — when the assistant misroutes, you can read the keyword score and know exactly why. LLM routers are black boxes.

**Cost:** keyword sets need maintenance as vocabulary drifts. Mitigated by defaulting to `rag` on no match (safer than guessing a tool call).

---

## ADR-003 · TF-IDF retrieval, not embeddings

**Date:** 2026-04
**Status:** Accepted

`rag.py` builds a scikit-learn TF-IDF index over `data/docs.txt` at startup. Retrieval is `cosine_similarity` against the query vector.

**Why:** the corpus is 7 policy sections (~a few hundred tokens total). Embeddings would mean a vector DB, an embedding API call per chunk at index time, and another per query at retrieval time — total overkill at this scale. TF-IDF runs locally in milliseconds with no external dependencies.

**When to revisit:** if the corpus grows past ~500 chunks, swap to FAISS + sentence-transformers. The retrieval contract (`top-k by cosine similarity`) doesn't change.

---

## ADR-004 · Grounded synthesis with explicit "no inventing" rule

**Date:** 2026-04
**Status:** Accepted

The system prompt is explicit: *"For billing questions, base your answer ONLY on provided `[TOOL DATA]` and `[POLICY DOCS]`. Do not invent numbers, rates, or policies."*

**Why:** the classic failure mode of LLM-powered billing assistants is fabricating a peak-pricing rate, an inflated charge, or a fake policy clause. The grounding rule + the structural separation (LLM never sees raw DB, only tool output) is the defense.

**Cost:** the LLM is less "smart-sounding" when context is sparse. That's the point — we want `LOW` confidence to be reachable.

---

## ADR-005 · Explicit `CONFIDENCE: HIGH/MEDIUM/LOW` label

**Date:** 2026-04
**Status:** Accepted

LLM is instructed to end every response with `CONFIDENCE: HIGH/MEDIUM/LOW`. A deterministic parser extracts it.

**Why:** most LLM apps treat every response as equally confident. We make uncertainty a first-class field. `LOW` confidence means *the context was insufficient* — the user-visible "I don't know" path that demos skip.

**Cost:** about 5 extra tokens per response. Negligible. The parser is 10 lines.

---

## ADR-006 · Mock mode runs every code path except the LLM call

**Date:** 2026-04
**Status:** Accepted

Without `GEMINI_API_KEY`, `llm.py` calls `_mock_generate()` which produces deterministic template responses keyed off tool output.

**Why:** developers and CI shouldn't need an API key to exercise the app. Routing, tools, RAG, response shape, confidence label — all reachable on the mock path. Only the synthesis is mocked, and the mock's confidence rules mirror what a real LLM would produce.

---

## ADR-007 · Single `app/llm.py` is the LLM-vendor boundary

**Date:** 2026-04
**Status:** Accepted, post-migration

`app/llm.py` was originally `anthropic` + Claude. The current commit migrated to `google-generativeai` + `gemini-2.5-flash`. Routing, tools, RAG, response shape, confidence parsing — none changed.

**Why this matters:** the architecture took a vendor swap in stride because the boundary was a single function (`generate_answer`). The migration was a one-file diff. This is what boundary-driven architecture buys you.

**When to revisit:** if we want to support multiple providers concurrently (e.g., fallback dispatch like RapidTriage's `AI_MODEL_TYPE`), `llm.py` becomes a registry instead of a single implementation.

---

## ADR-008 · Two surfaces (FastAPI + Streamlit), one core

**Date:** 2026-04
**Status:** Accepted

`app/main.py` (FastAPI REST) and `ui.py` (Streamlit chat) both import `route_query` → tools → `retrieve_docs` → `generate_answer`. Same code, two presentation skins.

**Why:** FastAPI is for programmatic / curl-test use; Streamlit is for human demos. Forcing both through one core means a bug in routing or grounding shows up identically on both surfaces — and a fix flows to both for free.

---

## ADR-009 · Customer ID regex on `C\d{3}` pattern

**Date:** 2026-04
**Status:** Accepted

`router.py` extracts customer IDs via regex on `C\d{3}` (e.g., `C003`). Falls back to an explicit `customer_id` field in the request body.

**Why:** users will say "for C003" inline in natural language. Forcing a separate field is a worse UX. Regex extraction is robust for the demo schema; production would expand to broader ID patterns.

---

## ADR-010 · Secrets via `.env` + `python-dotenv`, gitignored

**Date:** 2026-04
**Status:** Accepted

`GEMINI_API_KEY` loaded from `.env` via `python-dotenv`. `.env` is gitignored. `.env.example` is committed.

**Why:** simplest secret pattern that works locally and in CI (CI sets `GEMINI_API_KEY` as a workflow env var, no `.env` file needed). Never commits secrets, never blocks local dev.
