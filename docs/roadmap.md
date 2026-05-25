# Roadmap

## Phase 1 — Grounding hardening (1 week)

- Output validation: parse the LLM response, extract all numeric tokens, reject the response if any number isn't in `[TOOL DATA]`.
- Citation surfacing: LLM names which doc sections it used (e.g., "per the Peak Pricing policy"); UI renders them as collapsible references.
- Hallucination regression tests: 10–20 known-bad-context queries, assert `LOW` confidence is reached.

## Phase 2 — Semantic routing fallback (1 week)

- If the keyword router returns no match (currently defaults to `rag`), call a cheap LLM classifier (`gemini-2.5-flash` with a 50-token prompt) for the route decision.
- Log the classifier verdict so we can mine misrouted queries to grow the keyword set.

## Phase 3 — Streaming responses (3–5 days)

- Wire Gemini's streaming API in `llm.py`.
- FastAPI endpoint returns Server-Sent Events.
- Streamlit UI renders token-by-token (`st.write_stream` exists).
- Mock mode emits tokens with synthetic delay for UX parity.

## Phase 4 — Auth + rate limiting (1 week)

- API key per client; passed via `X-API-Key` header.
- Per-key rate limit (e.g., 100 req/min) via `slowapi` (FastAPI rate-limiter).
- 429 responses with `Retry-After` header.

## Phase 5 — Persistence + history (1 week)

- Postgres for customer data (replace `data/billing_data.json`).
- Conversation history table; `POST /query` accepts `session_id` and includes recent turns in the LLM context.
- Per-customer query log for analytics.

## Phase 6 — Confidence calibration (research)

- Collect 200+ real query / response / human-judged-correctness triples.
- Compute Brier score / calibration curve on `HIGH/MEDIUM/LOW`.
- Adjust LLM prompt or post-hoc temperature scaling so reported confidence tracks actual accuracy.

## Phase 7 — Embedding upgrade (if corpus grows)

- Trigger: corpus exceeds ~500 chunks or recall on the existing TF-IDF index drops below an acceptable floor.
- Swap retrieval implementation: FAISS index + `text-embedding-3-small` (OpenAI) or `gemini-embedding-001` (Google).
- Same retrieval contract: `top-k by cosine similarity`. The rest of the code doesn't change.

## Out of scope

- **Building a real billing system.** This is a pattern demo, not a competitor to utility billing platforms.
- **Multi-language i18n.** English-only for the demo.
- **Mobile app.** REST API is consumable from any frontend.
