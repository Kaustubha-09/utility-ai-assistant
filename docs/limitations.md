# Limitations

## Scope

- **Single-domain prototype.** Electricity billing only. Demonstrates the patterns; not a production billing system.
- **Mock data.** 4 customers (C001–C004), 7 policy sections. Not a real customer database.
- **No auth.** `POST /query` is open. A real deployment would gate by API key or JWT.
- **No persistence of queries / answers.** Each request is stateless. No conversation history beyond Streamlit's per-session state.

## Routing

- **Fixed keyword sets.** Vocabulary drift (new phrasings, synonyms, typos) will route incorrectly. Mitigated by defaulting to RAG on no match — but a query with no recognized keywords still gets a generic answer.
- **No semantic routing.** "How much do I owe?" doesn't contain `bill` or `total` but is a tool query. A real router would use embeddings or LLM classification for these cases.
- **No multi-turn intent.** Each query is classified independently. "Tell me more" after a billing answer won't route back to the same tool.

## RAG

- **TF-IDF is bag-of-words.** Synonyms, paraphrases, semantic similarity beyond term overlap are missed. The corpus is small enough that this is rarely a problem in practice.
- **No re-ranking.** Top-k by raw cosine similarity. No second-stage re-ranker, no MMR diversification.
- **No grounding citations.** The LLM doesn't say *"per Section X of the policy docs"* — it just uses the context. A real assistant would surface which docs were retrieved.

## LLM grounding

- **System prompt is the only enforcement.** "Do not invent numbers" is a soft constraint. The model can still violate it, especially on edge phrasings or long-form responses.
- **No output validation.** We trust the model to follow the format. A real deployment would parse outputs and reject responses that introduce numbers not in `[TOOL DATA]`.
- **No hallucination tests.** No regression test catches *"the model invented a $0.07/kWh rate that doesn't exist in the data"*. Tracked in roadmap.

## Confidence calibration

- **Confidence levels are self-reported by the model.** They are not calibrated — `HIGH` from the model doesn't mean "I'm right 90% of the time".
- **No threshold-based escalation.** A `LOW`-confidence response isn't routed to a human or a different model. Production would.

## Performance

- **No caching.** Same query repeated → same end-to-end pipeline. A request-level cache (with appropriate TTL) would cut latency on hot paths.
- **No batching.** Each query is a separate LLM call.
- **No streaming response.** Gemini supports streaming but `llm.py` waits for the full response. Streaming would improve perceived latency.

## Operational

- **No structured logging.** `print()` and unstructured FastAPI logs. Production would use `structlog` JSON output.
- **No metrics export.** No Prometheus / Datadog instrumentation. No latency histograms.
- **No rate limiting.** A single client could exhaust the Gemini API key. Production would put this behind a rate limiter.
- **No error budget / SLOs.** No formal availability target.
