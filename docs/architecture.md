# Architecture

Four production-grade patterns in one small codebase: MCP-style tool calling, RAG retrieval, keyword intent routing, and grounded LLM synthesis with explicit confidence scoring.

## Request flow

```
POST /query  {"query": "Why is my bill higher? C003"}
        │
        ▼
    router.py
    ├─ Scores query against tool keywords (bill, usage, kwh…)
    │  and RAG keywords (why, explain, policy…)
    ├─ Both score > 0  →  mode: "both"
    └─ Extracts customer_id: "C003"  (regex on "C\d{3}")
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
    tools.py                           rag.py
    compare_usage("C003")              retrieve_docs(query, top_k=2)
    → current: 1240 kWh                → "Why Is My Bill Higher..."
    → previous: 890 kWh                → "What Is Peak Pricing"
    → trend: significantly_higher        (TF-IDF cosine similarity)
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
                   llm.py
                   _build_context: composes [TOOL DATA] + [POLICY DOCS]
                   Calls gemini-2.5-flash with grounded system prompt
                   _parse_confidence: extracts CONFIDENCE label from tail
                       │
                       ▼
        {
          "answer": "Your usage jumped 39%...",
          "source": "tool+rag",
          "confidence": "HIGH"
        }
```

## Components

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app; wires router → tools → rag → llm; exposes `POST /query` |
| `app/router.py` | Keyword scoring + intent classification (`tool` / `rag` / `both`); customer ID regex extraction; default to `rag` on no match |
| `app/tools.py` | Three MCP-style functions reading `data/billing_data.json` and returning structured JSON: `get_bill`, `compare_usage`, `explain_charges` |
| `app/rag.py` | Chunk `data/docs.txt` by `SECTION:` headers; build TF-IDF index at startup; `retrieve_docs(query, top_k)` returns title + content + cosine score |
| `app/llm.py` | Build context block from tool + RAG output; call `gemini-2.5-flash` with strict grounding prompt; parse `CONFIDENCE: HIGH/MEDIUM/LOW` label; falls back to deterministic template responses if `GEMINI_API_KEY` is unset |
| `ui.py` | Streamlit chat UI; sidebar example queries; per-customer session state; color-coded confidence badges; expandable tool/RAG payloads |

## Routing semantics

```python
TOOL_KEYWORDS = {"bill", "usage", "kwh", "charge", "breakdown", ...}
RAG_KEYWORDS  = {"why", "explain", "policy", "peak", "reduce", ...}

tool_score = sum(1 for w in query.split() if w.lower() in TOOL_KEYWORDS)
rag_score  = sum(1 for w in query.split() if w.lower() in RAG_KEYWORDS)

if tool_score > 0 and rag_score > 0: mode = "both"
elif tool_score > 0:                  mode = "tool"
elif rag_score > 0:                   mode = "rag"
else:                                  mode = "rag"   # safe default
```

The default-to-rag rule is intentional: if we can't classify, retrieving policy docs is safer than guessing a tool call with a maybe-correct customer ID.

## Grounded synthesis

The LLM never sees raw customer data. The `_build_context()` function composes:

```
[TOOL DATA]
<JSON from tool, or error message>

[POLICY DOCS]
--- Section title ---
<section body>
--- Section title ---
<section body>
```

The system prompt instructs:
> **For billing questions, base your answer ONLY on provided `[TOOL DATA]` and `[POLICY DOCS]`. Do not invent numbers, rates, or policies.**
> **Quote exact dollar and kWh figures from the data.**

This separation is the defense against the classic *"LLM confidently fabricates a peak-pricing rate"* failure mode.

## Mock mode (no API key)

If `GEMINI_API_KEY` is unset, `llm.py` returns deterministic template responses keyed off tool output. Routing, tools, and RAG still run end-to-end. The UI exercises every Resource state. Only the synthesis is mocked.

Mock-mode confidence rules:
- Tool success → `HIGH`
- Tool error → `LOW`
- RAG only, top score > 0.2 → `HIGH`; else `MEDIUM`
- No tool + no RAG (greeting / capabilities) → `HIGH`

## Confidence parser

LLM is instructed to end every response with one of:
```
CONFIDENCE: HIGH
CONFIDENCE: MEDIUM
CONFIDENCE: LOW
```

`_parse_confidence(raw_text)` scans the response from the bottom up, strips the label, and returns `(clean_answer, confidence_level)`. If no label found, defaults to `LOW`.

## What runs where

| Concern | Lives in |
|---|---|
| HTTP routing | `app/main.py` (FastAPI) |
| Intent + ID extraction | `app/router.py` |
| Structured tool functions | `app/tools.py` |
| TF-IDF retrieval | `app/rag.py` (scikit-learn) |
| LLM call + grounding | `app/llm.py` (`google-generativeai`) |
| Streamlit UI | `ui.py` |
| Mock customer data | `data/billing_data.json` |
| Policy corpus | `data/docs.txt` |
