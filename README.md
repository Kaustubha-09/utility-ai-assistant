# Utility AI Assistant

> A prototype AI system for electricity billing support, demonstrating four production-grade patterns in one small codebase: MCP-style tool calling, RAG over policy docs, keyword-based intent routing, and grounded LLM synthesis with explicit confidence scoring.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.6-D97757)](https://www.anthropic.com)

A naive LLM given *"Why is my bill higher than last month?"* would guess at numbers and policies. This system separates that question into two parts — *what are my actual numbers?* (tool) and *what explains them?* (RAG) — and only sends the LLM grounded context to synthesize from.

| Pattern | Implementation |
|---|---|
| MCP-style tool calling | Deterministic, structured data access — no guessing |
| RAG pipeline | Policy retrieval via TF-IDF; answers grounded in documents |
| Intelligent routing | Keyword intent classifier sends each query to the right source |
| Reliable AI design | Confidence scoring; "I don't know" fallback; no hallucination |

---

## Screenshots

Drop the canonical three into [`Screenshots/`](Screenshots/) — `01_chat_ui.png` (Streamlit chat), `02_tool_expansion.png` (tool payload expanded), `03_swagger.png` (FastAPI `/docs`).

---

## Features

- **Two surfaces, one core** — Streamlit chat UI and FastAPI REST endpoint share the same router, tools, RAG index, and LLM module.
- **Three MCP-style tools** — `get_bill`, `compare_usage`, `explain_charges`, all reading from `data/billing_data.json` and returning structured JSON.
- **TF-IDF RAG over policy docs** — chunks `data/docs.txt` by `SECTION:` headers, builds the index at startup, retrieves top-k at query time. No embedding API required.
- **Keyword router** — scores each query against tool keywords + RAG keywords; falls back to `rag` rather than guessing tool calls.
- **Customer ID extraction** — regex pulls `C001`-style IDs out of free text, or accepts an explicit `customer_id` field.
- **Confidence-scored answers** — LLM appends `CONFIDENCE: HIGH/MEDIUM/LOW`, parsed into a separate field. The Streamlit UI renders these as color-coded badges.
- **Grounded synthesis** — system prompt forbids inventing numbers or policies; the LLM only sees tool output + retrieved RAG sections.
- **Debug mode** — `DEBUG=true` adds route decisions, tool calls, and RAG chunk counts to every response.

---

## Architecture

### Request flow

```
POST /query  {"query": "Why is my bill higher? C003"}
        │
        ▼
    router.py
    ├─ Scores query against tool keywords (bill, usage, kwh…)
    │  and RAG keywords (why, explain, policy…)
    ├─ Both score > 0  →  mode: "both"
    └─ Extracts customer_id: "C003"
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
    tools.py                           rag.py
    compare_usage("C003")              retrieve_docs(query)
    → current: 1240 kWh                → "Why Is My Bill Higher..."
    → previous: 890 kWh                → "What Is Peak Pricing"
    → trend: significantly_higher
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
                   llm.py
                   Builds context block from tool data + policy docs
                   Calls Claude Sonnet 4.6 with strict grounding rules
                   Parses answer and confidence label
                       │
                       ▼
        {
          "answer": "Your usage jumped 39%...",
          "source": "tool+rag",
          "confidence": "HIGH"
        }
```

### Components

| File | Responsibility |
|---|---|
| `app/router.py` | Score query against keyword sets, return `mode: tool / rag / both`, extract customer ID. Defaults to `rag` on no match. |
| `app/tools.py` | Three MCP-style functions reading `data/billing_data.json`: `get_bill`, `compare_usage`, `explain_charges` |
| `app/rag.py` | Chunk `data/docs.txt` by `SECTION:` headers, build TF-IDF index, retrieve top-k by cosine similarity |
| `app/llm.py` | Assemble context block from tool output + RAG sections, call Claude Sonnet 4.6 with grounding rules, parse confidence |
| `app/main.py` | FastAPI app — wires router → tools → rag → llm, exposes `POST /query` |
| `ui.py` | Streamlit chat UI — sidebar example queries, per-customer session state, color-coded confidence badges, expandable tool / RAG data |

### Project structure

```
utility-ai-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              FastAPI app
│   ├── router.py            Intent classification + ID extraction
│   ├── tools.py             get_bill · compare_usage · explain_charges
│   ├── rag.py               TF-IDF index over policy docs
│   └── llm.py               Claude Sonnet 4.6 grounded synthesis
├── data/
│   ├── billing_data.json    Mock customers C001–C004
│   └── docs.txt             7 policy sections
├── Screenshots/             UI captures referenced from this README
├── ui.py                    Streamlit chat UI
├── requirements.txt
└── README.md
```

### Confidence levels

| Level | Meaning |
|---|---|
| `HIGH` | Direct account data fully answers the question |
| `MEDIUM` | Partial data, or answer inferred from policy |
| `LOW` | Insufficient context; answer may be incomplete |

---

## Tech Stack

| Layer | Choice |
|---|---|
| API | FastAPI 0.115 |
| UI | Streamlit 1.41 |
| LLM | Claude Sonnet 4.6 via the Anthropic API |
| RAG | scikit-learn TF-IDF (no embedding API) |
| Validation | Pydantic 2.10 |
| Env | python-dotenv |
| Numerics | numpy 1.26 |

Note: `requirements.txt` pins `google-generativeai` from an earlier Gemini build; the production path uses `anthropic` and Claude Sonnet 4.6 via `llm.py`. Remove the Gemini pin once you confirm it's not imported anywhere.

---

## Getting Started

```bash
cd ~/Desktop/utility-ai-assistant

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
# or copy .env.example → .env and fill in the key
```

### Chat UI (Streamlit)

```bash
streamlit run ui.py
```

Opens at `http://localhost:8501` — chat interface with sidebar examples, color-coded confidence badges, and expandable tool/RAG data.

### REST API (FastAPI)

```bash
uvicorn app.main:app --reload
```

API at `http://127.0.0.1:8000` · Swagger docs at `/docs`.

### Debug mode

```bash
DEBUG=true uvicorn app.main:app --reload
```

Adds a `"debug"` field to every response showing the route decision, tool called, and number of RAG chunks retrieved.

---

## API

### `POST /query`

```json
// Request
{ "query": "Why is my bill higher than last month for C003?" }

// Response
{
  "answer": "Your usage jumped from 890 kWh to 1,240 kWh — a 39.3% increase...",
  "source": "tool+rag",
  "confidence": "HIGH"
}
```

`customer_id` can be included in the query text (`"for C003"`) or passed explicitly as `"customer_id": "C003"`.

### Example queries

```bash
BASE="http://127.0.0.1:8000/query"

# Bill summary (tool)
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"query": "What is my total bill?", "customer_id": "C001"}' | python3 -m json.tool

# Usage comparison (tool)
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"query": "How does my usage compare to last month for C003?"}' | python3 -m json.tool

# Charge breakdown (tool)
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"query": "Break down my charges for C002"}' | python3 -m json.tool

# Policy question (rag)
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"query": "What is peak pricing and when does it apply?"}' | python3 -m json.tool

# Tips (rag)
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"query": "How can I reduce my electricity bill?"}' | python3 -m json.tool

# Mixed intent (tool + rag)
curl -s -X POST $BASE -H "Content-Type: application/json" \
  -d '{"query": "Why is my bill higher than last month for C003?"}' | python3 -m json.tool
```

---

## Mock Data

**Customers** (in `data/billing_data.json`): C001 Alice, C002 Bob, C003 Carol, C004 David — each with current/previous usage, peak/off-peak split, and itemized charges.

**Policy docs** (in `data/docs.txt`): 7 sections — bill calculation, peak pricing, reasons for high bills, how to reduce usage, charge breakdown explained, dispute process, payment options.

---

## License

Prototype project — no production license yet.
