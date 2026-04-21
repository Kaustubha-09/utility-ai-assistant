# Utility AI Assistant

A prototype AI system for electricity billing support, demonstrating four patterns common in production AI systems:

| Pattern | Implementation |
|---|---|
| MCP-style tool calling | Deterministic, structured data access — no guessing |
| RAG pipeline | Policy retrieval via TF-IDF; answers grounded in documents |
| Intelligent routing | Keyword intent classifier sends each query to the right source |
| Reliable AI design | Confidence scoring; "I don't know" fallback; no hallucination |

---

## How It Works

A naive LLM given *"Why is my bill higher than last month?"* would guess at numbers and policies. This system separates that question into two parts — *what are my actual numbers?* (tool) and *what explains them?* (RAG) — and only sends the LLM grounded context to synthesise from.

### Request Flow

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

**`router.py`** — Scores the query against two keyword sets and returns `mode: tool | rag | both`. Also extracts a customer ID with a regex. When nothing matches, defaults to `rag` — safer than hallucinating a tool call.

**`tools.py`** — Three MCP-style functions that read from `billing_data.json` and return structured JSON. The LLM never touches raw data; it only sees the tool's output.
- `get_bill(customer_id)` — total due, energy charge, tax
- `compare_usage(customer_id)` — delta, % change, trend label
- `explain_charges(customer_id)` — peak/off-peak split, rates applied, itemised math

**`rag.py`** — Chunks `docs.txt` by `SECTION:` headers, builds a TF-IDF index at startup, and retrieves the top-k sections by cosine similarity at query time. No embedding API needed.

**`llm.py`** — Assembles a context block from tool output and retrieved docs, then calls Claude Sonnet 4.6. The system prompt enforces grounding (no invented numbers or policies) and instructs the model to append `CONFIDENCE: HIGH/MEDIUM/LOW` which is parsed out as a separate field.

**`main.py`** — FastAPI app that wires everything together. The `source` field in the response tells you exactly where the answer came from: `"tool"`, `"rag"`, or `"tool+rag"`.

### Confidence Levels

| Level | Meaning |
|---|---|
| `HIGH` | Direct account data fully answers the question |
| `MEDIUM` | Partial data, or answer inferred from policy |
| `LOW` | Insufficient context; answer may be incomplete |

---

## Setup

```bash
cd ~/Desktop/utility-ai-assistant

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
# or copy .env.example → .env and fill in the key
```

**Chat UI (Streamlit)**
```bash
streamlit run ui.py
```
Opens at `http://localhost:8501` — chat interface with sidebar examples, color-coded confidence badges, and expandable tool/RAG data.

**REST API (FastAPI)**
```bash
uvicorn app.main:app --reload
```
API at `http://127.0.0.1:8000` · Swagger docs at `/docs`

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

`customer_id` can be included in the query text (e.g. `"for C003"`) or passed explicitly as `"customer_id": "C003"`.

---

## Example Queries

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

## Debug Mode

```bash
DEBUG=true uvicorn app.main:app --reload
```

Adds a `"debug"` field to every response showing the route decision, tool called, and number of RAG chunks retrieved.

---

## Mock Data

**Customers** (in `data/billing_data.json`): C001 Alice, C002 Bob, C003 Carol, C004 David — each with current/previous usage, peak/off-peak split, and itemised charges.

**Policy docs** (in `data/docs.txt`): 7 sections — bill calculation, peak pricing, reasons for high bills, how to reduce usage, charge breakdown explained, dispute process, payment options.
