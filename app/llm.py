import json
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """You are a helpful utility billing assistant for a fictional electricity company.

You help customers with their electricity bills, usage, charges, and billing policies.

When context is provided, use it. When it is absent (e.g. greetings, small talk, capability questions), respond naturally and conversationally.

Rules:
1. For billing questions: base your answer ONLY on provided [TOOL DATA] and [POLICY DOCS]. Do not invent numbers, rates, or policies.
2. For greetings or "what can you do" questions: respond warmly and list what you can help with (bills, usage comparison, charge breakdowns, peak pricing, tips to reduce bills).
3. If a billing question has no relevant context: say you don't have that information and suggest they try one of the available options.
4. Use plain, friendly language. Quote exact dollar and kWh figures from the data.
5. Keep answers concise unless a detailed breakdown is requested.
6. End every response with exactly one of these on its own line:
   CONFIDENCE: HIGH   — direct data fully answers the question
   CONFIDENCE: MEDIUM — partial data or inference from policy required
   CONFIDENCE: LOW    — context is insufficient or only loosely related
"""


# ── Mock mode (no API key) ────────────────────────────────────────────────────

def _mock_generate(tool_data: Optional[dict], rag_docs: Optional[list]) -> dict:
    """
    Template-based response builder used when no API key is set.
    Routing, tools, and RAG still run — only the final synthesis is mocked.
    """
    if tool_data and tool_data.get("status") == "ok":
        data = tool_data["data"]
        tool = tool_data.get("tool")

        if tool == "get_bill":
            bd = data["breakdown"]
            answer = (
                f"Hi {data['name']}! For {data['billing_period']}, your total bill is "
                f"**\\${bd['total_usd']:.2f}**. "
                f"That breaks down as \\${bd['energy_charge_usd']:.2f} in energy charges, "
                f"\\${bd['peak_surcharge_usd']:.2f} peak surcharge, and \\${bd['tax_usd']:.2f} tax. "
                f"Total usage was {data['total_usage_kwh']} kWh."
            )
            return {"answer": answer, "confidence": "HIGH"}

        if tool == "compare_usage":
            delta = data["delta_kwh"]
            direction = "higher" if delta > 0 else "lower"
            answer = (
                f"Hi {data['name']}! Your usage this {data['billing_period']} was "
                f"**{data['current_usage_kwh']} kWh**, compared to {data['previous_usage_kwh']} kWh "
                f"last month — that's **{abs(delta)} kWh {direction}** "
                f"({abs(data['percent_change'])}% change). "
                f"Trend: {data['trend'].replace('_', ' ')}."
            )
            return {"answer": answer, "confidence": "HIGH"}

        if tool == "explain_charges":
            ub = data["usage_breakdown"]
            r  = data["rates_applied"]
            cb = data["charge_breakdown"]
            answer = (
                f"Hi {data['name']}! Here's your charge breakdown for {data['billing_period']}. "
                f"You used **{ub['total_kwh']} kWh** total — {ub['peak_hours_kwh']} kWh during peak hours "
                f"and {ub['off_peak_hours_kwh']} kWh off-peak.\n\n"
                f"- Base rate: \\${r['base_rate_usd_per_kwh']}/kWh\n"
                f"- Peak surcharge rate: \\${r['peak_surcharge_rate']}/kWh\n"
                f"- Energy charge: \\${cb['energy_charge_usd']:.2f}\n"
                f"- Peak surcharge: \\${cb['peak_surcharge_usd']:.2f}\n"
                f"- Tax: \\${cb['tax_usd']:.2f}\n"
                f"- **Total: \\${cb['total_usd']:.2f}**"
            )
            return {"answer": answer, "confidence": "HIGH"}

        data_summary = json.dumps(data, indent=2)
        return {"answer": f"Here is your account data:\n```\n{data_summary}\n```", "confidence": "MEDIUM"}

    if tool_data and tool_data.get("status") == "error":
        return {"answer": tool_data.get("message", "Customer not found."), "confidence": "LOW"}

    if rag_docs:
        top = rag_docs[0]
        sentences = [s.strip() for s in top["content"].replace("\n", " ").split(". ") if s.strip()]
        snippet = ". ".join(sentences[:2]) + ("." if sentences else "")
        answer = f"**{top['title']}** — {snippet}"
        if len(rag_docs) > 1:
            answer += f"\n\n*(Also relevant: {rag_docs[1]['title']})*"
        confidence = "HIGH" if top["score"] > 0.2 else "MEDIUM"
        return {"answer": answer, "confidence": confidence}

    return {
        "answer": (
            "Hi there! I'm your utility billing assistant. Here's what I can help you with:\n\n"
            "- **Bill summary** — your total due and breakdown\n"
            "- **Usage comparison** — current vs. last month\n"
            "- **Charge breakdown** — peak/off-peak split, rates, tax\n"
            "- **Peak pricing** — when it applies and how to avoid it\n"
            "- **Tips to reduce your bill**\n\n"
            "Select a customer from the sidebar and ask me anything, or try one of the example buttons."
        ),
        "confidence": "HIGH",
    }


# ── Gemini LLM mode ───────────────────────────────────────────────────────────

def _build_context(tool_data: Optional[dict], rag_docs: Optional[list]) -> str:
    parts = []
    if tool_data:
        if tool_data.get("status") == "ok":
            parts.append("[TOOL DATA]\n" + json.dumps(tool_data["data"], indent=2))
        else:
            parts.append(f"[TOOL DATA]\nError: {tool_data.get('message', 'Unknown error')}")
    if rag_docs:
        doc_text = "\n\n".join(f"--- {d['title']} ---\n{d['content']}" for d in rag_docs)
        parts.append("[POLICY DOCS]\n" + doc_text)
    return "\n\n".join(parts) if parts else "[No relevant context available]"


def _parse_confidence(raw: str) -> tuple[str, str]:
    """Strip the CONFIDENCE label from the model's response and return it separately."""
    lines = raw.strip().splitlines()
    confidence = "LOW"
    clean = []
    found = False
    for line in reversed(lines):
        upper = line.strip().upper()
        if not found and upper.startswith("CONFIDENCE:"):
            label = upper.replace("CONFIDENCE:", "").strip()
            if label in ("HIGH", "MEDIUM", "LOW"):
                confidence = label
                found = True
                continue
        clean.insert(0, line)
    return "\n".join(clean).strip(), confidence


def generate_answer(
    query: str,
    tool_data: Optional[dict] = None,
    rag_docs: Optional[list] = None,
    history: Optional[list] = None,  # [{"role": "user"|"assistant", "content": str}, ...]
) -> dict:
    """Return { answer, confidence }. Uses mock templates if no API key is set."""
    if not _API_KEY:
        return _mock_generate(tool_data, rag_docs)

    import google.generativeai as genai

    genai.configure(api_key=_API_KEY)
    model = genai.GenerativeModel(model_name=_MODEL, system_instruction=_SYSTEM_PROMPT)

    # Gemini uses "model" instead of "assistant" for role names
    gemini_history = []
    for msg in (history or [])[-6:]:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    current_msg = (
        f"Customer question: {query}\n\n"
        f"Context:\n{_build_context(tool_data, rag_docs)}\n\n"
        "Answer using the context above and our conversation so far."
    )

    try:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(current_msg)
        answer, confidence = _parse_confidence(response.text)
        return {"answer": answer, "confidence": confidence}
    except Exception:
        return _mock_generate(tool_data, rag_docs)
