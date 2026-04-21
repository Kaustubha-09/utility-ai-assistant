import re
from typing import Optional

# Keywords that signal the user wants live account data
_TOOL_SIGNALS = {
    "bill", "balance", "due", "owe", "charge", "charges", "usage",
    "kwh", "kilowatt", "amount", "total", "invoice", "statement",
    "consumption", "compare", "previous", "last month", "this month",
    "peak hours", "off-peak", "breakdown", "how much",
}

# Keywords that signal the user wants a policy explanation or general advice
_RAG_SIGNALS = {
    "why", "explain", "how does", "what is", "what are", "policy",
    "reason", "high bill", "reduce", "lower", "save", "tip", "advice",
    "peak pricing", "peak rate", "pricing", "dispute", "payment",
    "late fee", "autopay", "credit", "calculate", "calculated",
}

# Hints for picking the right tool; checked in order, first match wins
_TOOL_HINTS = {
    "compare": "compare_usage",
    "comparison": "compare_usage",
    "increase": "compare_usage",
    "higher": "compare_usage",
    "lower": "compare_usage",
    "change": "compare_usage",
    "explain charges": "explain_charges",
    "explain my charges": "explain_charges",
    "breakdown": "explain_charges",
    "break down": "explain_charges",
    "itemize": "explain_charges",
}

_DEFAULT_TOOL = "get_bill"


def extract_customer_id(query: str) -> Optional[str]:
    """Extract a customer ID (C001–C999) from the query, or return None."""
    match = re.search(r"\bC\d{3,4}\b", query, re.IGNORECASE)
    return match.group(0).upper() if match else None


def route_query(query: str) -> dict:
    """
    Classify query intent and return routing metadata.

    Returns: { "mode": "tool"|"rag"|"both", "tool_name": str|None, "customer_id": str|None }
    """
    lower = query.lower()

    tool_score = sum(1 for kw in _TOOL_SIGNALS if kw in lower)
    rag_score = sum(1 for kw in _RAG_SIGNALS if kw in lower)

    if tool_score > 0 and rag_score > 0:
        mode = "both"
    elif tool_score > 0:
        mode = "tool"
    else:
        # Covers both pure RAG queries and ambiguous ones — safer than guessing tool calls
        mode = "rag"

    tool_name = None
    if mode in ("tool", "both"):
        tool_name = next(
            (name for hint, name in _TOOL_HINTS.items() if hint in lower),
            _DEFAULT_TOOL,
        )

    return {
        "mode": mode,
        "tool_name": tool_name,
        "customer_id": extract_customer_id(query),
    }
