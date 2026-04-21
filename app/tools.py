import json
import os
from typing import Optional

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "billing_data.json")

with open(_DATA_PATH) as f:
    _BILLING_DB = json.load(f)["customers"]


def _get_customer(customer_id: str) -> Optional[dict]:
    return _BILLING_DB.get(customer_id.upper())


def _not_found(tool: str, customer_id: str) -> dict:
    return {
        "tool": tool,
        "status": "error",
        "message": f"No customer found with ID '{customer_id}'. Valid IDs: C001–C004.",
    }


def get_bill(customer_id: str) -> dict:
    """Return the full bill summary for a customer."""
    customer = _get_customer(customer_id)
    if not customer:
        return _not_found("get_bill", customer_id)

    charges = customer["charges"]
    return {
        "tool": "get_bill",
        "status": "ok",
        "data": {
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "billing_period": customer["billing_period"],
            "total_usage_kwh": customer["usage"]["current_kwh"],
            "breakdown": {
                "energy_charge_usd": charges["energy_charge"],
                "peak_surcharge_usd": charges["peak_surcharge"],
                "tax_usd": charges["tax"],
                "total_usd": charges["total"],
            },
        },
    }


def compare_usage(customer_id: str) -> dict:
    """Compare current vs previous month usage and label the trend."""
    customer = _get_customer(customer_id)
    if not customer:
        return _not_found("compare_usage", customer_id)

    current = customer["usage"]["current_kwh"]
    previous = customer["usage"]["previous_kwh"]
    delta = current - previous
    pct_change = round((delta / previous) * 100, 1) if previous else 0.0

    if pct_change > 20:
        trend = "significantly_higher"
    elif pct_change > 5:
        trend = "slightly_higher"
    elif pct_change < -5:
        trend = "lower"
    else:
        trend = "stable"

    return {
        "tool": "compare_usage",
        "status": "ok",
        "data": {
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "billing_period": customer["billing_period"],
            "current_usage_kwh": current,
            "previous_usage_kwh": previous,
            "delta_kwh": delta,
            "percent_change": pct_change,
            "trend": trend,
        },
    }


def explain_charges(customer_id: str) -> dict:
    """Return an itemised breakdown of all charge components including peak/off-peak split."""
    customer = _get_customer(customer_id)
    if not customer:
        return _not_found("explain_charges", customer_id)

    usage = customer["usage"]
    rate = customer["rate"]
    charges = customer["charges"]

    return {
        "tool": "explain_charges",
        "status": "ok",
        "data": {
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "billing_period": customer["billing_period"],
            "usage_breakdown": {
                "total_kwh": usage["current_kwh"],
                "peak_hours_kwh": usage["peak_hours_kwh"],
                "off_peak_hours_kwh": usage["off_peak_hours_kwh"],
            },
            "rates_applied": {
                "base_rate_usd_per_kwh": rate["base_rate_per_kwh"],
                "peak_rate_usd_per_kwh": rate["peak_rate_per_kwh"],
                "peak_surcharge_rate": round(rate["peak_rate_per_kwh"] - rate["base_rate_per_kwh"], 2),
                "tax_rate_percent": rate["tax_rate_percent"],
            },
            "charge_breakdown": {
                "energy_charge_usd": charges["energy_charge"],
                "peak_surcharge_usd": charges["peak_surcharge"],
                "subtotal_usd": charges["subtotal"],
                "tax_usd": charges["tax"],
                "total_usd": charges["total"],
            },
        },
    }


TOOL_REGISTRY = {
    "get_bill": get_bill,
    "compare_usage": compare_usage,
    "explain_charges": explain_charges,
}
