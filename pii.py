"""PII masking used before customer messages are persisted or logged."""
import re


PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[PAYMENT_CARD]"),
    (re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)"), "[PHONE]"),
    (re.compile(r"\b(?:booking|reservation|account|order|ticket)\s*(?:id|number|no\.?|#)?\s*[:#-]?\s*"
                r"(?=[A-Z0-9-]*\d)[A-Z0-9-]{5,}\b", re.I), "[REFERENCE]"),
)


def mask_pii(text: str) -> str:
    value = str(text or "")
    for pattern, replacement in PATTERNS:
        value = pattern.sub(replacement, value)
    return value
