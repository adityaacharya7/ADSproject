"""Application service shared by the dashboard and API."""
from datetime import timedelta
import re
from typing import Any

from model_engine import EmotionEngine, get_engine
from pii import mask_pii
from ticket_store import TicketStore, utcnow


SLA_MINUTES = {"CRITICAL": 15, "HIGH": 60, "MEDIUM": 240, "LOW": 1440}
SAFETY_PATTERN = re.compile(
    r"\b(suicid(?:e|al)|self[- ]harm|medical emergency|life[- ]threatening|"
    r"account (?:was |is |has been )?hacked|"
    r"identity theft|fraudulent transaction|card stolen)\b", re.I)


class TriageService:
    def __init__(self, store: TicketStore | None = None, engine: EmotionEngine | None = None):
        self.store = store or TicketStore()
        self.engine = engine or get_engine()

    def create_ticket(self, text: str, external_id: str | None = None,
                      source: str = "manual", actor: str = "system") -> dict[str, Any]:
        raw_text = str(text or "").strip()
        masked = mask_pii(raw_text).strip()
        if not masked:
            raise ValueError("Ticket message cannot be empty")
        if external_id:
            existing = self.store.get_by_external_id(source, external_id)
            if existing:
                return existing
        prediction = self.engine.predict(masked)
        safety_match = SAFETY_PATTERN.search(raw_text)
        urgency = "CRITICAL" if safety_match else prediction["urgency"]
        routing = prediction["routing_recommendation"]
        if safety_match:
            routing = f"Safety escalation ({safety_match.group(0)}): immediate human review (<15 min SLA)"
        elif prediction["confidence"] < 0.60:
            routing = f"Human review required due to low confidence. {routing}"
        created_at = utcnow()
        values = {
            "external_id": external_id or None,
            "message": masked,
            "source": source,
            "primary_emotion": prediction["primary_emotion"],
            "confidence": prediction["confidence"],
            "sentiment_label": prediction["sentiment"]["label"],
            "sentiment_compound": prediction["sentiment"]["compound"],
            "urgency": urgency,
            "routing_recommendation": routing,
            "created_at": created_at,
            "sla_due_at": created_at + timedelta(minutes=SLA_MINUTES[urgency]),
        }
        return self.store.create(values, actor=actor)
