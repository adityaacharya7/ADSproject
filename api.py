"""REST API for ticket ingestion, queue management, feedback, and health checks."""
import os
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from triage_service import TriageService

app = FastAPI(title="Customer Support Triage API", version="1.0.0")
service = TriageService()


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


class TicketCreate(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    external_id: str | None = Field(default=None, max_length=120)
    source: str = Field(default="api", max_length=40)


class BatchCreate(BaseModel):
    tickets: list[TicketCreate] = Field(min_length=1, max_length=500)


class TicketUpdate(BaseModel):
    status: Literal["OPEN", "IN_PROGRESS", "ESCALATED", "RESOLVED"] | None = None
    assigned_to: str | None = Field(default=None, max_length=120)
    corrected_emotion: Literal["Anger / Frustration", "Joy / Gratitude", "Disappointment / Sadness",
                               "Fear / Anxiety", "Neutral / Inquiry"] | None = None
    corrected_urgency: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] | None = None
    resolution_note: str | None = Field(default=None, max_length=10_000)


@app.get("/health")
def health():
    database_ok = service.store.health()
    model_ok = service.engine.pipeline is not None
    return {"status": "healthy" if database_ok and model_ok else "degraded",
            "database": database_ok, "model": model_ok}


@app.post("/predict", dependencies=[Depends(require_api_key)])
def create_ticket(payload: TicketCreate):
    try:
        return service.create_ticket(payload.text, payload.external_id, payload.source, actor="api")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/batch-predict", dependencies=[Depends(require_api_key)])
def create_batch(payload: BatchCreate):
    return {"tickets": [service.create_ticket(item.text, item.external_id, item.source, actor="api")
                        for item in payload.tickets]}


@app.get("/tickets", dependencies=[Depends(require_api_key)])
def list_tickets(status: str | None = None, urgency: str | None = None,
                 assigned_to: str | None = None, limit: int = Query(default=200, ge=1, le=1000)):
    return service.store.list(status=status, urgency=urgency, assigned_to=assigned_to, limit=limit)


@app.get("/tickets/{ticket_id}", dependencies=[Depends(require_api_key)])
def get_ticket(ticket_id: str):
    ticket = service.store.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"ticket": ticket, "events": service.store.events(ticket_id)}


@app.patch("/tickets/{ticket_id}", dependencies=[Depends(require_api_key)])
def update_ticket(ticket_id: str, payload: TicketUpdate, x_actor: str = Header(default="api-user")):
    ticket = service.store.update(ticket_id, payload.model_dump(exclude_unset=True), actor=x_actor)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics():
    return service.store.metrics()
