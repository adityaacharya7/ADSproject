"""Persistent ticket queue backed by SQLite locally or PostgreSQL in production."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    external_id: Mapped[str | None] = mapped_column(String(120), index=True)
    message: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(40), default="manual", index=True)
    status: Mapped[str] = mapped_column(String(30), default="OPEN", index=True)
    assigned_to: Mapped[str | None] = mapped_column(String(120), index=True)
    primary_emotion: Mapped[str] = mapped_column(String(80), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    sentiment_label: Mapped[str] = mapped_column(String(20))
    sentiment_compound: Mapped[float] = mapped_column(Float)
    urgency: Mapped[str] = mapped_column(String(20), index=True)
    routing_recommendation: Mapped[str] = mapped_column(Text)
    corrected_emotion: Mapped[str | None] = mapped_column(String(80))
    corrected_urgency: Mapped[str | None] = mapped_column(String(20))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(80), default="best_emotion_model.joblib")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    sla_due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TicketEvent(Base):
    __tablename__ = "ticket_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(36), index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    detail: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TicketStore:
    def __init__(self, database_url: str | None = None):
        database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///support_tickets.db")
        if database_url.startswith("postgres://"):
            database_url = "postgresql+psycopg://" + database_url.removeprefix("postgres://")
        elif database_url.startswith("postgresql://") and not database_url.startswith("postgresql+"):
            database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
        kwargs = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=kwargs, pool_pre_ping=True)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @staticmethod
    def _row(ticket: Ticket) -> dict[str, Any]:
        return {column.name: getattr(ticket, column.name) for column in Ticket.__table__.columns}

    def create(self, values: dict[str, Any], actor: str = "system") -> dict[str, Any]:
        with self.Session.begin() as session:
            ticket = Ticket(**values)
            session.add(ticket)
            session.flush()
            session.add(TicketEvent(ticket_id=ticket.id, event_type="CREATED", actor=actor,
                                    detail=f"Predicted {ticket.primary_emotion}; {ticket.urgency}"))
        return self._row(ticket)

    def list(self, status: str | None = None, urgency: str | None = None,
             assigned_to: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        query = select(Ticket)
        if status and status != "ALL":
            query = query.where(Ticket.status == status)
        if urgency and urgency != "ALL":
            query = query.where(Ticket.urgency == urgency)
        if assigned_to:
            query = query.where(Ticket.assigned_to == assigned_to)
        query = query.order_by(Ticket.sla_due_at.asc()).limit(min(limit, 1000))
        with self.Session() as session:
            return [self._row(t) for t in session.scalars(query)]

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        with self.Session() as session:
            ticket = session.get(Ticket, ticket_id)
            return self._row(ticket) if ticket else None

    def get_by_external_id(self, source: str, external_id: str) -> dict[str, Any] | None:
        query = select(Ticket).where(Ticket.source == source, Ticket.external_id == external_id)
        with self.Session() as session:
            ticket = session.scalar(query)
            return self._row(ticket) if ticket else None

    def update(self, ticket_id: str, changes: dict[str, Any], actor: str = "agent") -> dict[str, Any] | None:
        allowed = {"status", "assigned_to", "corrected_emotion", "corrected_urgency", "resolution_note"}
        changes = {key: value for key, value in changes.items() if key in allowed}
        with self.Session.begin() as session:
            ticket = session.get(Ticket, ticket_id)
            if not ticket:
                return None
            before = {key: getattr(ticket, key) for key in changes}
            for key, value in changes.items():
                setattr(ticket, key, value or None)
            if ticket.status == "RESOLVED" and not ticket.resolved_at:
                ticket.resolved_at = utcnow()
            elif ticket.status != "RESOLVED":
                ticket.resolved_at = None
            ticket.updated_at = utcnow()
            detail = "; ".join(f"{key}: {before[key]} -> {value}" for key, value in changes.items())
            session.add(TicketEvent(ticket_id=ticket.id, event_type="UPDATED", actor=actor, detail=detail))
        return self._row(ticket)

    def events(self, ticket_id: str) -> list[dict[str, Any]]:
        query = select(TicketEvent).where(TicketEvent.ticket_id == ticket_id).order_by(TicketEvent.created_at.desc())
        with self.Session() as session:
            return [{column.name: getattr(event, column.name) for column in TicketEvent.__table__.columns}
                    for event in session.scalars(query)]

    def metrics(self) -> dict[str, Any]:
        now = utcnow()
        with self.Session() as session:
            total = session.scalar(select(func.count()).select_from(Ticket)) or 0
            open_count = session.scalar(select(func.count()).select_from(Ticket).where(Ticket.status != "RESOLVED")) or 0
            overdue = session.scalar(select(func.count()).select_from(Ticket).where(
                Ticket.status != "RESOLVED", Ticket.sla_due_at < now)) or 0
            corrected = session.scalar(select(func.count()).select_from(Ticket).where(
                (Ticket.corrected_emotion.is_not(None)) | (Ticket.corrected_urgency.is_not(None)))) or 0
            low_confidence = session.scalar(select(func.count()).select_from(Ticket).where(Ticket.confidence < 0.60)) or 0
            urgency_rows = session.execute(select(Ticket.urgency, func.count()).group_by(Ticket.urgency)).all()
            resolved_rows = session.execute(select(Ticket.created_at, Ticket.resolved_at, Ticket.sla_due_at).where(
                Ticket.resolved_at.is_not(None))).all()
        resolution_hours = [max(0, (resolved - created).total_seconds()) / 3600
                            for created, resolved, _ in resolved_rows]
        sla_met = [resolved <= due for _, resolved, due in resolved_rows]
        return {
            "total": total,
            "open": open_count,
            "overdue": overdue,
            "correction_rate": corrected / total if total else 0.0,
            "low_confidence_rate": low_confidence / total if total else 0.0,
            "average_resolution_hours": sum(resolution_hours) / len(resolution_hours) if resolution_hours else None,
            "sla_compliance_rate": sum(sla_met) / len(sla_met) if sla_met else None,
            "by_urgency": dict(urgency_rows),
        }

    def recent_messages(self, limit: int = 500) -> list[str]:
        query = select(Ticket.message).order_by(Ticket.created_at.desc()).limit(limit)
        with self.Session() as session:
            return list(session.scalars(query))

    def health(self) -> bool:
        with self.Session() as session:
            session.execute(select(1))
        return True
