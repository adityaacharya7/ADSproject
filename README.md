# Customer Support Triage Operations

A working support-operations dashboard and REST API built around a LightGBM, negation-aware TF-IDF, and VADER emotion model trained on the Twitter Customer Support dataset.

## What the application does

- Stores a persistent ticket queue in SQLite locally or PostgreSQL in production.
- Classifies emotion and sentiment, assigns urgency, and calculates an SLA deadline.
- Escalates explicit safety phrases and sends low-confidence predictions for human review.
- Lets agents assign, progress, correct, resolve, and annotate tickets.
- Records an audit event for every creation and update.
- Imports CSV queues with selectable ticket-ID and message columns and exports triaged results.
- Masks common email, phone, payment-card, and customer-reference patterns before storage.
- Reports open and overdue tickets, SLA compliance, resolution time, corrections, confidence, and urgency mix.
- Monitors text-length and sentiment drift against a fixed 5,000-message TWCS reference sample.
- Exposes the same workflow through a FastAPI service.

The model recognizes `Joy / Gratitude`, `Anger / Frustration`, `Disappointment / Sadness`, `Fear / Anxiety`, and `Neutral / Inquiry`.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. It creates `support_tickets.db` on first use.

Run the API in another terminal:

```bash
uvicorn api:app --reload --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy database URL; use PostgreSQL for durable cloud storage | `sqlite:///support_tickets.db` |
| `APP_PASSWORD` | Enables the dashboard's shared-password gate when set | Disabled |
| `API_KEY` | Requires an `X-API-Key` header on API operations when set | Disabled |

The dashboard password is a minimal deployment gate. Put the application behind an identity-aware proxy or SSO provider when individual identities and roles are required.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check model and database readiness |
| `POST` | `/predict` | Score and persist one ticket |
| `POST` | `/batch-predict` | Score and persist up to 500 tickets |
| `GET` | `/tickets` | Filter the queue by status, urgency, or assignee |
| `GET` | `/tickets/{id}` | Read a ticket and its audit history |
| `PATCH` | `/tickets/{id}` | Assign, correct, progress, or resolve a ticket |
| `GET` | `/metrics` | Read operational queue metrics |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"external_id":"CRM-123","text":"My account was hacked and I need urgent help"}'
```

## PostgreSQL and Railway

For production, provision PostgreSQL and set `DATABASE_URL` on both services. Deploy the repository twice:

1. Dashboard service: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
2. API service: `uvicorn api:app --host 0.0.0.0 --port=$PORT`

Set the same `DATABASE_URL` for both. Set `APP_PASSWORD` on the dashboard and `API_KEY` on the API. The included `railway.json` starts the dashboard by default; override the start command for the API service.

## Project structure

```text
app.py                  Streamlit operations dashboard
api.py                  FastAPI service
ticket_store.py         Ticket and audit-event persistence
triage_service.py       Shared inference, SLA, and escalation workflow
pii.py                  PII masking
model_engine.py         Model inference and drift monitoring
text_preprocessing.py   Training-compatible text preprocessing
build_drift_baseline.py Rebuilds aggregate drift reference data
models/                 Serialized model and drift reference
test_operational.py     Queue and API workflow tests
test_regressions.py     Model, drift, and dashboard regression tests
```

## Verification

```bash
python -m unittest discover -v
```

The tests exercise the trained artifact, dashboard rendering, PII masking, SLA calculations, safety escalation, persistence, agent corrections, audit events, drift behavior, and API operations.

## Operational boundaries

The application masks common PII patterns but is not a complete data-loss-prevention system. Review retention, encryption, identity, regional hosting, and incident-response requirements before processing real customer data. Human corrections are retained as the quality signal for later evaluation and retraining.

See [Responsible_AI.md](Responsible_AI.md) for the governance checklist.
