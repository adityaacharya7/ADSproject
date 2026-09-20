"""Friendly operations dashboard for AI-assisted customer-support triage."""
import html
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from model_engine import EmotionEngine, get_engine
from ticket_store import TicketStore
from triage_service import TriageService

CURRENT_DIR = Path(__file__).resolve().parent
EMOTIONS = EmotionEngine.EMOTIONS
URGENCIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
STATUSES = ["OPEN", "IN_PROGRESS", "ESCALATED", "RESOLVED"]
URGENCY_ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "LOW": "🟢"}
STATUS_LABELS = {
    "OPEN": "Open", "IN_PROGRESS": "In progress", "ESCALATED": "Escalated", "RESOLVED": "Resolved"
}

st.set_page_config(page_title="SupportFlow AI", page_icon="🎧", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1450px; padding-top: 1.35rem; padding-bottom: 3rem}
    [data-testid="stSidebar"] {background:var(--secondary-background-color);border-right:1px solid rgba(148,163,184,.25)}
    .hero {background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 65%,#2563eb 100%);
           padding:1.3rem 1.6rem;border-radius:16px;color:white;margin-bottom:1rem;
           box-shadow:0 8px 28px rgba(30,58,138,.18)}
    .hero h1 {font-size:1.9rem;margin:0}.hero p {color:#dbeafe;margin:.35rem 0 0;font-size:1rem}
    .step {background:var(--secondary-background-color);color:var(--text-color);
           border:1px solid rgba(148,163,184,.28);border-radius:12px;padding:1rem;min-height:128px}
    .step strong {color:var(--primary-color)}.eyebrow {font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.08em;font-weight:700}
    .ticket-message {background:var(--secondary-background-color);color:var(--text-color);
                     border-left:4px solid var(--primary-color);border-radius:8px;
                     padding:1rem 1.1rem;font-size:1.04rem;margin:.4rem 0 1rem}
    .health-ok {display:inline-block;background:#dcfce7;color:#166534;border-radius:999px;
                padding:.18rem .6rem;font-size:.78rem;font-weight:700}
    .health-warn {display:inline-block;background:#fee2e2;color:#991b1b;border-radius:999px;
                  padding:.18rem .6rem;font-size:.78rem;font-weight:700}
    div[data-testid="stMetric"] {background:var(--secondary-background-color);color:var(--text-color);
                                  border:1px solid rgba(148,163,184,.28);border-radius:12px;padding:.8rem}
    div[data-testid="stForm"] {border:1px solid rgba(148,163,184,.28);border-radius:14px;padding:1rem}
</style>
""", unsafe_allow_html=True)


def require_login():
    expected = os.getenv("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.title("🎧 SupportFlow AI")
        st.write("Sign in to access the support queue.")
        password = st.text_input("Dashboard password", type="password")
        if st.button("Sign in", type="primary", width="stretch") and password == expected:
            st.session_state.authenticated = True
            st.rerun()
        if password and password != expected:
            st.error("That password is incorrect.")
    st.stop()


@st.cache_resource
def load_service():
    return TriageService(TicketStore(), get_engine())


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


def display_time(value):
    return aware(value).astimezone().strftime("%d %b %Y, %I:%M %p") if value else "—"


def sla_details(ticket):
    if ticket["status"] == "RESOLVED":
        return "Resolved", False
    seconds = (aware(ticket["sla_due_at"]) - datetime.now(timezone.utc)).total_seconds()
    if seconds < 0:
        return f"Overdue by {abs(seconds) / 60:.0f} min", True
    if seconds < 3600:
        return f"{seconds / 60:.0f} min remaining", False
    return f"{seconds / 3600:.1f} hr remaining", False


def queue_frame(rows):
    records = []
    for row in rows:
        sla, overdue = sla_details(row)
        records.append({
            "Priority": f"{URGENCY_ICONS[row['urgency']]} {row['urgency'].title()}",
            "Ticket": row["external_id"] or row["id"][:8],
            "Customer message": row["message"][:110],
            "Emotion": row["corrected_emotion"] or row["primary_emotion"],
            "Confidence": f"{row['confidence']:.0%}",
            "Status": STATUS_LABELS[row["status"]],
            "SLA": f"⚠️ {sla}" if overdue else sla,
            "Owner": row["assigned_to"] or "Unassigned",
        })
    return pd.DataFrame(records)


def apply_action(ticket_id, changes, success_message):
    service.store.update(ticket_id, changes, actor=st.session_state.agent_name)
    st.toast(success_message, icon="✅")
    st.rerun()


require_login()
service = load_service()
model_ready = service.engine.pipeline is not None

with st.sidebar:
    st.title("🎧 SupportFlow")
    st.caption("AI-assisted ticket triage")
    st.markdown("---")
    st.markdown('<div class="eyebrow">Working as</div>', unsafe_allow_html=True)
    agent_name = st.text_input("Agent name", value=st.session_state.get("agent_name", "Support Agent"),
                               label_visibility="collapsed")
    st.session_state.agent_name = agent_name.strip() or "Support Agent"
    st.markdown("---")
    badge_class = "health-ok" if model_ready else "health-warn"
    badge_text = "● AI model ready" if model_ready else "● AI using fallback rules"
    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    st.caption("Customer messages are masked for common personal information before storage.")
    if os.getenv("APP_PASSWORD") and st.button("Sign out", width="stretch"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("""
<div class="hero"><h1>Customer Support Command Center</h1>
<p>See what needs attention, prioritize every message, and keep customer SLAs on track.</p></div>
""", unsafe_allow_html=True)

home_tab, queue_tab, intake_tab, batch_tab, analytics_tab, admin_tab = st.tabs([
    "🏠 Overview", "🎫 Ticket queue", "➕ Add ticket", "📄 Import CSV", "📊 Analytics", "⚙️ Setup"
])

with home_tab:
    metrics = service.store.metrics()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Needs attention", metrics["open"], help="All tickets that are not resolved")
    h2.metric("Past SLA", metrics["overdue"], help="Open tickets whose response deadline has passed")
    h3.metric("AI review rate", f"{metrics['low_confidence_rate']:.0%}",
              help="Tickets where the model is less than 60% confident")
    h4.metric("SLA success", f"{metrics['sla_compliance_rate']:.0%}"
              if metrics["sla_compliance_rate"] is not None else "No data",
              help="Resolved tickets completed before their SLA deadline")

    st.subheader("What should I work on next?")
    active = [row for row in service.store.list(limit=100) if row["status"] != "RESOLVED"]
    attention = [row for row in active if row["urgency"] in ("CRITICAL", "HIGH") or sla_details(row)[1]][:8]
    if attention:
        st.dataframe(queue_frame(attention), width="stretch", hide_index=True)
        st.caption("Open the Ticket queue tab to claim or update one of these tickets.")
    elif active:
        st.success("No critical, high-priority, or overdue tickets. The active queue is under control.")
    else:
        st.info("Your queue is empty. Add a ticket manually or import a CSV to get started.")

    with st.expander("New here? See how the workflow works", expanded=not bool(metrics["total"])):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown('<div class="step"><strong>1 · Add</strong><br>Paste one customer message or import a CSV support export.</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="step"><strong>2 · Prioritize</strong><br>The AI predicts emotion, urgency, routing, and an SLA deadline.</div>', unsafe_allow_html=True)
        with s3:
            st.markdown('<div class="step"><strong>3 · Review</strong><br>An agent claims the ticket, checks the AI result, and corrects it if needed.</div>', unsafe_allow_html=True)
        with s4:
            st.markdown('<div class="step"><strong>4 · Resolve</strong><br>Actions are audited and operations metrics update automatically.</div>', unsafe_allow_html=True)

with queue_tab:
    q_col1, q_col2 = st.columns([3.5, 1])
    with q_col1:
        st.subheader("Ticket queue")
        st.caption("Tickets are ordered by the closest SLA deadline. Use filters to focus your work.")
    with q_col2:
        if st.button("🔄 Refresh queue", width="stretch", help="Reload the latest tickets from the database"):
            st.rerun()
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.1, 1, 1, 1.2])
        with f1:
            status_filter = st.selectbox("Show", ["Active tickets", "All tickets", *STATUS_LABELS.values()])
        with f2:
            urgency_filter = st.selectbox("Priority", ["All priorities", *[u.title() for u in URGENCIES]])
        with f3:
            ownership = st.selectbox("Owner", ["Everyone", "Assigned to me", "Unassigned"])
        with f4:
            search = st.text_input("Search", placeholder="ID or message keywords")

    rows = service.store.list(limit=1000)
    if status_filter == "Active tickets":
        rows = [row for row in rows if row["status"] != "RESOLVED"]
    elif status_filter not in ("All tickets", "Active tickets"):
        selected_status = next(key for key, value in STATUS_LABELS.items() if value == status_filter)
        rows = [row for row in rows if row["status"] == selected_status]
    if urgency_filter != "All priorities":
        rows = [row for row in rows if row["urgency"] == urgency_filter.upper()]
    if ownership == "Assigned to me":
        rows = [row for row in rows if row["assigned_to"] == st.session_state.agent_name]
    elif ownership == "Unassigned":
        rows = [row for row in rows if not row["assigned_to"]]
    if search.strip():
        needle = search.casefold().strip()
        rows = [row for row in rows if needle in row["message"].casefold()
                or needle in (row["external_id"] or "").casefold() or needle in row["id"].casefold()]

    st.caption(f"{len(rows)} ticket{'s' if len(rows) != 1 else ''} shown")
    if not rows:
        st.info("No tickets match these filters.")
    else:
        st.dataframe(queue_frame(rows), width="stretch", hide_index=True)
        selected_id = st.selectbox(
            "Choose a ticket to open",
            [row["id"] for row in rows],
            format_func=lambda value: next(
                f"{URGENCY_ICONS[row['urgency']]} {row['external_id'] or row['id'][:8]} · {row['message'][:75]}"
                for row in rows if row["id"] == value),
        )
        ticket = service.store.get(selected_id)
        st.markdown("---")
        title_id = ticket["external_id"] or ticket["id"][:8]
        st.subheader(f"{URGENCY_ICONS[ticket['urgency']]} Ticket {title_id}")
        quick1, quick2, quick3, quick4 = st.columns(4)
        if quick1.button("Claim ticket", disabled=ticket["assigned_to"] == st.session_state.agent_name,
                         width="stretch"):
            apply_action(ticket["id"], {"assigned_to": st.session_state.agent_name}, "Ticket assigned to you")
        if quick2.button("Start work", disabled=ticket["status"] == "IN_PROGRESS", width="stretch"):
            apply_action(ticket["id"], {"status": "IN_PROGRESS", "assigned_to": st.session_state.agent_name},
                         "Ticket moved to In progress")
        if quick3.button("Escalate", disabled=ticket["status"] == "ESCALATED", width="stretch"):
            apply_action(ticket["id"], {"status": "ESCALATED"}, "Ticket escalated")
        if quick4.button("Mark resolved", disabled=ticket["status"] == "RESOLVED", width="stretch"):
            apply_action(ticket["id"], {"status": "RESOLVED"}, "Ticket resolved")

        detail_left, detail_right = st.columns([1.5, 1])
        with detail_left:
            st.markdown('<div class="eyebrow">Customer message</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ticket-message">{html.escape(ticket["message"])}</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("AI emotion", ticket["primary_emotion"])
            d2.metric("AI confidence", f"{ticket['confidence']:.0%}")
            d3.metric("Sentiment", ticket["sentiment_label"])
            sla_text, overdue = sla_details(ticket)
            if overdue:
                st.error(f"SLA: {sla_text} · Due {display_time(ticket['sla_due_at'])}")
            else:
                st.info(f"SLA: {sla_text} · Due {display_time(ticket['sla_due_at'])}")
            st.write("**Recommended handling:**", ticket["routing_recommendation"])
            with st.expander("View audit history"):
                history = pd.DataFrame(service.store.events(ticket["id"]))
                st.dataframe(history, width="stretch", hide_index=True)
        with detail_right:
            st.markdown("#### Review and update")
            st.caption("Correcting an AI result helps measure model quality.")
            with st.form("ticket_update"):
                status = st.selectbox("Ticket status", STATUSES, index=STATUSES.index(ticket["status"]),
                                      format_func=lambda value: STATUS_LABELS[value])
                assigned = st.text_input("Assigned agent", value=ticket["assigned_to"] or "")
                corrected_emotion = st.selectbox(
                    "Emotion review", ["AI prediction is correct", *EMOTIONS],
                    index=EMOTIONS.index(ticket["corrected_emotion"]) + 1
                    if ticket["corrected_emotion"] in EMOTIONS else 0)
                corrected_urgency = st.selectbox(
                    "Priority review", ["AI priority is correct", *URGENCIES],
                    index=URGENCIES.index(ticket["corrected_urgency"]) + 1
                    if ticket["corrected_urgency"] in URGENCIES else 0,
                    format_func=lambda value: value.title() if value in URGENCIES else value)
                note = st.text_area("Agent notes", value=ticket["resolution_note"] or "",
                                    placeholder="Add context, customer outcome, or resolution details…")
                if st.form_submit_button("Save ticket", type="primary", width="stretch"):
                    service.store.update(ticket["id"], {
                        "status": status, "assigned_to": assigned,
                        "corrected_emotion": None if corrected_emotion == "AI prediction is correct" else corrected_emotion,
                        "corrected_urgency": None if corrected_urgency == "AI priority is correct" else corrected_urgency,
                        "resolution_note": note,
                    }, actor=st.session_state.agent_name)
                    st.success("Changes saved")
                    st.rerun()

with intake_tab:
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Add one customer ticket")
        st.caption("The message is masked for common personal information, analyzed, and added to the queue.")
        examples = {
            "Write my own": "",
            "Account security": "My account was hacked and I do not recognize these transactions.",
            "Delivery problem": "My package arrived broken and I am very disappointed.",
            "General question": "Can you tell me when my subscription renews?",
            "Positive feedback": "Thank you! Your agent solved my issue quickly.",
        }
        example = st.selectbox("Try an example (optional)", list(examples))
        with st.form("new_ticket", clear_on_submit=False):
            external_id = st.text_input("Ticket ID", placeholder="Optional, for example CRM-1042",
                                        help="Repeated imports with the same source and ID will not create a duplicate.")
            message = st.text_area("Customer message", value=examples[example], height=180,
                                   placeholder="Paste the customer's message here…")
            submitted = st.form_submit_button("Analyze and add to queue", type="primary", width="stretch")
        if submitted:
            try:
                created = service.create_ticket(message, external_id, "manual", st.session_state.agent_name)
                st.session_state.last_created_ticket = created
                st.session_state.ticket_just_added = True
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.session_state.pop("ticket_just_added", False):
            st.success("Ticket added to the queue.")
    with right:
        st.subheader("Triage result")
        created = st.session_state.get("last_created_ticket")
        if created:
            st.metric("Priority", f"{URGENCY_ICONS[created['urgency']]} {created['urgency'].title()}")
            st.metric("Emotion", created["primary_emotion"])
            st.metric("Confidence", f"{created['confidence']:.0%}")
            st.info(f"Respond by {display_time(created['sla_due_at'])}")
            st.write(created["routing_recommendation"])
        else:
            st.info("Submit a message to see its emotion, confidence, priority, and SLA here.")

with batch_tab:
    st.subheader("Import a CSV ticket queue")
    st.caption("Upload an export from your support system, choose the matching columns, then review the first rows before importing.")
    template = pd.DataFrame({"ticket_id": ["CRM-1001", "CRM-1002"],
                             "message": ["My order is late", "Thank you for the quick help"]})
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       "ticket_import_template.csv", "text/csv")
    upload = st.file_uploader("Step 1 · Choose a CSV file", type=["csv"])
    if upload:
        try:
            incoming = pd.read_csv(upload)
            if incoming.empty:
                st.warning("This CSV has column headers but no ticket rows.")
            else:
                st.write(f"**Step 2 · Match columns** — {len(incoming):,} rows found")
                map1, map2 = st.columns(2)
                columns = list(incoming.columns)
                likely_text = next((i for i, name in enumerate(columns)
                                    if name.casefold() in {"message", "text", "description", "body"}), 0)
                with map1:
                    text_column = st.selectbox("Customer message column", columns, index=likely_text)
                with map2:
                    id_column = st.selectbox("Ticket ID column", ["No ID column", *columns])
                st.write("**Step 3 · Check the preview**")
                preview_columns = [text_column] if id_column == "No ID column" else [id_column, text_column]
                st.dataframe(incoming[preview_columns].head(10), width="stretch", hide_index=True)
                if len(incoming) > 1000:
                    st.warning("The dashboard processes the first 1,000 rows per import. Use the API for larger queues.")
                if st.button("Import tickets", type="primary"):
                    import_rows = incoming.head(1000)
                    progress = st.progress(0, text="Analyzing tickets…")
                    created_count, duplicate_count, errors = 0, 0, []
                    for position, (_, row) in enumerate(import_rows.iterrows(), start=1):
                        try:
                            if pd.isna(row[text_column]) or not str(row[text_column]).strip():
                                raise ValueError("Message is empty")
                            external = None if id_column == "No ID column" or pd.isna(row[id_column]) else str(row[id_column])
                            before = service.store.get_by_external_id("csv", external) if external else None
                            service.create_ticket(str(row[text_column]), external, "csv", st.session_state.agent_name)
                            duplicate_count += int(before is not None)
                            created_count += int(before is None)
                        except Exception as exc:
                            errors.append({"CSV row": position + 1, "Problem": str(exc)})
                        progress.progress(position / len(import_rows), text=f"Analyzed {position} of {len(import_rows)}")
                    st.success(f"Import complete: {created_count} added, {duplicate_count} duplicates skipped, {len(errors)} failed.")
                    st.button("🔄 View imported tickets in queue", on_click=st.rerun)
                    if errors:
                        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"This file could not be read as CSV: {exc}")
    existing = service.store.list(limit=1000)
    if existing:
        st.markdown("---")
        st.download_button("Export current queue", pd.DataFrame(existing).to_csv(index=False).encode("utf-8"),
                           "triaged_queue.csv", "text/csv")

with analytics_tab:
    st.subheader("Operations and AI quality")
    metrics = service.store.metrics()
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Total", metrics["total"])
    a2.metric("Active", metrics["open"])
    a3.metric("Overdue", metrics["overdue"])
    a4.metric("SLA success", f"{metrics['sla_compliance_rate']:.1%}"
              if metrics["sla_compliance_rate"] is not None else "—")
    a5.metric("Avg resolution", f"{metrics['average_resolution_hours']:.1f} hr"
              if metrics["average_resolution_hours"] is not None else "—")
    a6.metric("AI corrected", f"{metrics['correction_rate']:.1%}")

    all_tickets = service.store.list(limit=1000)
    if all_tickets:
        chart1, chart2 = st.columns(2)
        urgency_df = pd.DataFrame({"Priority": [key.title() for key in metrics["by_urgency"]],
                                   "Tickets": list(metrics["by_urgency"].values())})
        with chart1:
            fig = px.bar(urgency_df, x="Priority", y="Tickets", color="Priority",
                         category_orders={"Priority": [u.title() for u in URGENCIES]},
                         color_discrete_map={"Critical": "#dc2626", "High": "#f97316",
                                             "Medium": "#2563eb", "Low": "#16a34a"},
                         title="Tickets by priority")
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with chart2:
            status_counts = pd.Series([STATUS_LABELS[row["status"]] for row in all_tickets]).value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, hole=.58,
                         title="Queue by status", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### AI health")
        q1, q2 = st.columns(2)
        q1.metric("Low-confidence tickets", f"{metrics['low_confidence_rate']:.1%}",
                  help="Share of tickets below 60% confidence")
        q2.metric("Human correction rate", f"{metrics['correction_rate']:.1%}",
                  help="Share of tickets where an agent corrected emotion or priority")
        messages = service.store.recent_messages()
        drift = service.engine.check_data_drift(messages)
        with st.expander("Data drift details"):
            d1, d2, d3 = st.columns(3)
            d1.metric("Status", drift["status"])
            d2.metric("Message-length PSI", f"{drift['feature_psi']['length']:.3f}")
            d3.metric("Sentiment PSI", f"{drift['feature_psi']['sentiment']:.3f}")
            st.caption(f"Compared with {drift['baseline_source']} using {drift['sample_size']} recent tickets.")
            st.info(drift["recommendation"])
    else:
        st.info("Analytics will appear after the first ticket is added.")

with admin_tab:
    st.subheader("Setup and system status")
    health1, health2, health3 = st.columns(3)
    health1.metric("AI model", "Ready" if model_ready else "Fallback mode")
    health2.metric("Database", "Connected" if service.store.health() else "Unavailable")
    health3.metric("Stored tickets", service.store.metrics()["total"])

    st.markdown("#### Optional security settings")
    config_df = pd.DataFrame([
        {"Setting": "APP_PASSWORD", "Purpose": "Require a password before opening the dashboard",
         "Current state": "Enabled" if os.getenv("APP_PASSWORD") else "Not set"},
        {"Setting": "API_KEY", "Purpose": "Protect API operations with the X-API-Key header",
         "Current state": "Enabled" if os.getenv("API_KEY") else "Not set"},
        {"Setting": "DATABASE_URL", "Purpose": "Use PostgreSQL for durable cloud storage",
         "Current state": "PostgreSQL" if os.getenv("DATABASE_URL", "").startswith("postgres") else "Local SQLite"},
    ])
    st.dataframe(config_df, width="stretch", hide_index=True)

    with st.expander("Run the integration API"):
        st.code("uvicorn api:app --host 0.0.0.0 --port 8000", language="bash")
        st.markdown("Open `http://127.0.0.1:8000/docs` for interactive API documentation.")
    with st.expander("Privacy and audit behavior"):
        st.markdown("""
        - Common email, phone, payment-card, and customer-reference patterns are masked before storage.
        - The application does not intentionally retain the unmasked customer message.
        - Ticket creation and every agent update are recorded in the audit history.
        - Safety-related phrases can override the model priority and require immediate human review.
        """)
    rai_path = CURRENT_DIR / "Responsible_AI.md"
    if rai_path.exists():
        with st.expander("Responsible AI policy"):
            st.markdown(rai_path.read_text(encoding="utf-8"))
