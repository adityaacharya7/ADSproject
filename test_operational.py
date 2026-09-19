import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import api
from ticket_store import TicketStore
from triage_service import TriageService


class DummyPipeline:
    pass


class DummyEngine:
    pipeline = DummyPipeline()

    def predict(self, text):
        return {
            "primary_emotion": "Neutral / Inquiry",
            "confidence": 0.55,
            "sentiment": {"label": "Neutral", "compound": 0.0},
            "urgency": "MEDIUM",
            "routing_recommendation": "Standard queue",
        }


class OperationalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = Path(self.tempdir.name, "tickets.db").as_posix()
        self.store = TicketStore(f"sqlite:///{path}")
        self.service = TriageService(self.store, DummyEngine())

    def tearDown(self):
        self.store.engine.dispose()
        self.tempdir.cleanup()

    def test_ticket_masks_pii_sets_sla_and_writes_audit_event(self):
        ticket = self.service.create_ticket(
            "Email me at jane@example.com or +1 (555) 123-4567 about account 98XYZZ12",
            external_id="CRM-42", actor="Ada")
        self.assertNotIn("jane@example.com", ticket["message"])
        self.assertIn("[EMAIL]", ticket["message"])
        self.assertIn("[PHONE]", ticket["message"])
        self.assertIn("[REFERENCE]", ticket["message"])
        self.assertAlmostEqual((ticket["sla_due_at"] - ticket["created_at"]).total_seconds(), 4 * 3600)
        events = self.store.events(ticket["id"])
        self.assertEqual(events[0]["event_type"], "CREATED")
        self.assertEqual(events[0]["actor"], "Ada")

    def test_safety_language_overrides_model_urgency(self):
        for message in ("There is a fraudulent transaction on my card", "My account was hacked"):
            ticket = self.service.create_ticket(message)
            self.assertEqual(ticket["urgency"], "CRITICAL")
            self.assertIn("Safety escalation", ticket["routing_recommendation"])
            self.assertAlmostEqual((ticket["sla_due_at"] - ticket["created_at"]).total_seconds(), 15 * 60)

    def test_external_ticket_delivery_is_idempotent_within_source(self):
        first = self.service.create_ticket("Where is it?", external_id="CRM-1", source="api")
        second = self.service.create_ticket("Duplicate delivery", external_id="CRM-1", source="api")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.store.metrics()["total"], 1)

    def test_agent_correction_resolution_and_metrics(self):
        ticket = self.service.create_ticket("My parcel is late")
        updated = self.store.update(ticket["id"], {
            "status": "RESOLVED", "assigned_to": "Ada",
            "corrected_emotion": "Anger / Frustration", "resolution_note": "Refunded"
        }, actor="Ada")
        self.assertIsNotNone(updated["resolved_at"])
        self.assertEqual(len(self.store.events(ticket["id"])), 2)
        metrics = self.store.metrics()
        self.assertEqual(metrics["open"], 0)
        self.assertEqual(metrics["correction_rate"], 1.0)
        self.assertEqual(metrics["sla_compliance_rate"], 1.0)

    def test_api_create_list_update_and_health(self):
        original = api.service
        api.service = self.service
        try:
            client = TestClient(api.app)
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            created = client.post("/predict", json={"text": "Where is my order?", "external_id": "A-1"})
            self.assertEqual(created.status_code, 200)
            ticket_id = created.json()["id"]
            listed = client.get("/tickets")
            self.assertEqual(len(listed.json()), 1)
            updated = client.patch(f"/tickets/{ticket_id}", json={"status": "IN_PROGRESS"},
                                   headers={"X-Actor": "Ada"})
            self.assertEqual(updated.json()["status"], "IN_PROGRESS")
        finally:
            api.service = original


if __name__ == "__main__":
    unittest.main()
