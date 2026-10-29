"""
Adversarial Empirical Stress Test Suite for R1: Tenant-Safe FAQ Deletion
Author: Empirical Challenger (challenger_boundary_1)
Target: python-ai-service
Scope: DELETE /api/v1/faqs/{faq_id} endpoint and delete_faq_document engine boundary
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.typesense_engine import delete_faq_document
from app import config


class TestDeleteFaqEndpointExtremeParams(unittest.TestCase):
    """
    Stress-testing DELETE /api/v1/faqs/{faq_id} against malformed and boundary inputs.
    FastAPI Query parameter `workspace_id: int = Query(..., gt=0)` must fail closed
    with HTTP 422 Unprocessable Entity for all invalid or missing inputs.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_omitted_workspace_id_fails_closed_with_422(self):
        """Omitted workspace_id must immediately return HTTP 422."""
        response = self.client.delete("/api/v1/faqs/faq_test_1")
        self.assertEqual(response.status_code, 422)
        error_types = [err["type"] for err in response.json()["detail"]]
        self.assertIn("missing", error_types)

    def test_extreme_and_malformed_workspace_ids_fail_closed_with_422(self):
        """
        Adversarial matrix of invalid, malformed, non-positive, and non-integer inputs.
        All must fail closed with HTTP 422 Unprocessable Entity.
        """
        malformed_queries = [
            # Non-positive integers
            ("0", "workspace_id=0 (zero)"),
            ("-1", "workspace_id=-1 (negative one)"),
            ("-999999", "workspace_id=-999999 (large negative int)"),
            ("-0", "workspace_id=-0 (negative zero)"),
            # Non-numeric / Special float representations
            ("NaN", "workspace_id=NaN"),
            ("Infinity", "workspace_id=Infinity"),
            ("-Infinity", "workspace_id=-Infinity"),
            # Booleans (string representation in query param)
            ("true", "workspace_id=true"),
            ("false", "workspace_id=false"),
            ("True", "workspace_id=True"),
            ("False", "workspace_id=False"),
            ("TRUE", "workspace_id=TRUE"),
            ("FALSE", "workspace_id=FALSE"),
            # Strings / empty / floating point strings
            ("string", "workspace_id='string'"),
            ("", "workspace_id='' (empty string)"),
            ("1.5", "workspace_id='1.5' (float string)"),
            ("0.0", "workspace_id='0.0' (float zero)"),
            ("   ", "workspace_id='   ' (whitespace)"),
            # Null representations
            ("null", "workspace_id=null"),
            ("None", "workspace_id=None"),
            ("undefined", "workspace_id=undefined"),
            # Complex strings
            ("{}", "workspace_id={}"),
            ("[]", "workspace_id=[]"),
            ("1e5", "workspace_id=1e5 (scientific notation)"),
        ]

        for q_val, desc in malformed_queries:
            with self.subTest(msg=desc, val=q_val):
                response = self.client.delete(f"/api/v1/faqs/faq_stress_test?workspace_id={q_val}")
                self.assertEqual(
                    response.status_code,
                    422,
                    f"Expected 422 for {desc}, got {response.status_code} with body: {response.text}",
                )

    @patch("app.main.get_typesense_client")
    @patch("app.main.delete_faq_document")
    def test_valid_positive_workspace_id_succeeds(self, mock_delete, mock_get_client):
        """Valid positive integer workspace_id succeeds and forwards to delete_faq_document."""
        mock_get_client.return_value = MagicMock()
        mock_delete.return_value = {"id": "faq_valid", "num_deleted": 1}

        response = self.client.delete("/api/v1/faqs/faq_valid?workspace_id=42")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "deleted", "id": "faq_valid"})
        mock_delete.assert_called_once_with(mock_get_client.return_value, "faq_valid", workspace_id=42)


class TestDeleteFaqDocumentFunctionBoundary(unittest.TestCase):
    """
    Stress-testing delete_faq_document in app/typesense_engine.py directly.
    Must strictly enforce:
    1. workspace_id must be a positive int (not bool, not None, not <= 0, not non-int).
    2. doc_id must be a non-empty string.
    3. Typesense delete must strictly use filter_by: id:=... && workspace_id:=...
    4. Never call unscoped document delete.
    """

    def setUp(self):
        self.mock_client = MagicMock()

    def test_invalid_workspace_id_fails_closed_with_value_error(self):
        """Direct invocation with invalid workspace_id values must raise ValueError."""
        invalid_ws_values = [
            (None, "None"),
            (False, "False (boolean)"),
            (True, "True (boolean)"),
            (0, "0 (zero)"),
            (-10, "-10 (negative integer)"),
            (-999999, "-999999 (large negative integer)"),
            ("10", "'10' (string integer)"),
            ("workspace_1", "'workspace_1' (string)"),
            (1.5, "1.5 (float)"),
            (0.0, "0.0 (float zero)"),
            (-0.0, "-0.0 (negative float zero)"),
            (float("nan"), "float('nan')"),
            (float("inf"), "float('inf')"),
            ([], "empty list"),
            ({}, "empty dict"),
            (object(), "generic object"),
        ]

        for ws, desc in invalid_ws_values:
            with self.subTest(msg=desc, val=ws):
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {desc}") as ctx:
                    delete_faq_document(self.mock_client, "valid_faq_id", workspace_id=ws)
                self.assertIn("workspace_id must be a positive integer", str(ctx.exception))

        # Invariant: delete was never called on the client
        self.mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.assert_not_called()
        self.mock_client.collections[config.TYPESENSE_COLLECTION].documents.__getitem__.assert_not_called()

    def test_invalid_doc_id_fails_closed_with_value_error(self):
        """Direct invocation with invalid/empty doc_id values must raise ValueError."""
        invalid_doc_ids = [
            ("", "empty string ''"),
            ("   ", "whitespace only '   '"),
            ("\t\n", "whitespace tabs/newlines"),
            (None, "None"),
        ]

        for doc_id, desc in invalid_doc_ids:
            with self.subTest(msg=desc, val=doc_id):
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {desc}") as ctx:
                    delete_faq_document(self.mock_client, doc_id, workspace_id=1)
                self.assertIn("doc_id must be a non-empty string", str(ctx.exception))

        # Invariant: delete was never called on the client
        self.mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.assert_not_called()

    def test_delete_filter_strictly_scopes_and_strips_whitespace(self):
        """Valid invocation constructs strict filter_by and trims doc_id whitespace."""
        delete_faq_document(self.mock_client, "   faq_with_spaces   ", workspace_id=99)
        self.mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.assert_called_once_with({
            "filter_by": "id:=faq_with_spaces && workspace_id:=99"
        })
        # Unscoped index lookup must never happen
        self.mock_client.collections[config.TYPESENSE_COLLECTION].documents.__getitem__.assert_not_called()


class TestMultiTenantCollectionIsolationOracle(unittest.TestCase):
    """
    Empirical Oracle Test simulating a multi-tenant Typesense collection.
    Verifies that delete_faq_document cannot delete records across workspaces,
    even under cross-tenant attacks or doc ID collisions.
    """

    def setUp(self):
        # Simulated database/index containing documents for multiple workspaces
        self.collection_store = [
            {"id": "doc_t1_secret", "workspace_id": 1, "question": "Tenant 1 Confidential Financials"},
            {"id": "doc_t2_secret", "workspace_id": 2, "question": "Tenant 2 Patient Records"},
            {"id": "doc_collision", "workspace_id": 1, "question": "Tenant 1 Colliding Document"},
            {"id": "doc_collision", "workspace_id": 2, "question": "Tenant 2 Colliding Document"},
            {"id": "doc_t3_standard", "workspace_id": 3, "question": "Tenant 3 Normal FAQ"},
        ]

        def simulate_typesense_delete(delete_kwargs):
            filter_by = delete_kwargs.get("filter_by", "")
            clauses = [c.strip() for c in filter_by.split("&&")]
            filter_dict = {}
            for c in clauses:
                k, v = c.split(":=")
                filter_dict[k.strip()] = int(v.strip()) if k.strip() == "workspace_id" else v.strip()

            initial_count = len(self.collection_store)
            surviving = []
            num_deleted = 0
            for doc in self.collection_store:
                if all(doc.get(k) == v for k, v in filter_dict.items()):
                    num_deleted += 1
                else:
                    surviving.append(doc)
            self.collection_store = surviving
            return {"num_deleted": num_deleted}

        self.mock_client = MagicMock()
        self.mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.side_effect = simulate_typesense_delete

    def test_cross_tenant_deletion_attack_fails_to_delete_target(self):
        """
        Adversarial scenario: Tenant 2 attempts to delete Tenant 1's secret document.
        Result: 0 deleted, Tenant 1 document remains untouched in collection store.
        """
        result = delete_faq_document(self.mock_client, "doc_t1_secret", workspace_id=2)
        self.assertEqual(result["num_deleted"], 0)
        tenant_1_docs = [d for d in self.collection_store if d["id"] == "doc_t1_secret"]
        self.assertEqual(len(tenant_1_docs), 1)
        self.assertEqual(tenant_1_docs[0]["workspace_id"], 1)

    def test_colliding_doc_id_deletes_only_target_workspace_record(self):
        """
        Adversarial scenario: Doc ID 'doc_collision' exists in both Workspace 1 and Workspace 2.
        Tenant 1 deletes 'doc_collision' with workspace_id=1.
        Result: Exactly 1 record deleted. Workspace 1 record removed; Workspace 2 record preserved.
        """
        result = delete_faq_document(self.mock_client, "doc_collision", workspace_id=1)
        self.assertEqual(result["num_deleted"], 1)

        # Workspace 1 collision record must be gone
        self.assertFalse(any(d["id"] == "doc_collision" and d["workspace_id"] == 1 for d in self.collection_store))
        # Workspace 2 collision record MUST remain untouched
        tenant_2_colliding = [d for d in self.collection_store if d["id"] == "doc_collision" and d["workspace_id"] == 2]
        self.assertEqual(len(tenant_2_colliding), 1)

    def test_authorized_owner_deletion_succeeds(self):
        """
        Happy path: Workspace 2 deletes its own doc_t2_secret.
        Result: Exactly 1 record deleted, other workspaces' documents untouched.
        """
        result = delete_faq_document(self.mock_client, "doc_t2_secret", workspace_id=2)
        self.assertEqual(result["num_deleted"], 1)
        self.assertFalse(any(d["id"] == "doc_t2_secret" for d in self.collection_store))
        # Ensure remaining docs in store belong to workspaces 1 and 3
        self.assertTrue(all(d["workspace_id"] in (1, 2, 3) for d in self.collection_store))
        self.assertEqual(len(self.collection_store), 4)


if __name__ == "__main__":
    unittest.main()
