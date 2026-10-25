import unittest
from fastapi.testclient import TestClient
from app.main import app

DELIVERY_FAQ_ID = "01a05cb2-60df-716a-aad9-5c8425138262"
REFUND_FAQ_ID = "01a05c4f-06a1-733c-acb1-5756b7c8b714"
API_RATE_LIMIT_FAQ_ID = "019f59ec-b925-72af-b1cd-b97ecf0d5b61"


class TestRetrievalRelevanceBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Warmup model so cold-start load time does not skew test latency measurements
        from app.embedding import embed
        embed("warmup")

    def test_bangla_exact_delivery_query_matches_id_and_bypasses_expansion(self):
        """Bangla delivery query must return Delivery FAQ at rank 1 via Tier 1 in <150ms without noise."""
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "ডেলিভারি চার্জ কত?", "workspace_id": 1, "top_k": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        results = data.get("results", [])
        self.assertGreater(len(results), 0)
        top_hit = results[0]
        self.assertEqual(top_hit["id"], DELIVERY_FAQ_ID)
        self.assertEqual(top_hit["match_type"], "hybrid")
        self.assertGreaterEqual(top_hit["score"], 0.60)

        # Check telemetry: expansion must be bypassed (latency < 150ms)
        tele = data.get("telemetry", {})
        self.assertEqual(tele.get("tier_executed"), "tier1_raw_fastpath")
        self.assertFalse(tele.get("expansion_triggered"))
        self.assertLess(tele.get("total_retrieval_latency_ms", 9999), 200.0)

        # Noise pruning: API rate limit FAQ must NOT be in returned results
        returned_ids = [r["id"] for r in results]
        self.assertNotIn(API_RATE_LIMIT_FAQ_ID, returned_ids)

    def test_banglish_delivery_query_matches_delivery_faq(self):
        """Banglish query must return Delivery FAQ as top-1 hit."""
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "delivery charge koto ashbe?", "workspace_id": 1, "top_k": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        results = data.get("results", [])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], DELIVERY_FAQ_ID)

    def test_colloquial_synonym_delivery_matches_delivery_faq(self):
        """Colloquial query without 'charge' keyword must match Delivery FAQ at rank 1."""
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "পাঠাতে কত টাকা নিবে?", "workspace_id": 1, "top_k": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        results = data.get("results", [])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], DELIVERY_FAQ_ID)

    def test_rephrased_refund_query_matches_refund_faq(self):
        """Rephrased Bangla refund query must return Refund FAQ as top-1 hit."""
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "টাকা ফেরত পাওয়ার নিয়মাবলী", "workspace_id": 1, "top_k": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        results = data.get("results", [])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], REFUND_FAQ_ID)

    def test_tenant_isolation_rejects_missing_workspace_id(self):
        """Query without workspace_id must be rejected with HTTP 400 for tenant safety."""
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "ডেলিভারি চার্জ কত?", "top_k": 3},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("workspace_id is required", resp.json().get("detail", ""))

    def test_tenant_isolation_returns_zero_hits_for_unauthorized_workspace(self):
        """Query for foreign workspace_id must return 0 hits and never leak cross-tenant data."""
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "ডেলিভারি চার্জ কত?", "workspace_id": 999999, "top_k": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data.get("results", [])), 0)
        self.assertEqual(data.get("total_found"), 0)


if __name__ == "__main__":
    unittest.main()
