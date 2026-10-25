import unittest
from fastapi.testclient import TestClient
from app.main import app


from unittest.mock import patch


class TestDegradedEmbeddingFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.patcher1 = patch("app.main.EMBEDDING_AVAILABLE", False)
        self.patcher2 = patch("app.main.embed", None)
        self.patcher3 = patch("app.main.get_model", lambda: None)
        self.patcher4 = patch("app.retrieval_engine.embed", None)
        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()
        self.patcher4.start()

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()

    def test_health_check_returns_degraded_without_crashing(self):
        """Health check must gracefully return degraded status when embedding engine is unavailable."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "degraded")
        self.assertFalse(data.get("embedding_available"))
        self.assertFalse(data.get("model_loaded"))
        self.assertEqual(data.get("embedding_dimension"), 768)

    def test_search_faqs_returns_503_when_embeddings_unavailable(self):
        """Knowledge search must return explicit HTTP 503 instead of silent false matches or 500 crashes."""
        response = self.client.post(
            "/api/v1/search",
            json={"query": "test query", "workspace_id": 1, "top_k": 5},
        )
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("Embedding engine is unavailable", data.get("detail", ""))

    def test_sync_faq_returns_503_when_embeddings_unavailable(self):
        """FAQ sync must return explicit HTTP 503 when embeddings cannot be computed."""
        response = self.client.post(
            "/api/v1/faqs/sync",
            json={
                "id": "faq_test_1",
                "workspace_id": 1,
                "question": "What is the return policy?",
                "answer": "30 days return window.",
                "priority": 100,
                "is_active": True,
            },
        )
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("Embedding service is unavailable", data.get("detail", ""))

    def test_embed_endpoints_return_503_when_embeddings_unavailable(self):
        """Single and batch embedding endpoints must return explicit HTTP 503."""
        res_single = self.client.post("/embed", json={"text": "hello world"})
        self.assertEqual(res_single.status_code, 503)

        res_batch = self.client.post("/embed-batch", json={"texts": ["hello", "world"]})
        self.assertEqual(res_batch.status_code, 503)

    def test_excel_analytics_remains_available_independently(self):
        """Excel analytics data source listing must remain fully operational when embeddings are degraded."""
        response = self.client.get("/analytics/sources/list", params={"workspace_id": 1})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertIn("sources", data)


if __name__ == "__main__":
    unittest.main()
