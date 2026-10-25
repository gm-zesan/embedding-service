import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.embedding import embed, embed_batch, get_model, get_embedding_dimension


class TestDenseEmbeddingRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check_returns_healthy_dense_embedding(self):
        """Health check must show status ok and embedding_available true."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("embedding_available"))
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("embedding_dimension"), 768)
        self.assertEqual(data.get("model_name"), "paraphrase-multilingual-mpnet-base-v2")
        self.assertTrue(data.get("typesense_healthy"))

    def test_direct_dense_vector_generation(self):
        """Embedding generation must return 768-dimensional normalized vectors."""
        vec = embed("ডেলিভারি চার্জ কত?")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 768)
        self.assertIsInstance(vec[0], float)

        batch = embed_batch(["What is return policy?", "অর্ডার ট্র্যাক করব কিভাবে?"])
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), 768)
        self.assertEqual(len(batch[1]), 768)

    def test_semantic_retrieval_bengali_banglish_synonyms(self):
        """Knowledge retrieval must return relevant hits using dense vector embeddings."""
        test_queries = [
            ("ডেলিভারি চার্জ কত?", "Delivery fee / charges in Bengali"),
            ("delivery charge koto ashbe?", "Delivery fee in Banglish"),
            ("পাঠাতে কত টাকা নিবে?", "Colloquial synonym without explicit 'charge' keyword"),
            ("পণ্য নষ্ট হলে টাকা ফেরত দেবেন?", "Refund / return policy rephrased"),
        ]

        for query, desc in test_queries:
            with self.subTest(query=query, description=desc):
                response = self.client.post(
                    "/api/v1/search",
                    json={"query": query, "workspace_id": 1, "top_k": 3},
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("results", data)
                self.assertGreater(len(data["results"]), 0, f"Query '{query}' returned no results")
                top_hit = data["results"][0]
                self.assertIn("score", top_hit)
                self.assertGreater(top_hit["score"], 0.0)
                # Ensure match has non-zero semantic score or hybrid match
                telemetry = data.get("telemetry", {})
                self.assertIn("embedding_ms", telemetry)
                self.assertGreater(telemetry["embedding_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
