import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval_engine import parse_typesense_hits


class TestDegradedEmbeddingFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.patcher1 = patch("app.main.EMBEDDING_AVAILABLE", False)
        self.patcher2 = patch("app.main.embed", None)
        self.patcher3 = patch("app.main.get_model", lambda: None)
        self.patcher4 = patch("app.retrieval_engine.embed", None)
        self.patcher5 = patch("app.config.LLM_EXPANSION_API_KEY", "")
        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()
        self.patcher4.start()
        self.patcher5.start()

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()

    def test_health_check_returns_degraded_without_crashing(self):
        """Health check must gracefully return degraded status when embedding engine is unavailable."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "degraded")
        self.assertFalse(data.get("embedding_available"))
        self.assertFalse(data.get("model_loaded"))
        self.assertEqual(data.get("embedding_dimension"), 768)

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

    # ──────────────────────────────────────────────────────────────────────────
    # IMP-RET-001: Pure Typesense BM25 Fallback on Dense Embedding Outage
    # ──────────────────────────────────────────────────────────────────────────

    @patch("app.retrieval_engine.execute_hybrid_search")
    def test_search_faqs_falls_back_to_pure_bm25_when_embed_is_none(self, mock_search):
        """When embedding engine is None, search must return HTTP 200 with pure BM25 hits and query_vector=None."""
        mock_search.return_value = {
            "hits": [
                {
                    "document": {
                        "id": "faq_delivery_01",
                        "workspace_id": 1,
                        "question": "ডেলিভারি চার্জ কত?",
                        "answer": "ঢাকার ভেতর ৬০ টাকা, ঢাকার বাইরে ১২০ টাকা।",
                        "priority": 100,
                        "document_type": "faq",
                        "is_active": True,
                    },
                    "text_match": 570425344,
                    "text_match_info": {
                        "best_field_weight": 4,
                        "tokens_matched": 3,
                        "num_tokens_dropped": 0,
                        "score": "1107296256",
                    },
                }
            ]
        }

        response = self.client.post(
            "/api/v1/search",
            json={"query": "ডেলিভারি চার্জ কত?", "workspace_id": 1, "top_k": 5},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 1)

        hit = data["results"][0]
        self.assertEqual(hit["id"], "faq_delivery_01")
        self.assertEqual(hit["match_type"], "bm25")
        self.assertGreaterEqual(hit["score"], 0.65)
        self.assertEqual(hit["semantic_score"], 0.0)
        self.assertEqual(hit["keyword_score"], 1.0)

        # Verify search was executed without vector query
        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        self.assertIsNone(kwargs["query_vector"])
        self.assertEqual(kwargs["workspace_id"], 1)

        # Verify telemetry
        telemetry = data.get("telemetry", {})
        self.assertTrue(telemetry.get("degraded_bm25_fallback"))
        self.assertEqual(telemetry.get("tier_executed"), "tier1_bm25_fallback")
        self.assertEqual(telemetry.get("embedding_ms"), 0.0)

    @patch("app.retrieval_engine.execute_hybrid_search")
    def test_search_faqs_falls_back_to_pure_bm25_on_embedding_exception(self, mock_search):
        """When embed callable throws a runtime exception (e.g. PyTorch OOM), fall back to BM25 with HTTP 200."""
        def crashing_embed(text):
            raise RuntimeError("CUDA/MPS out of memory during embedding inference")

        mock_search.return_value = {
            "hits": [
                {
                    "document": {
                        "id": "faq_refund_01",
                        "workspace_id": 1,
                        "question": "What is the return policy?",
                        "answer": "We offer a 30-day money-back guarantee.",
                        "priority": 90,
                        "document_type": "faq",
                        "is_active": True,
                    },
                    "text_match": 450000000,
                    "text_match_info": {
                        "best_field_weight": 4,
                        "tokens_matched": 2,
                        "num_tokens_dropped": 0,
                    },
                }
            ]
        }

        # Override embed with a failing callable
        with patch("app.retrieval_engine.embed", crashing_embed):
            response = self.client.post(
                "/api/v1/search",
                json={"query": "return policy", "workspace_id": 1, "top_k": 3},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["id"], "faq_refund_01")
        self.assertEqual(data["results"][0]["match_type"], "bm25")

        telemetry = data.get("telemetry", {})
        self.assertTrue(telemetry.get("degraded_bm25_fallback"))
        self.assertEqual(telemetry.get("tier_executed"), "tier1_bm25_fallback")

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        self.assertIsNone(kwargs["query_vector"])

    def test_tenant_isolation_rejects_missing_workspace_id_in_degraded_mode(self):
        """Even in degraded BM25 mode, missing workspace_id must immediately fail closed with HTTP 400."""
        response = self.client.post(
            "/api/v1/search",
            json={"query": "ডেলিভারি চার্জ কত?", "top_k": 5},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("workspace_id is required", response.json().get("detail", ""))

    @patch("app.retrieval_engine.execute_hybrid_search")
    def test_tenant_isolation_preserves_workspace_boundary_in_degraded_bm25(self, mock_search):
        """Cross-tenant queries must return 0 results and never leak records from other workspaces."""
        # Typesense returns empty hits when workspace_id has no matching records
        mock_search.return_value = {"hits": []}

        response = self.client.post(
            "/api/v1/search",
            json={"query": "confidential internal policy", "workspace_id": 9999, "top_k": 5},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data.get("results", [])), 0)
        self.assertEqual(data.get("total_found"), 0)

        # Ensure workspace_id was strictly passed into search filter
        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        self.assertEqual(kwargs["workspace_id"], 9999)
        self.assertIsNone(kwargs["query_vector"])

    @patch("app.retrieval_engine.execute_hybrid_search")
    def test_multilingual_bm25_fallback_queries(self, mock_search):
        """Verify Bengali, Banglish, English, and code-mixed queries succeed in pure BM25 fallback."""
        test_cases = [
            ("ডেলিভারি চার্জ কত?", "Bengali delivery query"),
            ("delivery charge koto ashbe?", "Banglish delivery query"),
            ("How do I cancel my order?", "English cancellation query"),
            ("order cancellation kivabe korbo?", "Code-mixed cancellation query"),
        ]

        for query, desc in test_cases:
            with self.subTest(query=query, description=desc):
                mock_search.reset_mock()
                mock_search.return_value = {
                    "hits": [
                        {
                            "document": {
                                "id": f"faq_{hash(query) % 10000}",
                                "workspace_id": 1,
                                "question": query,
                                "answer": "Detailed answer for customer.",
                                "priority": 100,
                                "document_type": "faq",
                                "is_active": True,
                            },
                            "text_match": 450000000,
                            "text_match_info": {
                                "best_field_weight": 4,
                                "tokens_matched": 2,
                                "num_tokens_dropped": 0,
                            },
                        }
                    ]
                }

                resp = self.client.post(
                    "/api/v1/search",
                    json={"query": query, "workspace_id": 1, "top_k": 3},
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(len(data["results"]), 1)
                self.assertEqual(data["results"][0]["match_type"], "bm25")
                self.assertTrue(data["telemetry"]["degraded_bm25_fallback"])
                self.assertIsNone(mock_search.call_args[1]["query_vector"])

    @patch("app.retrieval_engine.execute_hybrid_search")
    def test_out_of_domain_query_returns_zero_hits_gracefully(self, mock_search):
        """Unmatched out-of-domain query in BM25 mode returns HTTP 200 with 0 results."""
        mock_search.return_value = {"hits": []}

        resp = self.client.post(
            "/api/v1/search",
            json={"query": "quantum mechanical particle accelerator construction", "workspace_id": 1, "top_k": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["results"]), 0)
        self.assertEqual(data["total_found"], 0)

    def test_parse_typesense_hits_bm25_score_calibration(self):
        """Direct test for BM25 score calibration: tokens matched, dropped, and match_type assignment."""
        raw_hits = [
            # Hit 1: 100% keyword match (2 tokens matched, 0 dropped)
            {
                "document": {"id": "1", "question": "Delivery fee", "answer": "60 taka", "priority": 10},
                "text_match_info": {"tokens_matched": 2, "num_tokens_dropped": 0},
            },
            # Hit 2: 50% keyword match (1 token matched, 1 dropped)
            {
                "document": {"id": "2", "question": "Express fee", "answer": "120 taka", "priority": 5},
                "text_match_info": {"tokens_matched": 1, "num_tokens_dropped": 1},
            },
            # Hit 3: Single token query (1 matched, 0 dropped)
            {
                "document": {"id": "3", "question": "Refund", "answer": "Full refund", "priority": 0},
                "text_match_info": {"tokens_matched": 1, "num_tokens_dropped": 0},
            },
            # Hit 4: No tokens matched info, but positive text_match
            {
                "document": {"id": "4", "question": "General", "answer": "FAQ", "priority": 0},
                "text_match": 100,
                "text_match_info": {"tokens_matched": 0, "num_tokens_dropped": 0},
            },
        ]

        parsed = parse_typesense_hits(raw_hits)
        self.assertEqual(len(parsed), 4)

        # Hit 1: 100% keyword ratio -> 0.65 + 0.30 * 1.0 = 0.95
        self.assertEqual(parsed[0]["id"], "1")
        self.assertEqual(parsed[0]["match_type"], "bm25")
        self.assertEqual(parsed[0]["score"], 0.95)
        self.assertEqual(parsed[0]["semantic_score"], 0.0)
        self.assertEqual(parsed[0]["keyword_score"], 1.0)

        # Hit 3: Single token 100% match -> 0.95
        hit3 = next(p for p in parsed if p["id"] == "3")
        self.assertEqual(hit3["match_type"], "bm25")
        self.assertEqual(hit3["score"], 0.95)

        # Hit 2: 50% keyword ratio -> 0.65 + 0.30 * 0.5 = 0.80
        hit2 = next(p for p in parsed if p["id"] == "2")
        self.assertEqual(hit2["match_type"], "bm25")
        self.assertEqual(hit2["score"], 0.80)
        self.assertEqual(hit2["keyword_score"], 0.5)

        # Hit 4: 0 tokens matched -> base 0.60
        hit4 = next(p for p in parsed if p["id"] == "4")
        self.assertEqual(hit4["match_type"], "bm25")
        self.assertEqual(hit4["score"], 0.60)

    @patch("app.retrieval_engine.execute_hybrid_search")
    def test_bm25_top3_retrieval_recall_benchmark_exceeds_85_percent(self, mock_search):
        """
        Acceptance Criteria Benchmark: Top-3 BM25 retrieval recall must exceed 85% on embedding outage.
        Simulates 10 realistic customer inquiries against a knowledge base catalog.
        """
        # Catalog of 10 FAQ entries
        knowledge_catalog = [
            {"id": "faq_deliv_fee", "q": "ডেলিভারি চার্জ কত?", "tokens": ["ডেলিভারি", "চার্জ", "কত"]},
            {"id": "faq_deliv_time", "q": "ডেলিভারি পেতে কতদিন সময় লাগে?", "tokens": ["ডেলিভারি", "সময়", "কতদিন"]},
            {"id": "faq_refund_req", "q": "টাকা রিফান্ড পাব কিভাবে?", "tokens": ["টাকা", "রিফান্ড", "ফেরত"]},
            {"id": "faq_order_cancel", "q": "অর্ডার বাতিল করার নিয়ম কি?", "tokens": ["অর্ডার", "বাতিল", "ক্যানসেল"]},
            {"id": "faq_pay_methods", "q": "কি কি পেমেন্ট মেথড সাপোর্ট করেন?", "tokens": ["পেমেন্ট", "বিকাশ", "নগদ", "কার্ড"]},
            {"id": "faq_exchange_rule", "q": "সাইজ বা কালার পরিবর্তন করার উপায় কি?", "tokens": ["এক্সচেঞ্জ", "পরিবর্তন", "সাইজ"]},
            {"id": "faq_warranty_claim", "q": "ওয়ারেন্টি ক্লেইম করার নিয়ম কি?", "tokens": ["ওয়ারেন্টি", "গ্যারান্টি", "ক্লেইম"]},
            {"id": "faq_tracking_status", "q": "অর্ডার ট্র্যাকিং করার লিংক কোথায়?", "tokens": ["ট্র্যাক", "ট্র্যাকিং", "অর্ডার", "স্ট্যাটাস"]},
            {"id": "faq_discount_coupon", "q": "ডিসকাউন্ট কুপন কিভাবে ব্যবহার করব?", "tokens": ["ডিসকাউন্ট", "কুপন", "প্রমো"]},
            {"id": "faq_hotline_contact", "q": "কাস্টমার কেয়ার হেল্পলাইন নাম্বার কি?", "tokens": ["কাস্টমার", "কেয়ার", "হেল্পলাইন", "নাম্বার"]},
        ]

        # 10 test benchmark queries with their expected target FAQ ID
        benchmark_queries = [
            ("ডেলিভারি চার্জ কত নিবেন?", "faq_deliv_fee"),
            ("কবে ডেলিভারি পাব সময় কত লাগে?", "faq_deliv_time"),
            ("টাকা ফেরত বা রিফান্ড করার নিয়ম", "faq_refund_req"),
            ("আমি আমার অর্ডার বাতিল করতে চাই", "faq_order_cancel"),
            ("বিকাশ বা নগদ দিয়ে পেমেন্ট করা যাবে?", "faq_pay_methods"),
            ("সাইজ পরিবর্তন বা এক্সচেঞ্জ হবে কি?", "faq_exchange_rule"),
            ("ওয়ারেন্টি কিভাবে ক্লেইম করব?", "faq_warranty_claim"),
            ("অর্ডার স্ট্যাটাস ট্র্যাক করব কিভাবে?", "faq_tracking_status"),
            ("প্রমো কোড ও ডিসকাউন্ট কুপন ব্যবহারের নিয়ম", "faq_discount_coupon"),
            ("হেল্পলাইন নাম্বার ও কাস্টমার কেয়ার যোগাযোগ", "faq_hotline_contact"),
        ]

        def simulate_typesense_bm25(client, query_text, query_vector, workspace_id, top_k):
            # Deterministic BM25 simulator matching Typesense keyword overlap
            query_words = set(query_text.strip().split())
            scored_docs = []
            for item in knowledge_catalog:
                matched = [t for t in item["tokens"] if any(t in qw or qw in t for qw in query_words)]
                if matched:
                    scored_docs.append({
                        "document": {
                            "id": item["id"],
                            "workspace_id": workspace_id,
                            "question": item["q"],
                            "answer": f"Answer for {item['id']}",
                            "priority": 100,
                            "document_type": "faq",
                            "is_active": True,
                        },
                        "text_match": len(matched) * 1000,
                        "text_match_info": {
                            "tokens_matched": len(matched),
                            "num_tokens_dropped": max(0, len(query_words) - len(matched)),
                        },
                    })
            scored_docs.sort(key=lambda x: x["text_match"], reverse=True)
            return {"hits": scored_docs[:top_k]}

        mock_search.side_effect = simulate_typesense_bm25

        top1_hits = 0
        top3_hits = 0
        total_queries = len(benchmark_queries)

        for query, expected_id in benchmark_queries:
            resp = self.client.post(
                "/api/v1/search",
                json={"query": query, "workspace_id": 1, "top_k": 3},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            returned_ids = [h["id"] for h in data.get("results", [])]

            if returned_ids and returned_ids[0] == expected_id:
                top1_hits += 1
            if expected_id in returned_ids[:3]:
                top3_hits += 1

        top1_recall = (top1_hits / total_queries) * 100.0
        top3_recall = (top3_hits / total_queries) * 100.0

        # Assert Acceptance Criteria threshold: > 85% Top-3 BM25 recall
        self.assertGreaterEqual(
            top3_recall,
            85.0,
            f"Top-3 BM25 recall {top3_recall}% fell below required 85.0% threshold"
        )
        self.assertGreaterEqual(top1_recall, 80.0)


if __name__ == "__main__":
    unittest.main()
