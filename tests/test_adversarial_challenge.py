import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models import SyncFAQRequest
from app.retrieval_engine import parse_typesense_hits


class TestAdversarialLexiconAndScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # =========================================================================
    # CHALLENGE 1: Numeric tokens and serialization boundary
    # =========================================================================
    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_accepts_numeric_tokens_without_http_422(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """
        Adversarial regression test: Numeric tokens in payload (integers or strings)
        are coerced to string primitives by Pydantic validator, preventing HTTP 422
        and indexing properly into Typesense.
        """
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_covid", "status": "upserted"}

        payload = {
            "id": "faq_covid",
            "workspace_id": 1,
            "question": "What is COVID-19 protocol?",
            "answer": "Room 101 requires masks.",
            "priority": 100,
            "is_active": True,
            "lexicon_terms": ["covid", 19, "protocol", "room", 101, "requires", "masks"],
        }
        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "synced")
        passed_doc = mock_upsert.call_args[0][1]
        expected_terms = ["covid", "19", "protocol", "room", "101", "requires", "masks"]
        self.assertEqual(passed_doc["lexicon_terms"], expected_terms)
        for term in passed_doc["lexicon_terms"]:
            self.assertIsInstance(term, str)

    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_handles_large_and_complex_unicode_when_all_strings(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """
        Adversarial test: Extreme inputs containing complex Bengali Unicode (conjuncts,
        halants, nuktas) and emojis succeed if all tokens are string instances.
        """
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_complex_unicode", "status": "upserted"}

        bengali_complex_terms = [
            "যুক্তবর্ণ", "ক্ষ", "জ্ঞ", "হ্ণ", "স্ট্র", "য়", "ড়", "ঢ়", "বাংলা"
        ]
        payload = {
            "id": "faq_complex_unicode",
            "workspace_id": 10,
            "question": "বাংলা ইউনিকোড পরীক্ষা?",
            "answer": "যুক্তবর্ণ এবং জটিল বর্ণমালা হ্যান্ডলিং।",
            "priority": 100,
            "is_active": True,
            "lexicon_terms": bengali_complex_terms,
        }
        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "synced")

    # =========================================================================
    # CHALLENGE 2: Single-token vs Multi-token noisy query scoring
    # =========================================================================
    def test_single_token_exact_match_passes_threshold_even_with_zero_semantic_score(self):
        """
        Adversarial test: For single-token queries matching an exact lexical term,
        keyword_ratio == 1.0 >= 0.8 and tokens_matched == 1 >= 1.
        Even if dense vector distance is 1.0 (semantic_score == 0.0), final_score must be:
        max(0.0, 0.50 * 0.0 + 0.50 * 1.0) = 0.50 >= 0.40 (survives retrieval cutoff).
        """
        hit = {
            "document": {
                "id": "doc_single_fee",
                "question": "ফি কত?",
                "answer": "ভর্তি ফি ১০০০ টাকা।",
                "priority": 100,
            },
            "vector_distance": 1.0,  # semantic_score = 0.0
            "text_match_info": {
                "tokens_matched": 1,
                "num_tokens_dropped": 0,  # 1/1 tokens matched -> ratio = 1.0
            },
        }
        results = parse_typesense_hits([hit])
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["match_type"], "hybrid")
        self.assertEqual(r["semantic_score"], 0.0)
        self.assertEqual(r["keyword_score"], 1.0)
        self.assertEqual(r["score"], 0.50)
        self.assertGreaterEqual(r["score"], 0.40)

    def test_noisy_multi_token_query_is_not_inflated_by_single_token_match(self):
        """
        Adversarial test: A noisy 4-token query (e.g. 'what is the fee') where only 1 token
        matches (ratio = 1/4 = 0.25 < 0.8).
        - If semantic_score <= 0.30: falls through to vector match and is NOT inflated.
        - If semantic_score = 0.35: final_score = 0.80*0.35 + 0.20*0.25 = 0.33 < 0.40 (dropped).
        - If semantic_score = 0.45: final_score = 0.80*0.45 + 0.20*0.25 = 0.41 (weak keyword dampens score).
        """
        # Case A: Low semantic support (0.20)
        hit_low_sem = {
            "document": {"id": "doc_noisy_1", "question": "q", "answer": "a"},
            "vector_distance": 0.80,  # semantic_score = 0.20 <= 0.30
            "text_match_info": {"tokens_matched": 1, "num_tokens_dropped": 3},  # ratio = 0.25
        }
        res_a = parse_typesense_hits([hit_low_sem])[0]
        self.assertEqual(res_a["match_type"], "vector")
        self.assertEqual(res_a["score"], 0.20)  # Un-inflated, below 0.40

        # Case B: Marginal semantic support (0.35)
        hit_mid_sem = {
            "document": {"id": "doc_noisy_2", "question": "q", "answer": "a"},
            "vector_distance": 0.65,  # semantic_score = 0.35 > 0.30
            "text_match_info": {"tokens_matched": 1, "num_tokens_dropped": 3},  # ratio = 0.25
        }
        res_b = parse_typesense_hits([hit_mid_sem])[0]
        self.assertEqual(res_b["match_type"], "hybrid")
        self.assertEqual(res_b["score"], 0.33)  # 0.80*0.35 + 0.20*0.25 = 0.33 < 0.40 (dropped)

        # Case C: Decent semantic support (0.45)
        hit_good_sem = {
            "document": {"id": "doc_noisy_3", "question": "q", "answer": "a"},
            "vector_distance": 0.55,  # semantic_score = 0.45
            "text_match_info": {"tokens_matched": 1, "num_tokens_dropped": 3},  # ratio = 0.25
        }
        res_c = parse_typesense_hits([hit_good_sem])[0]
        self.assertEqual(res_c["match_type"], "hybrid")
        # Notice 0.41 is LESS than pure semantic 0.45; the low keyword coverage dragged it down!
        self.assertEqual(res_c["score"], 0.41)
        self.assertLess(res_c["score"], res_c["semantic_score"])

    def test_two_token_query_half_match_ratio_0_5_does_not_trigger_first_branch(self):
        """
        Adversarial test: 2-token query where 1 matches (ratio = 1/2 = 0.50 < 0.80).
        Must NOT trigger keyword_ratio >= 0.80 branch.
        """
        hit_half = {
            "document": {"id": "doc_half", "question": "q", "answer": "a"},
            "vector_distance": 0.75,  # semantic_score = 0.25
            "text_match_info": {"tokens_matched": 1, "num_tokens_dropped": 1},  # ratio = 0.50
        }
        res = parse_typesense_hits([hit_half])[0]
        self.assertEqual(res["match_type"], "vector")
        self.assertEqual(res["score"], 0.25)

    # =========================================================================
    # CHALLENGE 3: Tenant isolation during search and deletion
    # =========================================================================
    def test_search_rejects_missing_workspace_id_fails_closed(self):
        """Search without workspace_id must immediately fail closed with HTTP 400."""
        response = self.client.post("/api/v1/search", json={"query": "test query", "top_k": 3})
        self.assertEqual(response.status_code, 400)
        self.assertIn("workspace_id is required", response.json()["detail"])

    @patch("app.main.get_typesense_client")
    @patch("app.main.delete_faq_document")
    def test_delete_faq_forwards_workspace_id_to_delete_document(self, mock_delete, mock_get_client):
        """
        Adversarial regression test on deletion tenant boundary:
        DELETE /api/v1/faqs/{faq_id}?workspace_id=999
        Verifies that delete_faq forwards workspace_id to delete_faq_document
        so Typesense can enforce atomic filter_by tenant scoping.
        """
        mock_get_client.return_value = MagicMock()
        mock_delete.return_value = {"id": "faq_ws_victim", "deleted": True}

        response = self.client.delete("/api/v1/faqs/faq_ws_victim?workspace_id=999")
        self.assertEqual(response.status_code, 200)
        mock_delete.assert_called_with(mock_get_client.return_value, "faq_ws_victim", workspace_id=999)

    @patch("app.main.get_typesense_client")
    @patch("app.main.delete_faq_document")
    def test_delete_faq_cross_tenant_simulation(self, mock_delete, mock_get_client):
        """
        Adversarial test: Tenant 2 attempts to delete Tenant 1's FAQ ID.
        Verifies that delete_faq scopes strictly to Tenant 2's workspace_id,
        ensuring Typesense filter never matches or mutates Tenant 1's records.
        """
        mock_get_client.return_value = MagicMock()
        mock_delete.return_value = {"id": "tenant1_secret_faq", "num_deleted": 0}

        # Tenant 2 calls delete on tenant1's FAQ
        response = self.client.delete("/api/v1/faqs/tenant1_secret_faq?workspace_id=2")
        self.assertEqual(response.status_code, 200)
        # Deletion query is locked to workspace_id=2
        mock_delete.assert_called_with(mock_get_client.return_value, "tenant1_secret_faq", workspace_id=2)

    def test_delete_faq_document_prevents_cross_tenant_deletion(self):
        """
        Adversarial test: Calling delete_faq_document with Tenant 2's workspace_id
        strictly sends filter_by with workspace_id:=2, preventing cross-tenant document deletion.
        """
        from app.typesense_engine import delete_faq_document
        from app import config

        mock_client = MagicMock()
        mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.return_value = {"num_deleted": 0}

        res = delete_faq_document(mock_client, "tenant1_faq_id", workspace_id=2)
        mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.assert_called_once_with({
            "filter_by": "id:=tenant1_faq_id && workspace_id:=2"
        })
        self.assertEqual(res, {"num_deleted": 0})


if __name__ == "__main__":
    unittest.main()
