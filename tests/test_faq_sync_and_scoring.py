import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models import SyncFAQRequest
from app.retrieval_engine import parse_typesense_hits
from app.typesense_engine import delete_faq_document


class TestFAQSyncAndScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_accepts_lexicon_terms_and_passes_to_upsert(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """Verify SyncFAQRequest accepts lexicon_terms and forwards them to Typesense upsert."""
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_cross_1", "status": "upserted"}

        payload = {
            "id": "faq_cross_1",
            "workspace_id": 42,
            "question": "Can I extend my stay?",
            "answer": "Hotel booking extension is allowed based on room availability.",
            "priority": 100,
            "is_active": True,
            "document_type": "faq",
            "lexicon_terms": ["extend", "hotel", "stay", "booking", "reservation"],
        }

        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "synced")
        self.assertEqual(data["id"], "faq_cross_1")
        self.assertEqual(data["workspace_id"], 42)

        mock_upsert.assert_called_once()
        passed_doc = mock_upsert.call_args[0][1]
        self.assertEqual(passed_doc["id"], "faq_cross_1")
        self.assertEqual(passed_doc["workspace_id"], 42)
        self.assertEqual(passed_doc["question"], "Can I extend my stay?")
        self.assertEqual(passed_doc["answer"], "Hotel booking extension is allowed based on room availability.")
        self.assertEqual(passed_doc["document_type"], "faq")
        self.assertEqual(passed_doc["priority"], 100)
        self.assertTrue(passed_doc["is_active"])
        self.assertEqual(
            passed_doc["lexicon_terms"],
            ["extend", "hotel", "stay", "booking", "reservation"],
        )
        self.assertEqual(len(passed_doc["embedding"]), 768)

    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_defaults_to_empty_lexicon_terms_when_omitted(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """Verify SyncFAQRequest defaults to empty lexicon_terms when omitted."""
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_no_terms", "status": "upserted"}

        payload = {
            "id": "faq_no_terms",
            "workspace_id": 1,
            "question": "General FAQ",
            "answer": "General answer",
            "priority": 50,
            "is_active": True,
        }

        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)

        mock_upsert.assert_called_once()
        passed_doc = mock_upsert.call_args[0][1]
        self.assertEqual(passed_doc["lexicon_terms"], [])

    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_strips_whitespace_and_filters_empty_terms(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """Verify whitespace is stripped and empty strings are filtered out from lexicon_terms."""
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_whitespace", "status": "upserted"}

        payload = {
            "id": "faq_whitespace",
            "workspace_id": 1,
            "question": "Test question",
            "answer": "Test answer",
            "priority": 10,
            "is_active": True,
            "lexicon_terms": ["  clinic  ", "   ", "doctor", "", "appointment  "],
        }

        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)

        mock_upsert.assert_called_once()
        passed_doc = mock_upsert.call_args[0][1]
        self.assertEqual(passed_doc["lexicon_terms"], ["clinic", "doctor", "appointment"])

    def test_parse_typesense_hits_single_token_exact_match_hybrid_scoring(self):
        """
        Verify that exact single-token keyword matches (keyword_ratio >= 0.8 and tokens_matched >= 1)
        receive hybrid score confirmation and pass the 0.40 threshold even with weak dense semantic score.
        """
        cross_domain_cases = [
            {"domain": "university", "token": "ফি", "distance": 0.75, "expected_score": 0.625},
            {"domain": "clinic", "token": "দেরি", "distance": 0.80, "expected_score": 0.600},
            {"domain": "hotel", "token": "extend", "distance": 0.70, "expected_score": 0.650},
            {"domain": "retail", "token": "exchange", "distance": 0.85, "expected_score": 0.575},
        ]

        for case in cross_domain_cases:
            raw_hit = {
                "document": {
                    "id": f"doc_{case['domain']}",
                    "question": f"Question for {case['domain']}",
                    "answer": f"Answer for {case['domain']}",
                    "document_type": "faq",
                    "priority": 100,
                },
                "vector_distance": case["distance"],
                "text_match_info": {
                    "tokens_matched": 1,
                    "num_tokens_dropped": 0,
                },
            }

            results = parse_typesense_hits([raw_hit])
            self.assertEqual(len(results), 1, f"Failed for domain {case['domain']}")
            res = results[0]

            semantic_score = round(1.0 - case["distance"], 4)
            expected_hybrid = round(max(semantic_score, 0.50 * semantic_score + 0.50 * 1.0), 4)

            self.assertEqual(res["match_type"], "hybrid", f"Expected hybrid match for domain {case['domain']}")
            self.assertEqual(res["keyword_score"], 1.0)
            self.assertEqual(res["semantic_score"], semantic_score)
            self.assertEqual(res["score"], expected_hybrid)
            self.assertGreaterEqual(
                res["score"], 0.40, f"Score {res['score']} should pass 0.40 threshold for {case['domain']}"
            )

    def test_parse_typesense_hits_partial_multi_token_match_remains_vector_when_low_semantic(self):
        """
        Verify that a single token matching out of a 3-token query (keyword_ratio = 1/3 = 0.33 < 0.8)
        with low semantic score (<= 0.30) remains classified as vector and does not falsely receive high confirmation.
        """
        raw_hit = {
            "document": {
                "id": "doc_partial",
                "question": "Some question",
                "answer": "Some answer",
                "document_type": "faq",
                "priority": 50,
            },
            "vector_distance": 0.80,  # semantic_score = 0.20 <= 0.30
            "text_match_info": {
                "tokens_matched": 1,
                "num_tokens_dropped": 2,  # total 3 tokens, ratio = 1/3 ~ 0.3333
            },
        }

        results = parse_typesense_hits([raw_hit])
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["match_type"], "vector")
        self.assertEqual(res["score"], 0.20)

    @patch("app.main.get_typesense_client")
    @patch("app.main.delete_faq_document")
    def test_delete_faq_requires_positive_workspace_id(
        self, mock_delete, mock_get_client
    ):
        """Verify DELETE /api/v1/faqs/{faq_id} enforces mandatory positive workspace_id query param."""
        mock_get_client.return_value = MagicMock()
        mock_delete.return_value = {"id": "faq_del_99", "deleted": True, "num_deleted": 1}

        # Valid positive workspace_id -> 200 OK
        response = self.client.delete("/api/v1/faqs/faq_del_99?workspace_id=7")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "deleted")
        self.assertEqual(data["id"], "faq_del_99")
        mock_delete.assert_called_with(mock_get_client.return_value, "faq_del_99", workspace_id=7)

        # Missing workspace_id -> 422 Unprocessable Entity
        mock_delete.reset_mock()
        resp_missing = self.client.delete("/api/v1/faqs/faq_del_99")
        self.assertEqual(resp_missing.status_code, 422)
        mock_delete.assert_not_called()

        # workspace_id = 0 -> 422 Unprocessable Entity
        resp_zero = self.client.delete("/api/v1/faqs/faq_del_99?workspace_id=0")
        self.assertEqual(resp_zero.status_code, 422)
        mock_delete.assert_not_called()

        # Negative workspace_id -> 422 Unprocessable Entity
        resp_neg = self.client.delete("/api/v1/faqs/faq_del_99?workspace_id=-5")
        self.assertEqual(resp_neg.status_code, 422)
        mock_delete.assert_not_called()

        # Non-integer workspace_id -> 422 Unprocessable Entity
        resp_str = self.client.delete("/api/v1/faqs/faq_del_99?workspace_id=invalid")
        self.assertEqual(resp_str.status_code, 422)
        mock_delete.assert_not_called()

    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_coerces_raw_integers_without_422(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """Verify SyncFAQRequest coerces raw integer items to strings and succeeds with HTTP 200."""
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_numeric_coerce", "status": "upserted"}

        payload = {
            "id": "faq_numeric_coerce",
            "workspace_id": 1,
            "question": "What is the COVID-19 protocol in Room 101?",
            "answer": "Open 24/7, fee 2500.",
            "priority": 100,
            "is_active": True,
            "lexicon_terms": ["covid", 19, "protocol", "room", 101, 24, 2500],
        }

        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)
        passed_doc = mock_upsert.call_args[0][1]
        expected_terms = ["covid", "19", "protocol", "room", "101", "24", "2500"]
        self.assertEqual(passed_doc["lexicon_terms"], expected_terms)
        for term in passed_doc["lexicon_terms"]:
            self.assertIsInstance(term, str)

    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.05] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_sync_faq_filters_malformed_objects_and_booleans_end_to_end(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        """Verify POST /api/v1/faqs/sync filters out booleans, dicts, lists, and nulls end-to-end."""
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_hardened", "status": "upserted"}

        payload = {
            "id": "faq_hardened",
            "workspace_id": 99,
            "question": "What is the policy?",
            "answer": "Policy details.",
            "priority": 50,
            "is_active": True,
            "lexicon_terms": [
                "clinic", 101, True, {"a": "b"}, None, ["sublist"], 19.5, False, "doctor"
            ],
        }

        response = self.client.post("/api/v1/faqs/sync", json=payload)
        self.assertEqual(response.status_code, 200)
        passed_doc = mock_upsert.call_args[0][1]
        self.assertEqual(passed_doc["lexicon_terms"], ["clinic", "101", "19.5", "doctor"])
        for term in passed_doc["lexicon_terms"]:
            self.assertIsInstance(term, str)

    def test_delete_faq_document_strictly_requires_scoped_filter(self):
        """Verify delete_faq_document strictly uses filter_by and rejects invalid workspace_id."""
        mock_client = MagicMock()
        from app import config

        # Scoped deletion with valid workspace_id
        delete_faq_document(mock_client, "faq_scoped_1", workspace_id=42)
        mock_client.collections[config.TYPESENSE_COLLECTION].documents.delete.assert_called_once_with({
            "filter_by": "id:=faq_scoped_1 && workspace_id:=42"
        })

        # Unscoped deletion is impossible: None, <= 0, bool, or non-int must raise ValueError
        for invalid_ws in [None, 0, -1, "42", True, False, 3.14]:
            with self.assertRaises(ValueError):
                delete_faq_document(mock_client, "faq_fail", workspace_id=invalid_ws)

        # Empty doc_id must raise ValueError
        for invalid_id in [None, "", "   "]:
            with self.assertRaises(ValueError):
                delete_faq_document(mock_client, invalid_id, workspace_id=1)

        # Assert unscoped documents[doc_id].delete() was never called
        mock_client.collections[config.TYPESENSE_COLLECTION].documents.__getitem__.assert_not_called()


class TestSyncFAQRequestLexiconHardening(unittest.TestCase):
    """Deterministic regression tests for hardened SyncFAQRequest lexicon_terms."""

    def test_valid_strings_whitespace_and_unicode(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=["  clinic  ", "doctor", "", "   ", "হাসপাতাল"]
        )
        self.assertEqual(req.lexicon_terms, ["clinic", "doctor", "হাসপাতাল"])

    def test_numeric_tokens_accepted_and_coerced(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[101, 19.5, 0, -5]
        )
        self.assertEqual(req.lexicon_terms, ["101", "19.5", "0", "-5"])

    def test_booleans_safely_filtered_out(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[True, False]
        )
        self.assertEqual(req.lexicon_terms, [])

    def test_dictionaries_safely_filtered_out(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[{"bad": "dict"}, {"nested": {"a": 1}}, {}]
        )
        self.assertEqual(req.lexicon_terms, [])

    def test_nested_lists_and_iterables_safely_filtered_out(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[["inner"], [1, 2], ("tuple", 3), {4, 5}]
        )
        self.assertEqual(req.lexicon_terms, [])

    def test_nulls_safely_filtered_out(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[None, None]
        )
        self.assertEqual(req.lexicon_terms, [])

    def test_mixed_array_filtering(self):
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[
                "covid", 19, True, {"payload": "invalid"},
                None, ["nested", 1], 24.5, False, "  protocol  ",
                float("nan"), float("inf"), 0
            ]
        )
        self.assertEqual(req.lexicon_terms, ["covid", "19", "24.5", "protocol", "0"])

    def test_scalar_and_none_field_inputs(self):
        self.assertEqual(SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms="single").lexicon_terms, ["single"])
        self.assertEqual(SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=42).lexicon_terms, ["42"])
        self.assertEqual(SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=True).lexicon_terms, [])
        self.assertEqual(SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=None).lexicon_terms, [])
        self.assertEqual(SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms={"key": 1}).lexicon_terms, [])


if __name__ == "__main__":
    unittest.main()
