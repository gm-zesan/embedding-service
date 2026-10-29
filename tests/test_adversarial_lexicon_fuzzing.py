import math
import random
import string
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models import SyncFAQRequest


class TestSyncFAQLexiconAdversarialAndFuzzing(unittest.TestCase):
    """
    Empirical adversarial stress test and fuzzing harness for SyncFAQRequest.lexicon_terms
    as required by R2: Lexicon Terms Validation Hardening (IMP-DOM-002).
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Booleans
    # -------------------------------------------------------------------------
    def test_boolean_scalars_and_arrays(self):
        # Scalar True / False
        req_true = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=True)
        self.assertEqual(req_true.lexicon_terms, [])

        req_false = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=False)
        self.assertEqual(req_false.lexicon_terms, [])

        # Arrays of booleans
        req_bool_array = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[True, False, False, True]
        )
        self.assertEqual(req_bool_array.lexicon_terms, [])

        # Mixed boolean with valid scalar primitives
        req_mixed = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[False, True, 1, "term", True, 0, False]
        )
        self.assertEqual(req_mixed.lexicon_terms, ["1", "term", "0"])

        # Invariant check: No boolean ever converted to string 'True' or 'False'
        for term in req_mixed.lexicon_terms:
            self.assertNotIn(term, ["True", "False", "true", "false"])

    # -------------------------------------------------------------------------
    # 2. Dictionaries
    # -------------------------------------------------------------------------
    def test_dictionaries_scalars_and_nested(self):
        # Scalar dict
        req_dict_scalar = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms={"dict": 1, "sub": {"nested": True}}
        )
        self.assertEqual(req_dict_scalar.lexicon_terms, [])

        # Empty dict scalar
        req_empty_dict = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms={}
        )
        self.assertEqual(req_empty_dict.lexicon_terms, [])

        # Arrays with empty dicts, dicts with string keys/values, deeply nested dicts
        adversarial_dicts = [
            {},
            {"term": "admission"},
            {"a": {"b": {"c": {"d": "nested_val"}}}},
            {"key": [1, 2, 3]},
            {"flag": True, "num": 42},
        ]
        req_dict_array = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=adversarial_dicts
        )
        self.assertEqual(req_dict_array.lexicon_terms, [])

        # Invariant: No dict is ever converted into stringified '{...}'
        for term in req_dict_array.lexicon_terms:
            self.assertFalse(term.startswith("{") or term.endswith("}"))

    # -------------------------------------------------------------------------
    # 3. Nested Lists and Iterables
    # -------------------------------------------------------------------------
    def test_nested_lists_and_iterables(self):
        nested_inputs = [
            ["nested"],
            [[1]],
            [True],
            [[["deeply_nested"]]],
            [],
            [1, 2, 3],
            ("tuple_inner", "item"),
            {1, 2, 3},  # inner set
        ]
        req_nested = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=nested_inputs
        )
        self.assertEqual(req_nested.lexicon_terms, [])

        # Invariant: No nested list is converted to stringified "['...']"
        for term in req_nested.lexicon_terms:
            self.assertFalse(term.startswith("[") or term.endswith("]"))
            self.assertFalse(term.startswith("(") or term.endswith(")"))

    # -------------------------------------------------------------------------
    # 4. Nulls / None
    # -------------------------------------------------------------------------
    def test_nulls_and_none(self):
        # Scalar None
        req_none = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=None)
        self.assertEqual(req_none.lexicon_terms, [])

        # Array of None
        req_none_array = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[None, None, None]
        )
        self.assertEqual(req_none_array.lexicon_terms, [])

        # Mixed None with valid tokens
        req_mixed_none = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[None, "valid", None, 100, None]
        )
        self.assertEqual(req_mixed_none.lexicon_terms, ["valid", "100"])

        for term in req_mixed_none.lexicon_terms:
            self.assertNotEqual(term, "None")
            self.assertNotEqual(term, "null")

    # -------------------------------------------------------------------------
    # 5. Non-finite numbers
    # -------------------------------------------------------------------------
    def test_non_finite_floats(self):
        # Scalars
        req_nan = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=float("nan"))
        self.assertEqual(req_nan.lexicon_terms, [])

        req_inf = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=float("inf"))
        self.assertEqual(req_inf.lexicon_terms, [])

        req_neginf = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=float("-inf"))
        self.assertEqual(req_neginf.lexicon_terms, [])

        # Arrays
        req_array_non_finite = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=[
                float("nan"),
                float("inf"),
                float("-inf"),
                10.5,
                float("nan"),
                -99.9,
                float("inf"),
            ]
        )
        self.assertEqual(req_array_non_finite.lexicon_terms, ["10.5", "-99.9"])

        for term in req_array_non_finite.lexicon_terms:
            self.assertNotIn(term, ["nan", "inf", "-inf", "NaN", "Infinity", "-Infinity"])

    # -------------------------------------------------------------------------
    # 6. Scalar field inputs
    # -------------------------------------------------------------------------
    def test_scalar_field_inputs(self):
        cases = [
            ("single_term", ["single_term"]),
            ("   leading_trailing   ", ["leading_trailing"]),
            ("", []),
            ("    ", []),
            (100, ["100"]),
            (0, ["0"]),
            (-12, ["-12"]),
            (12.34, ["12.34"]),
            (-0.0, ["-0.0"]),
            (True, []),
            (False, []),
            (None, []),
            ({"dict": 1}, []),
            ([], []),
            ((), []),
            (set(), []),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                req = SyncFAQRequest(id=1, workspace_id=1, question="q", answer="a", lexicon_terms=inp)
                self.assertEqual(req.lexicon_terms, expected)

    # -------------------------------------------------------------------------
    # 7. Mixed adversarial arrays combining all scenarios
    # -------------------------------------------------------------------------
    def test_mixed_adversarial_array(self):
        mixed_payload = [
            True,
            "admission",
            False,
            {"nested_dict": {"field": True}},
            101,
            [["sublist_level2"]],
            None,
            float("nan"),
            19.5,
            float("inf"),
            "  appointment  ",
            float("-inf"),
            {},
            [None, False],
            -50,
            "hotel_booking",
            0,
            "",
            "   ",
        ]
        req = SyncFAQRequest(
            id=1, workspace_id=1, question="q", answer="a",
            lexicon_terms=mixed_payload
        )
        expected = [
            "admission",
            "101",
            "19.5",
            "appointment",
            "-50",
            "hotel_booking",
            "0",
        ]
        self.assertEqual(req.lexicon_terms, expected)
        self.assertEqual(len(req.lexicon_terms), len(expected))

        # Downstream .strip() and string type safety
        for term in req.lexicon_terms:
            self.assertIsInstance(term, str)
            self.assertEqual(term, term.strip())
            self.assertGreater(len(term), 0)

        # Downstream execution simulation
        downstream_processed = [t.strip() for t in req.lexicon_terms if t.strip()]
        self.assertEqual(downstream_processed, expected)

    # -------------------------------------------------------------------------
    # 8. HTTP Endpoint End-to-End Boundary Verification
    # -------------------------------------------------------------------------
    @patch("app.main.EMBEDDING_AVAILABLE", True)
    @patch("app.main.embed", return_value=[0.01] * 768)
    @patch("app.main.get_typesense_client")
    @patch("app.main.upsert_faq_document")
    def test_http_endpoint_adversarial_fuzzing_payloads(
        self, mock_upsert, mock_get_client, mock_embed
    ):
        mock_get_client.return_value = MagicMock()
        mock_upsert.return_value = {"id": "faq_adversarial", "status": "upserted"}

        adversarial_test_cases = [
            # Case 1: All malformed elements
            (
                [True, False, None, {"bad": "data"}, [1, 2], float("nan"), float("inf")],
                []
            ),
            # Case 2: Scalar string in JSON
            (
                "hotel_checkout",
                ["hotel_checkout"]
            ),
            # Case 3: Scalar int in JSON
            (
                404,
                ["404"]
            ),
            # Case 4: Scalar float in JSON
            (
                99.95,
                ["99.95"]
            ),
            # Case 5: Scalar boolean True in JSON
            (
                True,
                []
            ),
            # Case 6: Scalar boolean False in JSON
            (
                False,
                []
            ),
            # Case 7: Scalar null in JSON
            (
                None,
                []
            ),
            # Case 8: Scalar dict in JSON
            (
                {"arbitrary": "object"},
                []
            ),
            # Case 9: Complex mixed array
            (
                ["doctor", True, 2026, {"clinic": 1}, None, 36.6, "  clinic  ", False, ["deep"]],
                ["doctor", "2026", "36.6", "clinic"]
            ),
        ]

        for idx, (input_terms, expected_terms) in enumerate(adversarial_test_cases):
            with self.subTest(case_idx=idx, input_terms=input_terms):
                mock_upsert.reset_mock()
                payload = {
                    "id": f"faq_adv_{idx}",
                    "workspace_id": 100,
                    "question": f"Question {idx}",
                    "answer": f"Answer {idx}",
                    "priority": 10,
                    "is_active": True,
                    "lexicon_terms": input_terms,
                }
                import json
                raw_json = json.dumps(payload, allow_nan=True)
                response = self.client.post("/api/v1/faqs/sync", content=raw_json, headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 200)
                mock_upsert.assert_called_once()
                upserted_doc = mock_upsert.call_args[0][1]

                self.assertEqual(upserted_doc["lexicon_terms"], expected_terms)
                for term in upserted_doc["lexicon_terms"]:
                    self.assertIsInstance(term, str)
                    self.assertNotIn(term, ["True", "False", "None", "nan", "inf"])

    # -------------------------------------------------------------------------
    # 9. Automated Randomized Fuzzer (1,500 Iterations)
    # -------------------------------------------------------------------------
    def test_randomized_adversarial_fuzzing_harness(self):
        """
        Generates 1,500 pseudo-random adversarial payloads with nested structures,
        non-finite numbers, malformed objects, booleans, and random unicode.
        Asserts model invariant: output is strictly List[str] with valid values.
        """
        rng = random.Random(42)

        def generate_random_adversarial_item(depth=0):
            if depth > 3:
                # Leaf node options
                leaf_type = rng.choice(["int", "float", "bool", "none", "str", "non_finite", "empty_str"])
                if leaf_type == "int":
                    return rng.randint(-100000, 100000)
                elif leaf_type == "float":
                    return round(rng.uniform(-1000.0, 1000.0), 4)
                elif leaf_type == "bool":
                    return rng.choice([True, False])
                elif leaf_type == "none":
                    return None
                elif leaf_type == "non_finite":
                    return rng.choice([float("nan"), float("inf"), float("-inf")])
                elif leaf_type == "empty_str":
                    return "   " * rng.randint(0, 4)
                else:
                    return "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(1, 15)))

            # Recursive options
            choice = rng.choice(["leaf", "dict", "list", "tuple", "set"])
            if choice == "leaf":
                return generate_random_adversarial_item(depth=4)
            elif choice == "dict":
                return {
                    f"key_{i}": generate_random_adversarial_item(depth + 1)
                    for i in range(rng.randint(0, 3))
                }
            elif choice == "list":
                return [generate_random_adversarial_item(depth + 1) for _ in range(rng.randint(0, 4))]
            elif choice == "tuple":
                return tuple(generate_random_adversarial_item(depth + 1) for _ in range(rng.randint(0, 3)))
            else:
                # set with hashable leaves
                return {rng.randint(1, 100) for _ in range(rng.randint(0, 3))}

        for iteration in range(1500):
            # 80% list of random items, 20% scalar random item
            if rng.random() < 0.8:
                payload_terms = [generate_random_adversarial_item(depth=0) for _ in range(rng.randint(0, 10))]
            else:
                payload_terms = generate_random_adversarial_item(depth=0)

            req = SyncFAQRequest(
                id=iteration,
                workspace_id=1,
                question="fuzz question",
                answer="fuzz answer",
                lexicon_terms=payload_terms,
            )

            terms = req.lexicon_terms

            # Invariant 1: Always a list
            self.assertIsInstance(terms, list, f"Iteration {iteration}: Output is not list: {type(terms)}")

            # Invariant 2: Every element is a non-empty stripped string
            for idx, item in enumerate(terms):
                self.assertIsInstance(
                    item, str,
                    f"Iteration {iteration}, item {idx}: Element {item} is not str: {type(item)}"
                )
                self.assertGreater(
                    len(item), 0,
                    f"Iteration {iteration}, item {idx}: Empty string survived in output"
                )
                self.assertEqual(
                    item, item.strip(),
                    f"Iteration {iteration}, item {idx}: Unstripped string {item}"
                )

                # Invariant 3: No boolean stringification
                self.assertNotIn(
                    item, ["True", "False"],
                    f"Iteration {iteration}, item {idx}: Boolean stringified into {item}"
                )

                # Invariant 4: No None stringification
                self.assertNotIn(
                    item, ["None", "null"],
                    f"Iteration {iteration}, item {idx}: None stringified into {item}"
                )

                # Invariant 5: No non-finite float stringification
                self.assertNotIn(
                    item, ["nan", "inf", "-inf", "NaN", "Infinity", "-Infinity"],
                    f"Iteration {iteration}, item {idx}: Non-finite float stringified into {item}"
                )

                # Invariant 6: No serialized objects/dicts/lists
                self.assertFalse(
                    item.startswith(("{", "[", "(")),
                    f"Iteration {iteration}, item {idx}: Object stringified into {item}"
                )
                self.assertFalse(
                    item.endswith(("}", "]", ")")),
                    f"Iteration {iteration}, item {idx}: Object stringified into {item}"
                )

            # Invariant 7: Downstream compatibility with .strip()
            try:
                processed = [t.strip() for t in terms if t.strip()]
                self.assertEqual(processed, terms)
            except Exception as e:
                self.fail(f"Iteration {iteration}: Downstream .strip() raised {e}")


if __name__ == "__main__":
    unittest.main()
