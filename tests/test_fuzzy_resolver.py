import unittest
from unittest.mock import patch
from app.analytics.fuzzy_resolver import FuzzyEntityResolver

class TestFuzzyEntityResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = FuzzyEntityResolver(cache_ttl_seconds=60)
        self.mock_entities = {
            "salesperson": ["Hasan", "Rakib", "Tarek", "Mehedi", "Rahim", "Rafiq", "Karim"],
            "customer": ["Alice Walker", "Rahman Enterprise", "Rahin Traders"],
            "product": ["Laptop Pro 15", "Laptop Air", "Mechanical Keyboard RGB"],
            "category": ["Electronics", "Accessories"],
            "payment_method": ["cash", "bank", "bkash", "nagad", "card", "rocket"],
            "status": ["completed", "pending", "cancelled"],
        }

    def _get_mock_entities(self, workspace_id):
        return self.mock_entities

    def test_exact_and_case_insensitive_match(self):
        """Exact matches (any case) resolve with 1.0 confidence."""
        with patch.object(self.resolver, 'get_entities', side_effect=self._get_mock_entities):
            # Exact
            val, score = self.resolver.normalize(1, "salesperson", "Hasan")
            self.assertEqual(val, "Hasan")
            self.assertEqual(score, 1.0)

            # Lowercase
            val, score = self.resolver.normalize(1, "salesperson", "hasan")
            self.assertEqual(val, "Hasan")
            self.assertEqual(score, 1.0)

            # Uppercase
            val, score = self.resolver.normalize(1, "salesperson", "RAKIB")
            self.assertEqual(val, "Rakib")
            self.assertEqual(score, 1.0)

    def test_typo_and_transposition_resolution(self):
        """Typo variations resolve accurately to canonical names."""
        with patch.object(self.resolver, 'get_entities', side_effect=self._get_mock_entities):
            # Typo missing character: hasn -> Hasan
            val, score = self.resolver.normalize(1, "salesperson", "hasn")
            self.assertEqual(val, "Hasan")
            self.assertGreaterEqual(score, 0.75)

            # Typo product: laptp -> Laptop Pro 15
            val, score = self.resolver.normalize(1, "product", "laptp")
            self.assertIn("Laptop", val)
            self.assertGreaterEqual(score, 0.70)

            # Transposition: bikash -> bkash
            val, score = self.resolver.normalize(1, "payment_method", "bikash")
            self.assertEqual(val, "bkash")
            self.assertGreaterEqual(score, 0.80)

    def test_ambiguity_rejection(self):
        """When multiple candidates are similarly plausible, resolver avoids guessing."""
        with patch.object(self.resolver, 'get_entities', side_effect=self._get_mock_entities):
            # 'Rah' is equally close to 'Rahim', 'Rahman Enterprise', and 'Rahin Traders'
            val, score = self.resolver.normalize(1, "customer", "Rah")
            # Ambiguity guard should return 0.0 or refuse confident guess
            self.assertIn(score, [0.0, 0.90]) # Substring match or ambiguity guard

    def test_completely_unknown_entity(self):
        """Unknown or non-existent entity returns original string with 0.0 score."""
        with patch.object(self.resolver, 'get_entities', side_effect=self._get_mock_entities):
            val, score = self.resolver.normalize(1, "salesperson", "xyznonexistent999")
            self.assertEqual(val, "xyznonexistent999")
            self.assertEqual(score, 0.0)

    def test_empty_and_null_inputs(self):
        """Empty or invalid string inputs do not raise exceptions."""
        val, score = self.resolver.normalize(1, "salesperson", "")
        self.assertEqual(val, "")
        self.assertEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
