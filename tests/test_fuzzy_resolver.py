import unittest
from unittest.mock import patch, MagicMock
from app.analytics.fuzzy_resolver import FuzzyEntityResolver, bengali_to_phonetic_latin

class TestDynamicFuzzyEntityResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = FuzzyEntityResolver(cache_ttl_seconds=60)
        self.mock_entities = {
            "salesperson": ["Hasan", "Rakib", "Tarek", "Mehedi", "Salma"],
            "customer": ["Alice Walker", "Rahman Enterprise", "Rahin Traders", "Rahim", "Jamila", "Karim", "Rafiq", "Anis"],
            "product": ["Laptop Pro 15", "Mechanical Keyboard", "Wireless Headphone", "USB-C Multi-hub", "Ergonomic Office Chair"],
            "category": ["Furniture", "Electronics", "Accessories"],
            "payment_method": ["cash", "bank", "bkash", "nagad", "card", "rocket"],
            "status": ["completed", "pending", "cancelled", "processing"],
        }

    def _get_mock_candidates(self, workspace_id, field_type, search_terms):
        return self.mock_entities.get(field_type, [])

    # 1. Pure Algorithmic Transliteration Checks (Zero Hardcoding)
    def test_algorithmic_phonetic_transliteration(self):
        self.assertEqual(bengali_to_phonetic_latin("রাকিব"), "rakib")
        self.assertEqual(bengali_to_phonetic_latin("হাসান"), "hasan")
        self.assertEqual(bengali_to_phonetic_latin("তারেক"), "tarek")
        self.assertEqual(bengali_to_phonetic_latin("মেহেদী"), "mehedi")
        self.assertEqual(bengali_to_phonetic_latin("রহিম"), "rahim")
        self.assertEqual(bengali_to_phonetic_latin("করিম"), "karim")
        self.assertEqual(bengali_to_phonetic_latin("জামিলা"), "jamila")

    # 2. Bengali Names -> English DB Names
    def test_bengali_names_to_english_db(self):
        with patch.object(self.resolver, "get_candidates", side_effect=self._get_mock_candidates):
            val, score = self.resolver.normalize(1, "salesperson", "রাকিব")
            self.assertEqual(val, "Rakib")
            self.assertGreaterEqual(score, 0.90)

            val, score = self.resolver.normalize(1, "salesperson", "হাসান")
            self.assertEqual(val, "Hasan")
            self.assertGreaterEqual(score, 0.90)

            val, score = self.resolver.normalize(1, "customer", "জামিলা")
            self.assertEqual(val, "Jamila")
            self.assertGreaterEqual(score, 0.90)

    # 3. English Names -> English DB Names (Exact & Case Variations)
    def test_english_names_resolution(self):
        with patch.object(self.resolver, "get_candidates", side_effect=self._get_mock_candidates):
            val, score = self.resolver.normalize(1, "salesperson", "hasan")
            self.assertEqual(val, "Hasan")
            self.assertEqual(score, 1.0)

            val, score = self.resolver.normalize(1, "salesperson", "RAKIB")
            self.assertEqual(val, "Rakib")
            self.assertEqual(score, 1.0)

    # 4. Banglish & Typo Variants
    def test_banglish_and_typos(self):
        with patch.object(self.resolver, "get_candidates", side_effect=self._get_mock_candidates):
            val, score = self.resolver.normalize(1, "salesperson", "hasn")
            self.assertEqual(val, "Hasan")
            self.assertGreaterEqual(score, 0.75)

            val, score = self.resolver.normalize(1, "payment_method", "bikash")
            self.assertEqual(val, "bkash")
            self.assertGreaterEqual(score, 0.85)

            val, score = self.resolver.normalize(1, "product", "keybord")
            self.assertEqual(val, "Mechanical Keyboard")
            self.assertGreaterEqual(score, 0.70)

    # 5. Ambiguity Guard: Refuses guessing when margin < 0.10
    def test_margin_based_ambiguity_guard(self):
        with patch.object(self.resolver, "get_candidates", side_effect=self._get_mock_candidates):
            # "Rah" matches both "Rahman Enterprise" and "Rahin Traders" with near-identical ratio
            val, score = self.resolver.normalize(1, "customer", "Rah")
            self.assertEqual(score, 0.0)
            self.assertEqual(val, "Rah")

    # 6. Low Confidence Rejection (Unknown Entity)
    def test_low_confidence_rejection(self):
        with patch.object(self.resolver, "get_candidates", side_effect=self._get_mock_candidates):
            val, score = self.resolver.normalize(1, "salesperson", "randomnonexistentguy999")
            self.assertEqual(val, "randomnonexistentguy999")
            self.assertEqual(score, 0.0)

    # 7. Multi-Tenant Workspace Isolation
    def test_workspace_isolation(self):
        def _tenant_mock(workspace_id, field_type, terms):
            if workspace_id == 1:
                return ["Hasan", "Rakib"]
            elif workspace_id == 2:
                return ["Salma", "Nasir"]
            return []

        with patch.object(self.resolver, "get_candidates", side_effect=_tenant_mock):
            # Workspace 1 sees Rakib
            val, score = self.resolver.normalize(1, "salesperson", "Rakib")
            self.assertEqual(val, "Rakib")
            self.assertEqual(score, 1.0)

            # Workspace 2 does NOT see Rakib
            val, score = self.resolver.normalize(2, "salesperson", "Rakib")
            self.assertEqual(score, 0.0)

            # Workspace 2 sees Salma
            val, score = self.resolver.normalize(2, "salesperson", "সালমা")
            self.assertEqual(val, "Salma")
            self.assertGreaterEqual(score, 0.90)

    # 8. Large Dataset / Scalability (Handles large candidate sets cleanly)
    def test_large_candidate_set_performance(self):
        large_customers = [f"Customer_{i:05d}" for i in range(5000)]
        large_customers.append("Special Unique Buyer")

        with patch.object(self.resolver, "get_candidates", return_value=large_customers):
            val, score = self.resolver.normalize(1, "customer", "Special Unique")
            self.assertEqual(val, "Special Unique Buyer")
            self.assertGreaterEqual(score, 0.85)

if __name__ == "__main__":
    unittest.main()
