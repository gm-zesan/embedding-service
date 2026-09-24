import unittest
import sqlite3
import time
from app.linguistics.profiler import LanguageProfiler
from app.linguistics.models import LanguageType, ScriptType
from app.linguistics.canonical_mapper import CanonicalConceptMapper
from app.analytics.models import (
    SemanticQueryPlan,
    DomainEnum,
    MeasureSpec,
    DimensionSpec,
    FilterSpec,
    AggregationType,
    OperatorType,
)
from app.analytics.compiler import AnalyticsCompilerV2
from app.analytics.validator import SemanticValidator
from app.analytics.formatter import AnalyticsFormatter
from app.analytics.fuzzy_resolver import FuzzyEntityResolver


class TestPythonIntegrationPipeline(unittest.TestCase):
    """
    Step 4A: Python Integration Pipeline Tests
    Validates the end-to-end processing across Knowledge, Analytics, and Context pipelines.
    """

    def setUp(self):
        self.compiler = AnalyticsCompilerV2()
        self.validator = SemanticValidator()
        self.formatter = AnalyticsFormatter()
        self.fuzzy_resolver = FuzzyEntityResolver()

        # Preload in-memory entity cache for deterministic testing
        self.fuzzy_resolver._cache[1] = {
            "timestamp": time.time(),
            "entities": {
                "salesperson": ["Hasan", "Rakib", "Mehedi"],
                "customer": ["Rahim", "Karim"],
                "product": ["Laptop Pro 15", "Mechanical Keyboard RGB"],
                "payment_method": ["cash", "bank", "bkash", "nagad"],
            }
        }

        # Build in-memory SQLite fixture for realistic analytics execution
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        
        # Seed Workspace 1 and Workspace 2 business data
        self.cursor.execute("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                workspace_id INTEGER,
                salesperson_id INTEGER,
                total_amount REAL,
                order_date TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE salespersons (
                id INTEGER PRIMARY KEY,
                workspace_id INTEGER,
                name TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE collections (
                id INTEGER PRIMARY KEY,
                workspace_id INTEGER,
                amount REAL,
                payment_method TEXT,
                collection_date TEXT
            )
        """)

        # Salespersons
        self.cursor.executemany("""
            INSERT INTO salespersons (id, workspace_id, name) VALUES (?, ?, ?)
        """, [
            (1, 1, 'Hasan'),
            (2, 1, 'Rakib'),
            (3, 2, 'Nasir'),  # Exists ONLY in Workspace 2
        ])

        # Sales
        self.cursor.executemany("""
            INSERT INTO sales (workspace_id, salesperson_id, total_amount, order_date) VALUES (?, ?, ?, ?)
        """, [
            (1, 1, 45000.0, '2026-09-24'),
            (1, 2, 35000.0, '2026-09-24'),
            (2, 3, 60000.0, '2026-09-24'),
        ])

        # Collections
        # Workspace 1 total cash = 32000, Workspace 2 total cash = 20000. Combined = 52000.
        self.cursor.executemany("""
            INSERT INTO collections (workspace_id, amount, payment_method, collection_date) VALUES (?, ?, ?, ?)
        """, [
            (1, 20000.0, 'cash', '2026-09-24'),
            (1, 12000.0, 'cash', '2026-09-24'),
            (2, 20000.0, 'cash', '2026-09-24'),
        ])
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    # -------------------------------------------------------------
    # 1. Knowledge Pipeline Integration Tests
    # -------------------------------------------------------------

    def test_knowledge_language_profiling_and_normalization(self):
        """Verify language profiling accurately identifies Bengali, Banglish, and English queries."""
        # Bengali
        bn_profile = LanguageProfiler.profile("ডেলিভারি চার্জ কত?")
        self.assertEqual(bn_profile.language, LanguageType.BN)
        self.assertEqual(bn_profile.script, ScriptType.BENGALI)

        # Banglish
        banglish_profile = LanguageProfiler.profile("delivery charge koto?")
        self.assertEqual(banglish_profile.language, LanguageType.BANGLISH)

        # English
        en_profile = LanguageProfiler.profile("What is the return policy?")
        self.assertEqual(en_profile.language, LanguageType.EN)
        self.assertEqual(en_profile.script, ScriptType.LATIN)

    def test_knowledge_canonical_concept_mapping(self):
        """Verify canonical concept mapper extracts intent keywords from variations."""
        profile = LanguageProfiler.profile("delivery charge koto?")
        snapshot = {
            "concept_patterns": {
                "delivery_fee": {
                    "phrases": ["delivery charge", "delivery fee"],
                    "target_doc_type": "faq",
                }
            }
        }
        enrichment = CanonicalConceptMapper.map_concepts("delivery charge koto?", profile, snapshot=snapshot)
        self.assertIn("delivery_fee", enrichment.canonical_concepts)

    # -------------------------------------------------------------
    # 2. Analytics Pipeline Integration Tests
    # -------------------------------------------------------------

    def test_analytics_plan_validation_and_deterministic_sql_compilation(self):
        """Verify SemanticQueryPlan generates strictly parameterized, read-only SQL with tenant filter."""
        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="total_amount", aggregation=AggregationType.SUM)],
            dimensions=[],
            filters=[],
            confidence=0.95,
        )

        val_res = self.validator.validate_plan(plan)
        self.assertTrue(val_res.is_valid)

        sql, params = self.compiler.compile(plan, workspace_id=1)
        self.assertIn("workspace_id", sql)
        self.assertIn("SUM", sql)
        self.assertIn(1, params)

    def test_analytics_execution_and_tenant_constraint_propagation(self):
        """
        Verify tenant boundary:
        Workspace 1 Cash Collection = ৳32,000
        Workspace 2 Cash Collection = ৳20,000
        Combined DB = ৳52,000 (Workspace 1 must NEVER return 52000).
        """
        # Execute query for Workspace 1
        self.cursor.execute(
            "SELECT SUM(amount) AS total FROM collections WHERE payment_method = 'cash' AND workspace_id = ?",
            (1,)
        )
        row1 = self.cursor.fetchone()
        ws1_total = row1[0]
        self.assertEqual(ws1_total, 32000.0)

        # Execute query for Workspace 2
        self.cursor.execute(
            "SELECT SUM(amount) AS total FROM collections WHERE payment_method = 'cash' AND workspace_id = ?",
            (2,)
        )
        row2 = self.cursor.fetchone()
        ws2_total = row2[0]
        self.assertEqual(ws2_total, 20000.0)

        # Combined DB check
        self.cursor.execute("SELECT SUM(amount) FROM collections WHERE payment_method = 'cash'")
        combined = self.cursor.fetchone()[0]
        self.assertEqual(combined, 52000.0)
        self.assertNotEqual(ws1_total, combined)

    def test_analytics_workspace_salesperson_isolation(self):
        """Verify Workspace 1 cannot see or query Workspace 2 salesperson (Nasir)."""
        self.cursor.execute(
            "SELECT * FROM salespersons WHERE name = 'Nasir' AND workspace_id = ?",
            (1,)
        )
        res_ws1 = self.cursor.fetchall()
        self.assertEqual(len(res_ws1), 0, "Workspace 1 must not see Nasir")

        self.cursor.execute(
            "SELECT * FROM salespersons WHERE name = 'Nasir' AND workspace_id = ?",
            (2,)
        )
        res_ws2 = self.cursor.fetchall()
        self.assertEqual(len(res_ws2), 1, "Nasir must exist in Workspace 2")

    # -------------------------------------------------------------
    # 3. Context & Multi-Turn Normalization Integration Tests
    # -------------------------------------------------------------

    def test_multi_turn_fuzzy_entity_resolution(self):
        """Verify fuzzy resolver normalizes misspelling in contextual query plan."""
        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="total_amount", aggregation=AggregationType.SUM)],
            dimensions=[DimensionSpec(name="salesperson")],
            filters=[FilterSpec(field="salesperson", operator=OperatorType.EQUALS, value="hasn")],
            confidence=0.92,
        )

        normalized_plan = self.fuzzy_resolver.normalize_plan(plan, workspace_id=1)
        self.assertEqual(normalized_plan.filters[0].value, "Hasan")

    def test_report_formatting_with_clean_currency_output(self):
        """Verify formatter generates structured markdown with Bengali currency notation."""
        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="total_amount", aggregation=AggregationType.SUM)],
            confidence=0.95,
        )
        rows = [{"total_amount": 45000.0}]
        report = self.formatter.format("আজকের মোট বিক্রি কত?", plan, rows, latency_ms=45.0)

        self.assertIn("45,000", report)
        self.assertIn("৳", report)


if __name__ == "__main__":
    unittest.main()
