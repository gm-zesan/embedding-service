"""
Tests for Upload-Time Searchable Metadata Generation and Automatic Source Resolution.
Covers:
- Deterministic and bilingual metadata generation
- Sanitization / no secrets leaked
- Typesense collection schema and indexing
- /analytics/sources/resolve endpoint (explicit, auto-single, multi-match, ambiguous, not found)
- Tenant isolation across workspaces
- Deletion cleanup
"""

import shutil
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.analytics.data_source import DataSource, Dataset, DataSourceFormat
from app.analytics.metadata_generator import metadata_generator
from app.analytics.source_registry import source_registry


class TestAnalyticsSourceResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_storage_dir = source_registry.storage_dir
        source_registry.storage_dir = self.test_dir

    def tearDown(self):
        source_registry.storage_dir = self.original_storage_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_metadata_generator_bilingual_expansion(self):
        """Verify that metadata generator extracts bilingual (English & Bengali) synonyms."""
        dset = Dataset(
            dataset_id="payroll_table",
            dataset_name="Monthly Payroll",
            sheet_name="Payroll 2024",
            columns=["employee_id", "employee_name", "department", "salary", "bonus"],
            data_types={"employee_id": "INTEGER", "employee_name": "TEXT", "department": "TEXT", "salary": "REAL", "bonus": "REAL"},
            numeric_measures=["salary", "bonus"],
            text_dimensions=["employee_name", "department"],
            row_count=50,
        )
        src = DataSource(
            workspace_id=1,
            source_id="payroll_src_1",
            original_filename="employee_payroll.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/fake_payroll.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="payroll_src_1",
            datasets=[dset],
            total_rows=50,
        )

        meta = metadata_generator.generate_metadata(src, document_id="doc_101", use_llm=False)

        self.assertEqual(meta.workspace_id, 1)
        self.assertEqual(meta.source_id, "payroll_src_1")
        self.assertEqual(meta.document_id, "doc_101")
        self.assertEqual(meta.ingestion_purpose, "data_analytics")
        self.assertEqual(meta.lifecycle_status, "active")
        self.assertIn("employee_payroll.xlsx", meta.dataset_summary)
        self.assertIn("Payroll 2024", meta.sheet_names)

        # Check that Bengali and English keywords were extracted
        keywords = set(meta.searchable_keywords)
        self.assertIn("salary", keywords)
        self.assertIn("বেতন", keywords)
        self.assertIn("employee", keywords)
        self.assertIn("কর্মী", keywords)
        self.assertIn("bonus", keywords)
        self.assertIn("বোনাস", keywords)
        self.assertIn("department", keywords)
        self.assertIn("বিভাগ", keywords)

    def test_metadata_generator_never_leaks_secrets(self):
        """Verify that sensitive column names or secrets are not dumped as raw plain values."""
        dset = Dataset(
            dataset_id="secure_table",
            dataset_name="User Accounts",
            sheet_name="Users",
            columns=["user_id", "password_hash", "api_key", "salary"],
            data_types={"user_id": "INTEGER", "password_hash": "TEXT", "api_key": "TEXT", "salary": "REAL"},
            row_count=10,
        )
        src = DataSource(
            workspace_id=1,
            source_id="sec_src_1",
            original_filename="accounts.csv",
            format=DataSourceFormat.CSV,
            stored_path="/tmp/fake.csv",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sec_src_1",
            datasets=[dset],
            total_rows=10,
        )

        meta = metadata_generator.generate_metadata(src, use_llm=False)
        self.assertIn("password_hash", meta.columns_summary)
        self.assertEqual(meta.workspace_id, 1)

    def test_resolve_source_explicit_override(self):
        """Explicit source ID overrides auto-resolution with 100% confidence."""
        src = DataSource(
            workspace_id=42,
            source_id="src_explicit_99",
            original_filename="budget_2024.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/b.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="src_explicit_99",
            datasets=[],
        )
        source_registry.register_source(src)

        res = self.client.post("/analytics/sources/resolve", json={
            "query": "What is the total expenditure?",
            "workspace_id": 42,
            "explicit_source_id": "src_explicit_99",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "resolved")
        self.assertEqual(data["source_id"], "src_explicit_99")
        self.assertEqual(data["confidence"], 1.0)

    def test_resolve_source_tenant_isolation(self):
        """Cross-workspace source resolution is strictly rejected."""
        src_ws1 = DataSource(
            workspace_id=1,
            source_id="src_ws1",
            original_filename="private_sales_ws1.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/ws1.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="src_ws1",
            datasets=[],
        )
        source_registry.register_source(src_ws1)

        # Workspace 2 attempts to explicitly or implicitly access ws 1 source
        res = self.client.post("/analytics/sources/resolve", json={
            "query": "Show sales report",
            "workspace_id": 2,
            "explicit_source_id": "src_ws1",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["status"], "not_found")

        # Workspace 2 implicit search returns not found because ws 2 has 0 sources
        res_implicit = self.client.post("/analytics/sources/resolve", json={
            "query": "Show sales report",
            "workspace_id": 2,
        })
        self.assertEqual(res_implicit.status_code, 200)
        self.assertEqual(res_implicit.json()["status"], "not_found")

    def test_resolve_source_single_active_auto_select(self):
        """When a workspace has exactly 1 active generic spreadsheet, it is auto-resolved."""
        src = DataSource(
            workspace_id=5,
            source_id="sales_only",
            original_filename="q1_sales.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/sales.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sales_only",
            datasets=[
                Dataset(
                    dataset_id="q1",
                    dataset_name="Q1",
                    columns=["order_id", "total_price"],
                    data_types={"order_id": "INTEGER", "total_price": "REAL"},
                )
            ],
        )
        source_registry.register_source(src)

        res = self.client.post("/analytics/sources/resolve", json={
            "query": "মোট কত বিক্রি হয়েছে?",
            "workspace_id": 5,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "resolved")
        self.assertEqual(data["source_id"], "sales_only")
        self.assertEqual(data["confidence"], 1.0)

    def test_resolve_source_typesense_winner(self):
        """When Typesense returns a clear winner, that source is resolved."""
        src1 = DataSource(
            workspace_id=10,
            source_id="payroll_10",
            original_filename="company_payroll.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/p.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="payroll_10",
            datasets=[],
        )
        src2 = DataSource(
            workspace_id=10,
            source_id="inventory_10",
            original_filename="warehouse_inventory.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/i.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="inventory_10",
            datasets=[],
        )
        source_registry.register_source(src1)
        source_registry.register_source(src2)

        fake_hits = [
            {
                "document": {
                    "source_id": "payroll_10",
                    "original_filename": "company_payroll.xlsx",
                    "dataset_summary": "Employee salaries and department payroll data",
                },
                "text_match": 1500,
            },
            {
                "document": {
                    "source_id": "inventory_10",
                    "original_filename": "warehouse_inventory.xlsx",
                    "dataset_summary": "Stock items and warehouse quantities",
                },
                "text_match": 200,
            },
        ]

        with patch("app.typesense_engine.search_analytics_metadata", return_value=fake_hits):
            res = self.client.post("/analytics/sources/resolve", json={
                "query": "কার বেতন ৫০ হাজারের বেশি?",
                "workspace_id": 10,
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "resolved")
            self.assertEqual(data["source_id"], "payroll_10")
            self.assertEqual(data["confidence"], 0.9)

    def test_resolve_source_ambiguous_two_candidates(self):
        """When two sources have very close scores, status is ambiguous with options."""
        src1 = DataSource(
            workspace_id=10,
            source_id="sales_2023",
            original_filename="sales_2023.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/s23.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sales_2023",
            datasets=[],
        )
        src2 = DataSource(
            workspace_id=10,
            source_id="sales_2024",
            original_filename="sales_2024.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/s24.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sales_2024",
            datasets=[],
        )
        source_registry.register_source(src1)
        source_registry.register_source(src2)

        fake_hits = [
            {
                "document": {
                    "source_id": "sales_2023",
                    "original_filename": "sales_2023.xlsx",
                    "dataset_summary": "Annual sales for year 2023",
                },
                "text_match": 1000,
            },
            {
                "document": {
                    "source_id": "sales_2024",
                    "original_filename": "sales_2024.xlsx",
                    "dataset_summary": "Annual sales for year 2024",
                },
                "text_match": 980,  # Very close to 1000 -> ambiguous!
            },
        ]

        with patch("app.typesense_engine.search_analytics_metadata", return_value=fake_hits):
            res = self.client.post("/analytics/sources/resolve", json={
                "query": "Total sales koto?",
                "workspace_id": 10,
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["status"], "ambiguous")
            self.assertGreaterEqual(len(data["clarification_options"]), 2)
            self.assertTrue(any("sales_2023" in opt for opt in data["clarification_options"]))
            self.assertTrue(any("sales_2024" in opt for opt in data["clarification_options"]))

    def test_delete_source_removes_from_registry_and_typesense(self):
        """Deleting a source removes it from source registry and cleans Typesense."""
        src = DataSource(
            workspace_id=7,
            source_id="src_to_delete",
            original_filename="temp.csv",
            format=DataSourceFormat.CSV,
            stored_path="/tmp/temp.csv",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="src_to_delete",
            datasets=[],
        )
        source_registry.register_source(src)
        self.assertIsNotNone(source_registry.get_source(7, "src_to_delete"))

        with patch("app.typesense_engine.delete_analytics_metadata_by_source", return_value=True) as mock_ts_del:
            res = self.client.delete("/analytics/sources/src_to_delete?workspace_id=7")
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["success"])
            self.assertIsNone(source_registry.get_source(7, "src_to_delete"))
            mock_ts_del.assert_called_once_with(mock_ts_del.call_args[0][0], workspace_id=7, source_id="src_to_delete")

    def test_typesense_score_margin_payroll_vs_sales(self):
        """Clear winner test: payroll vs sales datasets resolve cleanly based on BM25 score separation."""
        src_payroll = DataSource(
            workspace_id=20,
            source_id="payroll_src",
            original_filename="employee_payroll.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/p.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="payroll_src",
            datasets=[],
        )
        src_sales = DataSource(
            workspace_id=20,
            source_id="sales_src",
            original_filename="store_sales.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/s.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sales_src",
            datasets=[],
        )
        source_registry.register_source(src_payroll)
        source_registry.register_source(src_sales)

        # 1. Query targeting payroll: payroll has high BM25 score, sales has 0
        payroll_hits = [
            {
                "document": {"source_id": "payroll_src", "original_filename": "employee_payroll.xlsx"},
                "text_match": 850,
            },
            {
                "document": {"source_id": "sales_src", "original_filename": "store_sales.xlsx"},
                "text_match": 0,
            },
        ]
        with patch("app.typesense_engine.search_analytics_metadata", return_value=payroll_hits):
            res = self.client.post("/analytics/sources/resolve", json={
                "query": "কর্মীদের বেতন তালিকা",
                "workspace_id": 20,
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "resolved")
            self.assertEqual(data["source_id"], "payroll_src")

        # 2. Query targeting sales: sales has high score (750), payroll has negligible/zero (0)
        sales_hits = [
            {
                "document": {"source_id": "sales_src", "original_filename": "store_sales.xlsx"},
                "text_match": 750,
            },
            {
                "document": {"source_id": "payroll_src", "original_filename": "employee_payroll.xlsx"},
                "text_match": 0,
            },
        ]
        with patch("app.typesense_engine.search_analytics_metadata", return_value=sales_hits):
            res = self.client.post("/analytics/sources/resolve", json={
                "query": "Total sales volume and revenue",
                "workspace_id": 20,
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "resolved")
            self.assertEqual(data["source_id"], "sales_src")

    def test_typesense_duplicate_similar_datasets_exact_tie_ambiguous(self):
        """Exact tie or score within 1.20x margin between duplicate/similar datasets is ambiguous."""
        src1 = DataSource(
            workspace_id=25,
            source_id="payroll_dhaka",
            original_filename="payroll_dhaka.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/d.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="payroll_dhaka",
            datasets=[],
        )
        src2 = DataSource(
            workspace_id=25,
            source_id="payroll_ctg",
            original_filename="payroll_ctg.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/c.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="payroll_ctg",
            datasets=[],
        )
        source_registry.register_source(src1)
        source_registry.register_source(src2)

        # Exact tie: BM25 score 500 vs 500
        tied_hits = [
            {
                "document": {"source_id": "payroll_dhaka", "original_filename": "payroll_dhaka.xlsx"},
                "text_match": 500,
            },
            {
                "document": {"source_id": "payroll_ctg", "original_filename": "payroll_ctg.xlsx"},
                "text_match": 500,
            },
        ]
        with patch("app.typesense_engine.search_analytics_metadata", return_value=tied_hits):
            res = self.client.post("/analytics/sources/resolve", json={
                "query": "What is the payroll expenditure?",
                "workspace_id": 25,
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["status"], "ambiguous")
            self.assertEqual(len(data["clarification_options"]), 2)

    def test_typesense_weak_matches_and_zero_scores_never_resolve(self):
        """Hits below MIN_TEXT_MATCH_SCORE (15) or with score 0 are treated as weak/no match."""
        src1 = DataSource(
            workspace_id=30,
            source_id="doc_a",
            original_filename="random_doc.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/a.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="doc_a",
            datasets=[],
        )
        src2 = DataSource(
            workspace_id=30,
            source_id="doc_b",
            original_filename="other_doc.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/b.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="doc_b",
            datasets=[],
        )
        source_registry.register_source(src1)
        source_registry.register_source(src2)

        # Both scores are below threshold (e.g. 5 and 3)
        weak_hits = [
            {
                "document": {"source_id": "doc_a", "original_filename": "random_doc.xlsx"},
                "text_match": 5,
            },
            {
                "document": {"source_id": "doc_b", "original_filename": "other_doc.xlsx"},
                "text_match": 3,
            },
        ]
        with patch("app.typesense_engine.search_analytics_metadata", return_value=weak_hits):
            res = self.client.post("/analytics/sources/resolve", json={
                "query": "completely unrelated query zzz",
                "workspace_id": 30,
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            # Weak hits are ignored; since neither matches locally, returns not_found
            self.assertFalse(data["success"])
            self.assertEqual(data["status"], "not_found")

    def test_cross_workspace_identical_filenames_and_schemas(self):
        """Identical filenames (sales.xlsx) in two workspaces are strictly isolated by workspace_id."""
        src_ws1 = DataSource(
            workspace_id=101,
            source_id="sales_ws101",
            original_filename="sales.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/ws101_sales.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sales_ws101",
            datasets=[
                Dataset(
                    dataset_id="sales",
                    dataset_name="Sales",
                    columns=["order_id", "total_amount"],
                    data_types={"order_id": "INTEGER", "total_amount": "REAL"},
                )
            ],
        )
        src_ws2 = DataSource(
            workspace_id=102,
            source_id="sales_ws102",
            original_filename="sales.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/ws102_sales.xlsx",
            uploaded_at="2026-10-19 12:00:00",
            file_hash="sales_ws102",
            datasets=[
                Dataset(
                    dataset_id="sales",
                    dataset_name="Sales",
                    columns=["order_id", "total_amount"],
                    data_types={"order_id": "INTEGER", "total_amount": "REAL"},
                )
            ],
        )
        source_registry.register_source(src_ws1)
        source_registry.register_source(src_ws2)

        # Query in workspace 101 must resolve to sales_ws101 only
        res1 = self.client.post("/analytics/sources/resolve", json={
            "query": "sales.xlsx total",
            "workspace_id": 101,
        })
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["source_id"], "sales_ws101")

        # Query in workspace 102 must resolve to sales_ws102 only
        res2 = self.client.post("/analytics/sources/resolve", json={
            "query": "sales.xlsx total",
            "workspace_id": 102,
        })
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["source_id"], "sales_ws102")

        # Attempting explicit override from WS 102 with WS 101 source_id fails
        res_cross = self.client.post("/analytics/sources/resolve", json={
            "query": "sales.xlsx total",
            "workspace_id": 102,
            "explicit_source_id": "sales_ws101",
        })
        self.assertEqual(res_cross.status_code, 200)
        self.assertEqual(res_cross.json()["status"], "not_found")


if __name__ == "__main__":
    unittest.main()

