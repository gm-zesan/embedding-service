import unittest
import os
import sqlite3
import tempfile
import pandas as pd
from app.analytics.data_source import (
    DataSource,
    Dataset,
    DataSourceFormat,
    DocumentAnalyticsContext,
    ResolutionStatus,
    SourceResolutionResult,
)
from app.analytics.source_registry import DataSourceRegistry
from app.analytics.source_resolver import DataSourceResolver
from app.analytics.excel_engine import ExcelDatabaseEngine

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "excel")


class TestDataSourceArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_storage = tempfile.mkdtemp(prefix="test_sources_")
        cls.registry = DataSourceRegistry(storage_dir=cls.temp_storage)
        cls.resolver = DataSourceResolver(registry=cls.registry)
        cls.engine = ExcelDatabaseEngine(registry=cls.registry, resolver=cls.resolver)

        # Ingest test dataset 1 for Workspace 1: products.xlsx
        p1 = os.path.join(FIXTURES_DIR, "01_single_sheet.xlsx")
        cls.src1 = cls.engine.ingest_excel_file(p1, "products.xlsx", workspace_id=1)

        # Ingest test dataset 2 for Workspace 1: multi-sheet sales_and_employees.xlsx
        p2 = os.path.join(FIXTURES_DIR, "02_multi_sheet.xlsx")
        cls.src2 = cls.engine.ingest_excel_file(p2, "sales_and_employees.xlsx", workspace_id=1)

        # Ingest test dataset 3 for Workspace 1: CSV file
        p3_csv = os.path.join(cls.temp_storage, "courier_deliveries.csv")
        df_csv = pd.DataFrame({
            "tracking_id": ["TRK-1", "TRK-2"],
            "cod_amount": [4500, 3200],
            "destination": ["Dhaka", "Chittagong"],
            "status": ["Delivered", "In Transit"]
        })
        df_csv.to_csv(p3_csv, index=False)
        cls.src3 = cls.engine.ingest_excel_file(p3_csv, "courier_deliveries.csv", workspace_id=1)

        # Ingest test dataset for Workspace 2 (Tenant B)
        cls.src_ws2 = cls.engine.ingest_excel_file(p1, "tenant_b_data.xlsx", workspace_id=2)

    # ── 1. Source Resolution Tests ────────────────────────────────────

    def test_01_explicit_source_id_resolution(self):
        """Explicit source_id correctly resolves the targeted source."""
        res = self.resolver.resolve(
            query="Total price koto?",
            workspace_id=1,
            explicit_source_id=self.src1["file_id"]
        )
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, self.src1["file_id"])
        self.assertEqual(res.source.original_filename, "products.xlsx")

    def test_02_explicit_filename_in_query_resolution(self):
        """Query mentioning 'products.xlsx' or 'sales_and_employees' explicitly resolves."""
        res1 = self.resolver.resolve(query="products.xlsx file e stock koto?", workspace_id=1)
        self.assertEqual(res1.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res1.source.original_filename, "products.xlsx")

        res2 = self.resolver.resolve(query="sales_and_employees file er total sales koto?", workspace_id=1)
        self.assertEqual(res2.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res2.source.original_filename, "sales_and_employees.xlsx")

    def test_03_explicit_sheet_dataset_resolution(self):
        """Query mentioning specific sheet 'employees' inside multi-sheet file resolves to that sheet."""
        res = self.resolver.resolve(query="employees sheet e total salary koto?", workspace_id=1)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source.original_filename, "sales_and_employees.xlsx")
        self.assertEqual(res.dataset_ids, ["employees"])

    def test_04_csv_source_resolution(self):
        """CSV file resolves correctly by filename and unique columns."""
        res = self.resolver.resolve(query="courier_deliveries file er delivered COD amount koto?", workspace_id=1)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source.original_filename, "courier_deliveries.csv")
        self.assertEqual(res.source.format, DataSourceFormat.CSV)

    def test_05_ambiguity_detection_multiple_matching_candidates(self):
        """When multiple files match a generic term, status is AMBIGUOUS with clarification options."""
        # Create a second product catalog with overlapping 'price' and 'quantity' columns
        p_extra = os.path.join(self.temp_storage, "catalog_secondary.xlsx")
        df_extra = pd.DataFrame({"product": ["Rice", "Oil"], "price": [120, 800], "quantity": [50, 20]})
        df_extra.to_excel(p_extra, index=False)
        self.engine.ingest_excel_file(p_extra, "catalog_secondary.xlsx", workspace_id=1)

        res = self.resolver.resolve(query="Rice er price koto?", workspace_id=1)
        self.assertEqual(res.status, ResolutionStatus.AMBIGUOUS)
        self.assertGreaterEqual(len(res.candidates), 2)
        self.assertTrue(any(c["filename"] == "products.xlsx" for c in res.candidates))
        self.assertTrue(any(c["filename"] == "catalog_secondary.xlsx" for c in res.candidates))
        self.assertIn("🤔 **Ambiguous Source Context**", res.clarification_message)
        self.assertTrue(len(res.clarification_options) >= 2)

    def test_06_not_found_when_no_matching_source(self):
        """Non-existent workspace or completely unrelated query returns NOT_FOUND."""
        res = self.resolver.resolve(query="Show spaceship trajectory data", workspace_id=999)
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)

    def test_07_cross_source_comparison_rejection(self):
        """Asking to compare two separate files returns UNSUPPORTED_CROSS_SOURCE."""
        res = self.resolver.resolve(
            query="products.xlsx আর catalog_secondary.xlsx এর মধ্যে তুলনা করো",
            workspace_id=1
        )
        self.assertEqual(res.status, ResolutionStatus.UNSUPPORTED_CROSS_SOURCE)
        self.assertIn("Multi-Source Comparison Notice", res.clarification_message)

    # ── 2. Conversational Context & Multi-Turn Tests ──────────────────

    def test_08_conversational_followup_retains_active_source_and_dataset(self):
        """Follow-up turn 'আর Oil-এরটা?' retains active source and dataset."""
        ctx = DocumentAnalyticsContext(
            workspace_id=1,
            active_source_id=self.src1["file_id"],
            active_dataset_id="products",
            history=[
                {"role": "user", "content": "products.xlsx er Rice er stock koto?"},
                {"role": "assistant", "content": "Rice stock is 10 units."}
            ]
        )
        res = self.resolver.resolve(
            query="আর Oil-এরটা?",
            workspace_id=1,
            context=ctx
        )
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, self.src1["file_id"])

    def test_09_explicit_new_source_overrides_previous_context(self):
        """Explicitly naming a new source overrides the prior active context."""
        ctx = DocumentAnalyticsContext(
            workspace_id=1,
            active_source_id=self.src1["file_id"],
            active_dataset_id="products",
        )
        res = self.resolver.resolve(
            query="এখন courier_deliveries file er total COD koto?",
            workspace_id=1,
            context=ctx
        )
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source.original_filename, "courier_deliveries.csv")

    # ── 3. Workspace Tenant Security Tests ────────────────────────────

    def test_10_workspace_tenant_isolation_cannot_see_other_sources(self):
        """Workspace 1 cannot list or see Workspace 2's data sources."""
        ws1_sources = self.registry.list_sources(workspace_id=1)
        ws2_sources = self.registry.list_sources(workspace_id=2)

        ws1_names = [s.original_filename for s in ws1_sources]
        ws2_names = [s.original_filename for s in ws2_sources]

        self.assertNotIn("tenant_b_data.xlsx", ws1_names)
        self.assertIn("tenant_b_data.xlsx", ws2_names)

    def test_11_workspace_cannot_resolve_other_workspace_source_id(self):
        """P2-2: Attempting to resolve Workspace 2's source_id from Workspace 1 returns strictly NOT_FOUND without falling back."""
        res = self.resolver.resolve(
            query="Rice er stock koto?",
            workspace_id=1,
            explicit_source_id=self.src_ws2["file_id"]
        )
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res.source)
        self.assertIn("not found in workspace", res.reason)

    # ── 4. Query Safety & Read-Only Sandbox Tests ─────────────────────

    def test_12_read_only_sqlite_sandbox_rejects_mutations(self):
        """SQL generation with mutation statements (INSERT, DROP, DELETE, ATTACH) is strictly blocked."""
        res_del = self.engine.query_excel(workspace_id=1, file_id=self.src1["file_id"], question="DROP TABLE products")
        self.assertFalse(res_del["success"] and bool(res_del.get("rows")))

        # Check table still exists and data is intact
        conn = sqlite3.connect(self.src1["db_path"])
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM products")
        count = cur.fetchone()[0]
        conn.close()
        self.assertGreater(count, 0)

    # ── 5. Lifecycle & Deletion Tests ─────────────────────────────────

    def test_13_delete_source_removes_metadata_and_sqlite(self):
        """Deleting a source removes its descriptor and physical SQLite database."""
        temp_file = os.path.join(self.temp_storage, "to_delete.xlsx")
        df = pd.DataFrame({"col": [1, 2, 3]})
        df.to_excel(temp_file, index=False)
        created = self.engine.ingest_excel_file(temp_file, "to_delete.xlsx", workspace_id=99)
        src_id = created["file_id"]

        self.assertIsNotNone(self.registry.get_source(99, src_id))
        self.assertTrue(os.path.exists(created["db_path"]))

        deleted = self.registry.delete_source(workspace_id=99, source_id=src_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.registry.get_source(99, src_id))
        self.assertFalse(os.path.exists(created["db_path"]))


if __name__ == "__main__":
    unittest.main()
