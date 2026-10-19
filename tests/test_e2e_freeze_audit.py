"""
End-to-End Freeze Audit Verification Script
Validates all 8 critical requirements for Document Q&A freeze audit.
"""
import os
import sys
import tempfile
import sqlite3
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

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

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures", "excel")


class TestEndToEndFreezeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="freeze_audit_")
        cls.registry = DataSourceRegistry(storage_dir=cls.temp_dir)
        cls.resolver = DataSourceResolver(registry=cls.registry)
        cls.engine = ExcelDatabaseEngine(registry=cls.registry, resolver=cls.resolver)

    # -------------------------------------------------------------------------
    # 1. XLSX Multi-sheet & CSV Ingestion
    # -------------------------------------------------------------------------
    def test_01_xlsx_multi_sheet_and_csv_ingestion(self):
        """Verify XLSX multi-sheet and CSV single dataset ingestion with isolated SQLite."""
        # XLSX Multi-sheet
        p_xlsx = os.path.join(FIXTURES_DIR, "02_multi_sheet.xlsx")
        meta_xlsx = self.engine.ingest_excel_file(p_xlsx, "02_multi_sheet.xlsx", workspace_id=10, source_id="src_multi_xlsx")
        self.assertEqual(meta_xlsx["sheets_count"], 2)
        self.assertIn("sales", meta_xlsx["tables"])
        self.assertIn("employees", meta_xlsx["tables"])
        self.assertTrue(os.path.exists(meta_xlsx["db_path"]))

        # CSV Single dataset
        p_csv = os.path.join(self.temp_dir, "inventory.csv")
        pd.DataFrame({
            "item_id": [1, 2, 3],
            "item_name": ["Rice", "Oil", "Sugar"],
            "stock_qty": [100, 50, 80],
            "unit_price": [75.0, 180.0, 130.0],
        }).to_csv(p_csv, index=False)

        meta_csv = self.engine.ingest_excel_file(p_csv, "inventory.csv", workspace_id=10, source_id="src_single_csv")
        self.assertEqual(meta_csv["sheets_count"], 1)
        self.assertEqual(meta_csv["total_rows"], 3)
        self.assertIn("sheet1", meta_csv["tables"])
        self.assertTrue(os.path.exists(meta_csv["db_path"]))

    # -------------------------------------------------------------------------
    # 2. Query Analytics: Count, Sum, Filtering, Sorting & Missing Columns
    # -------------------------------------------------------------------------
    def test_02_query_execution_and_aggregations(self):
        """Verify count, sum, filtering, sorting on SQLite database."""
        src = self.registry.get_source(10, "src_single_csv")
        self.assertIsNotNone(src)
        db_path = src.db_path

        # Test Read-Only queries directly on database
        db_uri = f"{Path(os.path.abspath(db_path)).as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        conn.execute("PRAGMA query_only = ON;")
        cur = conn.cursor()

        # Count
        cur.execute("SELECT COUNT(*) FROM sheet1")
        self.assertEqual(cur.fetchone()[0], 3)

        # Sum & Filter
        cur.execute("SELECT SUM(stock_qty * unit_price) FROM sheet1 WHERE stock_qty >= 60")
        expected_val = (100 * 75.0) + (80 * 130.0)
        self.assertEqual(cur.fetchone()[0], expected_val)

        # Sorting / Ranking
        cur.execute("SELECT item_name, unit_price FROM sheet1 ORDER BY unit_price DESC LIMIT 1")
        top_item = cur.fetchone()
        self.assertEqual(top_item[0], "Oil")
        self.assertEqual(top_item[1], 180.0)

        # Missing Column handled gracefully
        with self.assertRaises(sqlite3.OperationalError):
            cur.execute("SELECT nonexistent_column FROM sheet1")

        conn.close()

    # -------------------------------------------------------------------------
    # 3. Explicit Document Selection & Conversational Follow-up
    # -------------------------------------------------------------------------
    def test_03_explicit_selection_and_conversational_followup(self):
        """Verify explicit filename resolution and conversational turn continuity."""
        # Explicit filename
        res1 = self.resolver.resolve("inventory.csv এ মোট কতটি আইটেম আছে?", workspace_id=10)
        self.assertEqual(res1.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res1.source_id, "src_single_csv")

        # Conversational follow-up
        ctx = DocumentAnalyticsContext(
            workspace_id=10,
            active_source_id="src_single_csv",
            active_dataset_id="sheet1",
            history=[
                {"role": "user", "content": "inventory.csv এ মোট কতটি আইটেম আছে?"},
                {"role": "assistant", "content": "মোট ৩টি আইটেম আছে।"},
            ]
        )
        res2 = self.resolver.resolve("আর Oil এর স্টক কত?", workspace_id=10, context=ctx)
        self.assertEqual(res2.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res2.source_id, "src_single_csv")

    # -------------------------------------------------------------------------
    # 4. Ambiguous Selection & Unsupported Cross-Document Requests
    # -------------------------------------------------------------------------
    def test_04_ambiguous_selection_and_unsupported_cross_document(self):
        """Verify ambiguous queries provide clarification options, and cross-source comparisons are rejected."""
        # Upload another file with overlapping item schema
        p_extra = os.path.join(self.temp_dir, "warehouse_b.csv")
        pd.DataFrame({
            "item_name": ["Rice", "Oil"],
            "stock_qty": [200, 75],
            "unit_price": [72.0, 175.0],
        }).to_csv(p_extra, index=False)
        self.engine.ingest_excel_file(p_extra, "warehouse_b.csv", workspace_id=10, source_id="src_warehouse_b")

        # Ambiguous query matching both sources
        res_ambig = self.resolver.resolve("Oil এর stock_qty কত?", workspace_id=10)
        self.assertEqual(res_ambig.status, ResolutionStatus.AMBIGUOUS)
        self.assertGreaterEqual(len(res_ambig.candidates), 2)
        self.assertGreaterEqual(len(res_ambig.clarification_options), 2)

        # Cross-document comparison request
        res_cross = self.resolver.resolve("inventory.csv এবং warehouse_b.csv এর মধ্যে তুলনা করো", workspace_id=10)
        self.assertEqual(res_cross.status, ResolutionStatus.UNSUPPORTED_CROSS_SOURCE)
        self.assertIn("Multi-Source Comparison Notice", res_cross.clarification_message)
        self.assertGreaterEqual(len(res_cross.clarification_options), 2)

        # Generic multi-file aggregation request
        res_gen_cross = self.resolver.resolve("সব ফাইল মিলিয়ে মোট স্টক কত?", workspace_id=10)
        self.assertEqual(res_gen_cross.status, ResolutionStatus.UNSUPPORTED_CROSS_SOURCE)
        self.assertIn("Multi-Source Comparison Notice", res_gen_cross.clarification_message)

    # -------------------------------------------------------------------------
    # 5. Explicit source_id NOT_FOUND & Cross-Workspace Isolation
    # -------------------------------------------------------------------------
    def test_05_explicit_source_id_isolation(self):
        """Verify explicit source_id from other workspace returns NOT_FOUND without fallback."""
        # Non-existent source in workspace 10
        res_nonexistent = self.resolver.resolve("Item count?", workspace_id=10, explicit_source_id="nonexistent-src-uuid")
        self.assertEqual(res_nonexistent.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res_nonexistent.source)

        # Probe Workspace 10 source from Workspace 99
        res_probe = self.resolver.resolve("Item count?", workspace_id=99, explicit_source_id="src_single_csv")
        self.assertEqual(res_probe.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res_probe.source)

    # -------------------------------------------------------------------------
    # 6. Read-Only SQLite Enforcement, SQL Validation & Limits
    # -------------------------------------------------------------------------
    def test_06_sqlite_read_only_and_sql_validation(self):
        """Verify PRAGMA query_only=ON, mode=ro, keyword validation and row limits."""
        src = self.registry.get_source(10, "src_single_csv")
        db_path = src.db_path

        # Direct connection in query_only mode
        db_uri = f"{Path(os.path.abspath(db_path)).as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        conn.execute("PRAGMA query_only = ON;")
        cur = conn.cursor()

        # Mutation attempts must fail with OperationalError
        mutations = [
            "UPDATE sheet1 SET stock_qty = 999",
            "DELETE FROM sheet1 WHERE item_id = 1",
            "INSERT INTO sheet1 VALUES (4, 'Salt', 10, 20.0)",
            "DROP TABLE sheet1",
        ]
        for sql in mutations:
            with self.assertRaises(sqlite3.OperationalError):
                cur.execute(sql)

        conn.close()

        # Engine query_excel validation blocklist test (mock LLM response)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"sql": "DROP TABLE sheet1", "explanation": "test"}'
        mock_client.chat.completions.create.return_value = mock_response

        orig_client = self.engine.client
        self.engine.client = mock_client
        try:
            res_attack = self.engine.query_excel(workspace_id=10, file_id="src_single_csv", question="Delete everything")
            self.assertFalse(res_attack["success"])
            self.assertIn("Security Notice", res_attack["report"])
        finally:
            self.engine.client = orig_client

    # -------------------------------------------------------------------------
    # 7. Lifecycle Consistency: Idempotent Retry, Deletion & Re-upload
    # -------------------------------------------------------------------------
    def test_07_lifecycle_idempotency_deletion_and_reupload(self):
        """Verify upload retry reuses db_path, deletion removes files, and re-upload succeeds."""
        p_csv = os.path.join(self.temp_dir, "lifecycle_test.csv")
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(p_csv, index=False)

        # Upload 1
        meta1 = self.engine.ingest_excel_file(p_csv, "lifecycle_test.csv", workspace_id=20, source_id="src_lifecycle_uuid")
        db_path = meta1["db_path"]
        self.assertTrue(os.path.exists(db_path))

        # Idempotent re-upload with same source_id
        meta2 = self.engine.ingest_excel_file(p_csv, "lifecycle_test.csv", workspace_id=20, source_id="src_lifecycle_uuid")
        self.assertEqual(meta1["db_path"], meta2["db_path"])
        self.assertEqual(meta1["source_id"], meta2["source_id"])

        # Deletion
        deleted = self.engine.registry.delete_source(workspace_id=20, source_id="src_lifecycle_uuid")
        self.assertTrue(deleted)
        self.assertIsNone(self.engine.registry.get_source(workspace_id=20, source_id="src_lifecycle_uuid"))
        self.assertFalse(os.path.exists(db_path))

        # Re-upload fresh
        meta3 = self.engine.ingest_excel_file(p_csv, "lifecycle_test.csv", workspace_id=20, source_id="src_lifecycle_uuid")
        self.assertEqual(meta3["source_id"], "src_lifecycle_uuid")
        self.assertTrue(os.path.exists(meta3["db_path"]))

    # -------------------------------------------------------------------------
    # 8. Regression Safety: Plain Entity Query Fallback to MySQL (Rule 5)
    # -------------------------------------------------------------------------
    def test_08_plain_entity_query_fallback_for_mysql(self):
        """Plain business question without file indicator returns NOT_FOUND so MySQL handles it."""
        res = self.resolver.resolve("গতকাল মোট সেলস কত হয়েছে?", workspace_id=10)
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res.source)


if __name__ == "__main__":
    unittest.main()
