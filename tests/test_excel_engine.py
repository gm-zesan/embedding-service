import unittest
import os
import sqlite3
from app.analytics.excel_engine import ExcelDatabaseEngine

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "excel")

class TestExcelDatabaseEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ExcelDatabaseEngine()

    def test_single_sheet_ingestion_and_sql(self):
        """01_single_sheet: Verify virtual table creation and numeric aggregation."""
        file_path = os.path.join(FIXTURES_DIR, "01_single_sheet.xlsx")
        meta = self.engine.ingest_excel_file(file_path, "01_single_sheet.xlsx", workspace_id=101)
        self.assertEqual(meta["sheets_count"], 1)
        self.assertIn("products", meta["tables"])

        conn = sqlite3.connect(meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT SUM(price * quantity) as total_val FROM products")
        row = cur.fetchone()
        conn.close()

        expected_val = 75000*2 + 1500*10 + 3500*5
        self.assertEqual(row["total_val"], expected_val)

    def test_multi_sheet_virtual_tables(self):
        """02_multi_sheet: Verify multiple sheets become distinct virtual tables."""
        file_path = os.path.join(FIXTURES_DIR, "02_multi_sheet.xlsx")
        meta = self.engine.ingest_excel_file(file_path, "02_multi_sheet.xlsx", workspace_id=102)
        self.assertEqual(meta["sheets_count"], 2)
        self.assertIn("sales", meta["tables"])
        self.assertIn("employees", meta["tables"])

        conn = sqlite3.connect(meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as emp_count FROM employees")
        emp_row = cur.fetchone()
        conn.close()

        self.assertEqual(emp_row["emp_count"], 2)

    def test_dates_and_numeric_filtering(self):
        """04_dates: Verify date filtering and aggregation."""
        file_path = os.path.join(FIXTURES_DIR, "04_dates.xlsx")
        meta = self.engine.ingest_excel_file(file_path, "04_dates.xlsx", workspace_id=104)

        conn = sqlite3.connect(meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT SUM(total) as feb_total FROM sheet1 WHERE order_date >= '2026-02-01'")
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row["feb_total"], 8000 + 12000)

    def test_special_headers_and_characters_sanitization(self):
        """12_special_headers: Verify unusual headers are sanitized to safe SQL column names."""
        file_path = os.path.join(FIXTURES_DIR, "12_special_headers.xlsx")
        meta = self.engine.ingest_excel_file(file_path, "12_special_headers.xlsx", workspace_id=112)
        cols = meta["tables"]["sheet1"]["columns"]

        for col in cols:
            self.assertTrue(col.replace("_", "").isalnum(), f"Column '{col}' is not sanitized")

    def test_bangla_unicode_handling(self):
        """09_bangla: Verify Bengali text and headers are safely ingested."""
        file_path = os.path.join(FIXTURES_DIR, "09_bangla.xlsx")
        meta = self.engine.ingest_excel_file(file_path, "09_bangla.xlsx", workspace_id=109)

        conn = sqlite3.connect(meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total_rows FROM sheet1")
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row["total_rows"], 3)

    def test_large_fixture_ingestion(self):
        """15_large_fixture: Verify 100 row dataset ingests and aggregates properly."""
        file_path = os.path.join(FIXTURES_DIR, "15_large_fixture.xlsx")
        meta = self.engine.ingest_excel_file(file_path, "15_large_fixture.xlsx", workspace_id=115)
        self.assertEqual(meta["total_rows"], 100)

    def test_format_results_single_scalar_summary(self):
        """Verify _format_results formats scalar numbers with currency symbol."""
        report = self.engine._format_results(
            question="Total sales",
            filename="sales.xlsx",
            rows=[{"total_amount": 165500.0}],
            columns=["total_amount"],
            latency_ms=12.5
        )
        self.assertIn("Total Amount", report)
        self.assertIn("৳165,500.00", report)

    def test_unsupported_extension_rejected(self):
        """Security: Verify unsupported file extension raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.ingest_excel_file("/tmp/fake.docx", "fake.docx", workspace_id=999)

    def test_custom_source_id_ingestion(self):
        """Verify custom source_id is preserved and returned in metadata."""
        file_path = os.path.join(FIXTURES_DIR, "01_single_sheet.xlsx")
        meta = self.engine.ingest_excel_file(
            file_path,
            "01_single_sheet.xlsx",
            workspace_id=200,
            source_id="custom_src_abc_123"
        )
        self.assertEqual(meta["source_id"], "custom_src_abc_123")
        self.assertEqual(meta["file_id"], "custom_src_abc_123")

    def test_idempotent_reingestion_with_same_source_id(self):
        """Verify repeating ingestion with same source_id reuses database path without orphans."""
        file_path = os.path.join(FIXTURES_DIR, "01_single_sheet.xlsx")
        meta1 = self.engine.ingest_excel_file(
            file_path,
            "01_single_sheet.xlsx",
            workspace_id=201,
            source_id="idempotent_src_001"
        )
        meta2 = self.engine.ingest_excel_file(
            file_path,
            "01_single_sheet.xlsx",
            workspace_id=201,
            source_id="idempotent_src_001"
        )
        self.assertEqual(meta1["db_path"], meta2["db_path"])
        self.assertEqual(meta1["source_id"], meta2["source_id"])

    def test_csv_ingestion_with_source_id(self):
        """Verify CSV ingestion works smoothly with explicit source_id."""
        import tempfile
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tf:
            tf.write("id,item_name,price\n101,Book,250\n102,Pen,50\n")
            temp_csv_path = tf.name

        try:
            meta = self.engine.ingest_excel_file(
                temp_csv_path,
                "products.csv",
                workspace_id=202,
                source_id="csv_src_999"
            )
            self.assertEqual(meta["source_id"], "csv_src_999")
            self.assertEqual(meta["sheets_count"], 1)
            self.assertEqual(meta["total_rows"], 2)
            self.assertIn("sheet1", meta["tables"])
        finally:
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    def test_full_uuid_source_id_preserved(self):
        """Verify full 36-character UUID source_id is preserved without 32-char truncation."""
        uuid_str = "c0a80101-1234-5678-9abc-def012345678"
        file_path = os.path.join(FIXTURES_DIR, "01_single_sheet.xlsx")
        meta = self.engine.ingest_excel_file(
            file_path,
            "01_single_sheet.xlsx",
            workspace_id=300,
            source_id=uuid_str
        )
        self.assertEqual(meta["source_id"], uuid_str)
        self.assertEqual(meta["file_id"], uuid_str)
        self.assertIn(uuid_str, meta["db_path"])

    def test_cross_workspace_deletion_isolated(self):
        """Security: Verify workspace B cannot delete workspace A's registered source."""
        uuid_str = "sec-delete-test-uuid-301"
        file_path = os.path.join(FIXTURES_DIR, "01_single_sheet.xlsx")
        meta = self.engine.ingest_excel_file(
            file_path,
            "01_single_sheet.xlsx",
            workspace_id=301,
            source_id=uuid_str
        )
        # Attempt to delete from Workspace 302
        deleted_by_wrong_ws = self.engine.registry.delete_source(workspace_id=302, source_id=uuid_str)
        self.assertFalse(deleted_by_wrong_ws)

        # Confirm source still exists in Workspace 301
        source_in_301 = self.engine.registry.get_source(workspace_id=301, source_id=uuid_str)
        self.assertNotNull(source_in_301) if hasattr(self, 'assertNotNull') else self.assertIsNotNone(source_in_301)

        # Authorized deletion by Workspace 301 succeeds
        deleted_by_owner = self.engine.registry.delete_source(workspace_id=301, source_id=uuid_str)
        self.assertTrue(deleted_by_owner)

        # Source is now gone
        self.assertIsNone(self.engine.registry.get_source(workspace_id=301, source_id=uuid_str))

    def test_csv_utf8_bom_encoding_fallback(self):
        """Verify CSV with UTF-8 BOM encoding is parsed smoothly."""
        import tempfile
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tf:
            # UTF-8 BOM prefix
            content = "\ufeffsku,title,qty\nB01,Widget,10\n".encode("utf-8-sig")
            tf.write(content)
            temp_path = tf.name

        try:
            meta = self.engine.ingest_excel_file(
                temp_path,
                "bom_items.csv",
                workspace_id=303,
                source_id="bom_src_001"
            )
            self.assertEqual(meta["total_rows"], 1)
            self.assertEqual(meta["sheets_count"], 1)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main()
