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

if __name__ == '__main__':
    unittest.main()
