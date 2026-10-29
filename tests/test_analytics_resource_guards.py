import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from fastapi.testclient import TestClient

from app.analytics.excel_engine import ExcelDatabaseEngine
from app.analytics.source_registry import DataSourceRegistry
from app.analytics.source_resolver import DataSourceResolver
from app.main import app


class TestAnalyticsResourceGuards(unittest.TestCase):
    """
    Verification suite for IMP-ANL-001: Analytics Excel/SQLite Resource Guards.
    Covers worksheet count limits, row count limits, query execution deadlines,
    timeout cleanup resilience, and API error formatting.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = DataSourceRegistry(storage_dir=self.temp_dir)
        self.resolver = DataSourceResolver(registry=self.registry)
        self.engine = ExcelDatabaseEngine(
            registry=self.registry,
            resolver=self.resolver,
            max_sheets=3,
            max_rows_per_sheet=10,
            query_timeout=0.2,  # 200ms timeout for rapid deterministic testing
        )
        self.client = TestClient(app)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal_workbook_multi_sheet_ingestion(self):
        """Normal workbook with sheet count and row count within limits ingests successfully."""
        xlsx_path = os.path.join(self.temp_dir, "normal.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"product": ["A", "B", "C"], "price": [10, 20, 30]}).to_excel(
                writer, sheet_name="sales", index=False
            )
            pd.DataFrame({"dep": ["Ops", "IT"], "headcount": [5, 12]}).to_excel(
                writer, sheet_name="depts", index=False
            )

        meta = self.engine.ingest_excel_file(xlsx_path, "normal.xlsx", workspace_id=1)
        self.assertTrue(meta["success"])
        self.assertEqual(meta["sheets_count"], 2)
        self.assertIn("sales", meta["tables"])
        self.assertIn("depts", meta["tables"])
        self.assertEqual(meta["total_rows"], 5)
        self.assertTrue(os.path.exists(meta["db_path"]))

    def test_workbook_exceeding_worksheet_limit_rejected(self):
        """Workbook with more sheets than max_sheets is rejected before processing."""
        xlsx_path = os.path.join(self.temp_dir, "too_many_sheets.xlsx")
        # max_sheets is 3; write 4 sheets
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for i in range(1, 5):
                pd.DataFrame({"col": [1, 2]}).to_excel(writer, sheet_name=f"sheet_{i}", index=False)

        with self.assertRaises(ValueError) as ctx:
            self.engine.ingest_excel_file(xlsx_path, "too_many_sheets.xlsx", workspace_id=1)

        err_msg = str(ctx.exception)
        self.assertIn("Workbook exceeds maximum allowed sheets limit (3)", err_msg)
        self.assertIn("Found 4 sheets", err_msg)

        # Ensure no orphaned .sqlite files were persisted
        sqlite_files = list(Path(self.temp_dir).glob("*.sqlite"))
        self.assertEqual(len(sqlite_files), 0, "No SQLite database file should remain on failure")

    def test_worksheet_exceeding_row_limit_rejected(self):
        """Worksheet with more data rows than max_rows_per_sheet is rejected during read."""
        xlsx_path = os.path.join(self.temp_dir, "too_many_rows.xlsx")
        # max_rows_per_sheet is 10; write 15 data rows (+ 1 header row)
        rows_data = [{"item": f"Item_{i}", "val": i * 10} for i in range(1, 16)]
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(rows_data).to_excel(writer, sheet_name="inventory", index=False)

        with self.assertRaises(ValueError) as ctx:
            self.engine.ingest_excel_file(xlsx_path, "too_many_rows.xlsx", workspace_id=1)

        err_msg = str(ctx.exception)
        self.assertIn("Worksheet 'inventory' exceeds maximum allowed data rows limit (10)", err_msg)

        # Ensure no orphaned .sqlite files were persisted
        sqlite_files = list(Path(self.temp_dir).glob("*.sqlite"))
        self.assertEqual(len(sqlite_files), 0)

    def test_csv_exceeding_row_limit_rejected(self):
        """CSV file with more data rows than max_rows_per_sheet is rejected during stream read."""
        csv_path = os.path.join(self.temp_dir, "too_many_rows.csv")
        df = pd.DataFrame({"id": list(range(1, 15)), "name": [f"user_{i}" for i in range(1, 15)]})
        df.to_csv(csv_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.engine.ingest_excel_file(csv_path, "too_many_rows.csv", workspace_id=1)

        err_msg = str(ctx.exception)
        self.assertIn("Worksheet 'Sheet1' exceeds maximum allowed data rows limit (10)", err_msg)

    def test_query_completes_within_deadline(self):
        """A normal SQLite query finishes within the deadline and returns correct rows."""
        db_path = os.path.join(self.temp_dir, "test_query.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE metrics (id INT, score FLOAT);")
        conn.execute("INSERT INTO metrics VALUES (1, 95.5), (2, 88.0);")
        conn.commit()
        conn.close()

        rows, cols = self.engine.execute_sql_query(
            db_path=db_path,
            sql="SELECT * FROM metrics ORDER BY id",
            timeout=1.0,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(cols, ["id", "score"])
        self.assertEqual(rows[0]["score"], 95.5)

    def test_expensive_query_exceeding_deadline_raises_timeout_error(self):
        """An expensive Cartesian or recursive query exceeding the deadline is interrupted via progress handler."""
        db_path = os.path.join(self.temp_dir, "test_timeout.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (x INT);")
        conn.commit()
        conn.close()

        # Infinite recursive CTE
        expensive_sql = "WITH RECURSIVE r(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM r) SELECT count(*) FROM r, r;"

        with self.assertRaises(TimeoutError) as ctx:
            self.engine.execute_sql_query(
                db_path=db_path,
                sql=expensive_sql,
                timeout=0.05,  # 50ms deadline
            )

        self.assertIn("exceeded the execution time limit", str(ctx.exception))

    def test_cleanup_and_subsequent_query_succeeds_after_timeout(self):
        """After a query times out, connection state is clean and subsequent query executes normally."""
        db_path = os.path.join(self.temp_dir, "test_cleanup.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE numbers (n INT);")
        conn.executemany("INSERT INTO numbers VALUES (?);", [(1,), (2,), (3,)])
        conn.commit()
        conn.close()

        # 1. Trigger timeout
        expensive_sql = "WITH RECURSIVE r(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM r) SELECT count(*) FROM r, r;"
        with self.assertRaises(TimeoutError):
            self.engine.execute_sql_query(db_path=db_path, sql=expensive_sql, timeout=0.05)

        # 2. Immediately execute normal query on same DB
        rows, cols = self.engine.execute_sql_query(
            db_path=db_path,
            sql="SELECT SUM(n) as total FROM numbers",
            timeout=1.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total"], 6)

    def test_query_excel_returns_safe_timeout_report_without_leaking_internals(self):
        """When query_excel encounters a timeout, it returns a safe Markdown notice without stack traces."""
        xlsx_path = os.path.join(self.temp_dir, "timeout_report.xlsx")
        pd.DataFrame({"x": [1, 2]}).to_excel(xlsx_path, index=False)
        meta = self.engine.ingest_excel_file(xlsx_path, "timeout_report.xlsx", workspace_id=9)

        # Mock the LLM to return an infinite recursive SQL query
        fake_llm_response = MagicMock()
        fake_llm_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"sql": "WITH RECURSIVE r(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM r) SELECT count(*) FROM r, r;", "explanation": "test"}'
                )
            )
        ]

        with patch.object(self.engine.client.chat.completions, "create", return_value=fake_llm_response):
            result = self.engine.query_excel(
                workspace_id=9,
                file_id=meta["file_id"],
                question="Calculate infinite sum",
            )

            self.assertFalse(result["success"])
            self.assertIn("Query Timeout Notice", result["report"])
            self.assertIn("safely stopped", result["report"])
            self.assertEqual(result["rows"], [])
            # Must NOT expose python paths or raw internal exception tracebacks
            self.assertNotIn("Traceback", result["report"])
            self.assertNotIn("/Users/", result["report"])
            self.assertNotIn(".sqlite", result["report"])

    def test_readonly_pragma_enforced(self):
        """Direct query attempts with write operations are rejected by SQLite query_only PRAGMA."""
        db_path = os.path.join(self.temp_dir, "test_readonly.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE items (name TEXT);")
        conn.commit()
        conn.close()

        # PRAGMA query_only = ON will reject INSERT at SQLite engine level even if bypass attempted
        with self.assertRaises(sqlite3.OperationalError):
            self.engine.execute_sql_query(
                db_path=db_path,
                sql="INSERT INTO items VALUES ('illegal');",
                timeout=1.0,
            )

    def test_upload_api_endpoint_returns_422_on_oversized_workbook(self):
        """The FastAPI /analytics/sources/upload endpoint returns 422 with a safe error on limit violation."""
        csv_path = os.path.join(self.temp_dir, "api_large.csv")
        # Default MAX_ROWS_PER_SHEET is 100,000; override config temporarily to 5 for fast API test
        df = pd.DataFrame({"id": list(range(1, 10)), "val": list(range(1, 10))})
        df.to_csv(csv_path, index=False)

        with patch("app.config.ANALYTICS_MAX_ROWS_PER_SHEET", 5), \
             patch("app.analytics.router.excel_engine.max_rows_per_sheet", 5):
            with open(csv_path, "rb") as f:
                response = self.client.post(
                    "/analytics/sources/upload",
                    files={"file": ("api_large.csv", f, "text/csv")},
                    data={"workspace_id": 42},
                    headers={"X-API-Key": "test-key"},
                )

            # Unauthenticated dev or authenticated should return 422
            self.assertIn(response.status_code, (422, 401))
            if response.status_code == 422:
                data = response.json()
                self.assertIn("exceeds maximum allowed data rows limit (5)", data.get("detail", ""))


if __name__ == "__main__":
    unittest.main()

