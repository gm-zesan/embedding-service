import os
import unittest
import sqlite3
from app.analytics.excel_engine import ExcelDatabaseEngine
from app.analytics.source_registry import DataSourceRegistry
from app.analytics.source_resolver import DataSourceResolver
from app.analytics.data_source import ResolutionStatus

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "excel", "golden_multi_sheet_audit.xlsx")

class TestGoldenFixtureUploadToAnswer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace_id = 999
        cls.registry = DataSourceRegistry()
        cls.resolver = DataSourceResolver(registry=cls.registry)
        cls.engine = ExcelDatabaseEngine(registry=cls.registry, resolver=cls.resolver)

        # Ensure golden fixture exists
        if not os.path.exists(FIXTURE_PATH):
            raise FileNotFoundError(f"Fixture not found at {FIXTURE_PATH}")

        # Ingest golden workbook
        cls.meta = cls.engine.ingest_excel_file(
            file_path=FIXTURE_PATH,
            original_filename="golden_multi_sheet_audit.xlsx",
            workspace_id=cls.workspace_id,
            source_id="src_golden_e2e_verified"
        )

    def test_01_ingestion_sheets_and_empty_sheet_handling(self):
        """Verify 5 sheets are ingested and EmptySheet does not crash ingestion."""
        self.assertEqual(self.meta["source_id"], "src_golden_e2e_verified")
        self.assertIn("sales", self.meta["tables"])
        self.assertIn("customers", self.meta["tables"])
        self.assertIn("inventory", self.meta["tables"])
        self.assertIn("returns", self.meta["tables"])
        self.assertIn("notes", self.meta["tables"])

        # Empty sheet must be handled gracefully in failed_sheets or ignored from active tables
        self.assertNotIn("emptysheet", self.meta["tables"])
        failed_sheet_names = [s.get("sheet", "").lower() for s in self.meta.get("failed_sheets", [])]
        self.assertIn("emptysheet", failed_sheet_names)

    def test_02_sales_ground_truth_execution(self):
        """Verify Sales sheet: Gross=152,500; Laptop=150,000; Mouse=2,500; Qty=8; Orders=3; AOV=50,833.33."""
        conn = sqlite3.connect(self.meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Gross Revenue & Orders Count
        cur.execute("SELECT SUM(qty * unit_price) AS gross_rev, SUM(qty) AS total_qty, COUNT(*) AS total_orders FROM sales")
        r1 = cur.fetchone()
        self.assertEqual(r1["gross_rev"], 152500)
        self.assertEqual(r1["total_qty"], 8)
        self.assertEqual(r1["total_orders"], 3)

        # Laptop Revenue
        cur.execute("SELECT SUM(qty * unit_price) AS laptop_rev FROM sales WHERE LOWER(product) LIKE '%laptop%'")
        r2 = cur.fetchone()
        self.assertEqual(r2["laptop_rev"], 150000)

        # Mouse Revenue
        cur.execute("SELECT SUM(qty * unit_price) AS mouse_rev FROM sales WHERE LOWER(product) LIKE '%mouse%'")
        r3 = cur.fetchone()
        self.assertEqual(r3["mouse_rev"], 2500)

        # Average Order Value
        cur.execute("SELECT ROUND(AVG(qty * unit_price), 2) AS avg_order_val FROM sales")
        r4 = cur.fetchone()
        self.assertEqual(r4["avg_order_val"], 50833.33)

        conn.close()

        # Verify presentation report
        report = self.engine._format_results(
            question="Average Order Value",
            filename="golden_multi_sheet_audit.xlsx",
            rows=[{"avg_order_val": 50833.33}],
            columns=["avg_order_val"],
            latency_ms=0.5
        )
        self.assertIn("৳50,833.33", report)

    def test_03_customers_ground_truth_execution(self):
        """Verify Customers sheet: Total Due=28,200; Dhaka Due=20,500; Top Due=Rahim Ahmed (12,000); Zero Due=Fatima Begum (0)."""
        conn = sqlite3.connect(self.meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT SUM(due_amount) AS total_due FROM customers")
        self.assertEqual(cur.fetchone()["total_due"], 28200)

        cur.execute("SELECT SUM(due_amount) AS dhaka_due FROM customers WHERE LOWER(city) = 'dhaka'")
        self.assertEqual(cur.fetchone()["dhaka_due"], 20500)

        cur.execute("SELECT name, due_amount FROM customers ORDER BY due_amount DESC LIMIT 1")
        top_due = cur.fetchone()
        self.assertEqual(top_due["name"], "Rahim Ahmed")
        self.assertEqual(top_due["due_amount"], 12000)

        cur.execute("SELECT name FROM customers WHERE due_amount = 0")
        zero_due = cur.fetchone()
        self.assertEqual(zero_due["name"], "Fatima Begum")

        conn.close()

    def test_04_inventory_ground_truth_execution(self):
        """Verify Inventory sheet: Total Stock=160; Cost Valuation=1,056,000."""
        conn = sqlite3.connect(self.meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT SUM(stock) AS total_stock, SUM(stock * cost) AS total_cost_valuation FROM inventory")
        row = cur.fetchone()
        self.assertEqual(row["total_stock"], 160)
        self.assertEqual(row["total_cost_valuation"], 1056000)

        conn.close()

    def test_05_returns_ground_truth_execution(self):
        """Verify Returns sheet: 1 Approved Return totaling 500 BDT."""
        conn = sqlite3.connect(self.meta["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS approved_count, SUM(amount) AS approved_amount FROM returns WHERE status = 'Approved'")
        row = cur.fetchone()
        self.assertEqual(row["approved_count"], 1)
        self.assertEqual(row["approved_amount"], 500)

        conn.close()

    def test_06_independent_sheet_discovery(self):
        """Verify separate discovery of Sales vs Customers sheets without cross-pollution."""
        res_sales = self.resolver.resolve("Sales sheet এর মোট বিক্রি কত?", self.workspace_id, explicit_source_id="src_golden_e2e_verified")
        self.assertEqual([d.dataset_id for d in res_sales.selected_datasets], ["sales"])

        res_cust = self.resolver.resolve("Customers sheet এর মোট বকেয়া কত?", self.workspace_id, explicit_source_id="src_golden_e2e_verified")
        self.assertEqual([d.dataset_id for d in res_cust.selected_datasets], ["customers"])

    def test_07_unsupported_cross_workbook_join_defense(self):
        """Verify attempting to join across two separate workbooks returns UNSUPPORTED_CROSS_SOURCE."""
        # Register a second distinct workbook
        second_src = self.registry.get_source(self.workspace_id, "src_golden_e2e_verified").model_copy(update={
            "source_id": "src_payroll_distinct_02",
            "original_filename": "payroll_2026.xlsx"
        })
        self.registry.register_source(second_src)

        res = self.resolver.resolve(
            query="golden_multi_sheet_audit.xlsx এবং payroll_2026.xlsx তুলনা করো",
            workspace_id=self.workspace_id
        )
        self.assertEqual(res.status, ResolutionStatus.UNSUPPORTED_CROSS_SOURCE)
        self.assertIn("Cross-source comparison across separate files", res.reason)

if __name__ == "__main__":
    unittest.main()
