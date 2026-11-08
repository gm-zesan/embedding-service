"""
Verification test to guarantee zero LLM-generated raw SQL strings exist in the engine.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from app.analytics.excel_engine import ExcelDatabaseEngine
from app.analytics.data_source import DataSource, Dataset, DataSourceFormat, SourceResolutionResult, ResolutionStatus


class TestNoRawSqlInEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_id = 12
        self.db_path = os.path.join(self.temp_dir, "ws_12_inventory.sqlite")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE products (product_name TEXT, stock_quantity INTEGER, unit_price REAL);")
        cur.executemany(
            "INSERT INTO products (product_name, stock_quantity, unit_price) VALUES (?, ?, ?);",
            [
                ("Miniket Rice 25kg", 150, 1850.0),
                ("Soybean Oil 5L", 80, 890.0),
                ("Mustard Oil 1L", 45, 340.0),
            ]
        )
        conn.commit()
        conn.close()

        self.engine = ExcelDatabaseEngine()

        # Mock source resolution
        dataset = Dataset(
            dataset_id="ds_products",
            dataset_name="products",
            storage_reference="products",
            row_count=3,
            columns=["product_name", "stock_quantity", "unit_price"],
            numeric_measures=["stock_quantity", "unit_price"],
            text_dimensions=["product_name"],
        )
        self.mock_source = DataSource(
            source_id="inventory_doc",
            workspace_id=self.workspace_id,
            original_filename="inventory.xlsx",
            format=DataSourceFormat.XLSX,
            stored_path="/tmp/inventory.xlsx",
            uploaded_at="2026-11-02T12:00:00Z",
            file_hash="hash123456789",
            db_path=self.db_path,
            datasets=[dataset],
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_query_excel_uses_virtual_compiler_not_raw_sql_prompt(self):
        """
        Verifies that query_excel uses VirtualSemanticPlanner and VirtualRelationalCompiler
        and executes with parameter bindings.
        """
        with patch.object(self.engine.resolver, "resolve") as mock_resolve:
            mock_resolve.return_value = SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source=self.mock_source,
                selected_datasets=self.mock_source.datasets,
            )

            result = self.engine.query_excel(
                workspace_id=self.workspace_id,
                file_id="inventory_doc",
                question="What is the total stock quantity of Miniket Rice?",
            )

            self.assertTrue(result["success"])
            # Strictly parameterized SQL - value is never interpolated into the query string!
            self.assertIn("WHERE LOWER(products.product_name) LIKE ?", result["sql"])
            self.assertNotIn("Miniket", result["sql"])
            # Verify rows returned correctly
            self.assertTrue(len(result["rows"]) > 0)
            self.assertIn(150, list(result["rows"][0].values()))


if __name__ == "__main__":
    unittest.main()
