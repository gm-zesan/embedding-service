"""
Comprehensive Phase 2A Test Suite:
Deterministic Dataset Classification & Automatic Source Resolution

Tests:
1. Product catalog vs sales transaction classification.
2. Orders vs inventory stock classification.
3. Customer directory vs payment logs classification.
4. Bengali & Banglish schema token classification.
5. Ambiguous & low-confidence schema handling (UNKNOWN / OTHER).
6. Multi-sheet workbook with distinct dataset categories.
7. Automatic natural-language intent resolution (English).
8. Automatic natural-language intent resolution (Bengali & Banglish).
9. Multiple files sharing same business category -> AMBIGUOUS resolution.
10. Explicit source reference overriding category ambiguity.
11. Conversational context follow-up resolution.
12. Schema incompatibility detection -> INSUFFICIENT_SCHEMA.
13. Multi-category comparison -> UNSUPPORTED_CROSS_SOURCE.
14. Strict workspace isolation (no cross-tenant leakage).
15. Deleted / inactive source exclusion.
16. Legacy registry backward compatibility (schema hydration).
17. Plain entity queries fallback for MySQL handling (Rule 5).
"""

import os
import json
import shutil
import tempfile
import unittest
import pandas as pd

from app.analytics.data_source import (
    DataSource,
    Dataset,
    DataSourceFormat,
    DatasetCategory,
    DatasetClassification,
    DocumentAnalyticsContext,
    ResolutionStatus,
    SourceResolutionResult,
)
from app.analytics.dataset_classifier import dataset_classifier, DatasetClassifier
from app.analytics.source_registry import DataSourceRegistry
from app.analytics.source_resolver import DataSourceResolver
from app.analytics.excel_engine import ExcelDatabaseEngine


class TestDatasetClassificationAndResolution(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="phase2a_test_")
        self.storage_dir = os.path.join(self.test_dir, "registry")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.registry = DataSourceRegistry(storage_dir=self.storage_dir)
        self.resolver = DataSourceResolver(registry=self.registry)
        self.engine = ExcelDatabaseEngine(registry=self.registry, resolver=self.resolver)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Dataset Classification Unit Tests
    # -------------------------------------------------------------------------

    def test_product_catalog_vs_sales_classification(self):
        """Distinguish master product catalog from sales transaction log."""
        # Product catalog
        prod_cols = ["product_id", "product_name", "category", "unit_price", "cost_price", "description"]
        prod_types = {"product_id": "INTEGER", "product_name": "TEXT", "unit_price": "REAL"}
        clf_prod = dataset_classifier.classify_dataset(
            columns=prod_cols,
            inferred_types=prod_types,
            sheet_name="Products",
            filename="catalog_2024.xlsx",
        )
        self.assertEqual(clf_prod.category, DatasetCategory.PRODUCTS)
        self.assertGreaterEqual(clf_prod.confidence, 0.70)
        self.assertEqual(clf_prod.status, "classified")

        # Sales transactions (contains product_name and unit_price, but also invoice, order_date, customer)
        sales_cols = ["invoice_no", "order_date", "customer_name", "product_name", "unit_price", "quantity", "total_amount", "payment_method"]
        sales_types = {"invoice_no": "TEXT", "order_date": "DATE", "quantity": "INTEGER", "total_amount": "REAL"}
        clf_sales = dataset_classifier.classify_dataset(
            columns=sales_cols,
            inferred_types=sales_types,
            sheet_name="DailySales",
            filename="sales_q3.xlsx",
        )
        self.assertEqual(clf_sales.category, DatasetCategory.SALES)
        self.assertGreaterEqual(clf_sales.confidence, 0.70)

    def test_orders_vs_inventory_classification(self):
        """Distinguish orders/shipments from stock/inventory snapshots."""
        # Orders / Shipments
        order_cols = ["order_id", "customer_id", "order_status", "shipping_address", "courier_partner", "tracking_code", "cod_amount"]
        clf_order = dataset_classifier.classify_dataset(columns=order_cols, sheet_name="Orders", filename="courier_orders.csv")
        self.assertEqual(clf_order.category, DatasetCategory.ORDERS)
        self.assertGreaterEqual(clf_order.confidence, 0.65)

        # Inventory / Stock
        inv_cols = ["sku", "product_name", "current_stock", "reorder_level", "safety_stock", "warehouse_location", "supplier_name"]
        clf_inv = dataset_classifier.classify_dataset(columns=inv_cols, sheet_name="Stock", filename="warehouse_inventory.xlsx")
        self.assertEqual(clf_inv.category, DatasetCategory.INVENTORY)
        self.assertGreaterEqual(clf_inv.confidence, 0.70)

    def test_customers_and_payments_classification(self):
        """Classify customer directories and payment transaction records."""
        cust_cols = ["customer_id", "customer_name", "phone_number", "email_address", "district", "total_purchases"]
        clf_cust = dataset_classifier.classify_dataset(columns=cust_cols, sheet_name="Clients")
        self.assertEqual(clf_cust.category, DatasetCategory.CUSTOMERS)

        pay_cols = ["transaction_id", "payment_method", "amount", "paid_status", "gateway_fee", "payment_date"]
        clf_pay = dataset_classifier.classify_dataset(columns=pay_cols, sheet_name="Transactions")
        self.assertEqual(clf_pay.category, DatasetCategory.PAYMENTS)

    def test_bengali_banglish_columns_classification(self):
        """Classify datasets with Bengali script and Banglish transliterated column names."""
        # Bengali script products
        bn_prod_cols = ["পণ্যের_নাম", "মূল্য", "স্টক_পরিমাণ", "ক্যাটাগরি"]
        clf_bn = dataset_classifier.classify_dataset(columns=bn_prod_cols)
        self.assertEqual(clf_bn.category, DatasetCategory.PRODUCTS)

        # Banglish customer directory
        banglish_cust_cols = ["customer_er_naam", "phone_number", "thikana", "mot_kroy"]
        clf_bg = dataset_classifier.classify_dataset(columns=banglish_cust_cols)
        self.assertEqual(clf_bg.category, DatasetCategory.CUSTOMERS)

    def test_ambiguous_and_low_confidence_classification(self):
        """Opaque column names must fall back to UNKNOWN or OTHER rather than guessing."""
        opaque_cols = ["col_a", "col_b", "val_x", "ref_num"]
        clf = dataset_classifier.classify_dataset(columns=opaque_cols)
        self.assertEqual(clf.category, DatasetCategory.UNKNOWN)
        self.assertEqual(clf.status, "unclassified")
        self.assertLess(clf.confidence, 0.35)

    # -------------------------------------------------------------------------
    # 2. Multi-Sheet Ingestion & Descriptors
    # -------------------------------------------------------------------------

    def test_multi_sheet_workbook_descriptors(self):
        """An Excel workbook with multiple sheets produces distinct categorized datasets."""
        wb_path = os.path.join(self.test_dir, "company_data.xlsx")
        with pd.ExcelWriter(wb_path, engine="openpyxl") as writer:
            pd.DataFrame({
                "sku": ["SKU-01", "SKU-02"],
                "product_name": ["Wireless Mouse", "Mechanical Keyboard"],
                "unit_price": [1200, 3500],
                "category": ["Accessories", "Accessories"],
            }).to_excel(writer, sheet_name="ProductCatalog", index=False)

            pd.DataFrame({
                "invoice_id": ["INV-101", "INV-102"],
                "customer_name": ["Rahim", "Karim"],
                "total_amount": [1200, 3500],
                "sale_date": ["2024-01-10", "2024-01-11"],
            }).to_excel(writer, sheet_name="SalesRecords", index=False)

            pd.DataFrame({
                "sku": ["SKU-01", "SKU-02"],
                "product_name": ["Wireless Mouse", "Mechanical Keyboard"],
                "current_stock": [50, 20],
                "warehouse_bin": ["A1", "B2"],
            }).to_excel(writer, sheet_name="WarehouseStock", index=False)

        meta = self.engine.ingest_excel_file(wb_path, "company_data.xlsx", workspace_id=101)
        source = self.registry.get_source(101, meta["source_id"])
        self.assertIsNotNone(source)
        self.assertEqual(len(source.datasets), 3)

        # Verify dataset categories
        cats = {d.sheet_name: d.classification.category for d in source.datasets}
        self.assertEqual(cats["ProductCatalog"], DatasetCategory.PRODUCTS)
        self.assertEqual(cats["SalesRecords"], DatasetCategory.SALES)
        self.assertEqual(cats["WarehouseStock"], DatasetCategory.INVENTORY)

    # -------------------------------------------------------------------------
    # 3. Automatic Source & Dataset Resolution
    # -------------------------------------------------------------------------

    def test_automatic_resolution_by_business_intent_english(self):
        """Natural-language questions automatically resolve to matching business category."""
        # Ingest products file
        p_prod = os.path.join(self.test_dir, "products.csv")
        pd.DataFrame({
            "product_id": [1, 2],
            "product_name": ["Smartphone", "Tablet"],
            "unit_price": [25000, 18000],
        }).to_csv(p_prod, index=False)
        src_prod = self.engine.ingest_excel_file(p_prod, "products.csv", workspace_id=1)

        # Ingest sales file
        p_sales = os.path.join(self.test_dir, "sales.csv")
        pd.DataFrame({
            "invoice_id": [1001, 1002],
            "sale_date": ["2024-02-01", "2024-02-02"],
            "total_revenue": [50000, 36000],
            "salesperson": ["Jamal", "Kamal"],
        }).to_csv(p_sales, index=False)
        src_sales = self.engine.ingest_excel_file(p_sales, "sales.csv", workspace_id=1)

        # Query 1: Product intent
        res_prod = self.resolver.resolve("Which products have the highest price?", workspace_id=1)
        self.assertEqual(res_prod.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res_prod.source_id, src_prod["source_id"])
        self.assertEqual(res_prod.selected_datasets[0].classification.category, DatasetCategory.PRODUCTS)

        # Query 2: Sales intent
        res_sales = self.resolver.resolve("What was our total revenue last month?", workspace_id=1)
        self.assertEqual(res_sales.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res_sales.source_id, src_sales["source_id"])
        self.assertEqual(res_sales.selected_datasets[0].classification.category, DatasetCategory.SALES)

    def test_automatic_resolution_by_intent_bengali_banglish(self):
        """Bengali and Banglish questions correctly resolve to target datasets."""
        p_prod = os.path.join(self.test_dir, "catalog.csv")
        pd.DataFrame({
            "item_name": ["Shirt", "Pants"],
            "unit_price": [1500, 2200],
            "category": ["Clothing", "Clothing"],
        }).to_csv(p_prod, index=False)
        src_prod = self.engine.ingest_excel_file(p_prod, "catalog.csv", workspace_id=2)

        # Bengali script question
        res_bn = self.resolver.resolve("সবচেয়ে দামি পণ্য কোনটি?", workspace_id=2)
        self.assertEqual(res_bn.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res_bn.source_id, src_prod["source_id"])

        # Banglish question
        res_bg = self.resolver.resolve("shobcheye beshi dam kon item er?", workspace_id=2)
        self.assertEqual(res_bg.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res_bg.source_id, src_prod["source_id"])

    def test_multiple_files_same_category_returns_ambiguous(self):
        """When multiple files contain the same category, resolution returns AMBIGUOUS."""
        p_s1 = os.path.join(self.test_dir, "sales_2023.csv")
        pd.DataFrame({"invoice_id": [1], "order_date": ["2023-01-01"], "amount": [5000]}).to_csv(p_s1, index=False)
        self.engine.ingest_excel_file(p_s1, "sales_2023.csv", workspace_id=3)

        p_s2 = os.path.join(self.test_dir, "sales_2024.csv")
        pd.DataFrame({"invoice_id": [2], "order_date": ["2024-01-01"], "amount": [8000]}).to_csv(p_s2, index=False)
        self.engine.ingest_excel_file(p_s2, "sales_2024.csv", workspace_id=3)

        # Generic sales query with 2 competing sales datasets
        res = self.resolver.resolve("What were total sales?", workspace_id=3)
        self.assertEqual(res.status, ResolutionStatus.AMBIGUOUS)
        self.assertGreaterEqual(len(res.candidates), 2)
        self.assertIn("Ambiguous", res.clarification_message)

    def test_explicit_source_reference_overrides_category_ambiguity(self):
        """Explicit mention of filename resolves cleanly despite multiple category files."""
        p_s1 = os.path.join(self.test_dir, "sales_2023.csv")
        pd.DataFrame({"invoice_id": [1], "amount": [5000]}).to_csv(p_s1, index=False)
        self.engine.ingest_excel_file(p_s1, "sales_2023.csv", workspace_id=4)

        p_s2 = os.path.join(self.test_dir, "sales_2024.csv")
        pd.DataFrame({"invoice_id": [2], "amount": [8000]}).to_csv(p_s2, index=False)
        s2 = self.engine.ingest_excel_file(p_s2, "sales_2024.csv", workspace_id=4)

        res = self.resolver.resolve("sales_2024.csv এ total sales কত?", workspace_id=4)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, s2["source_id"])

    def test_conversation_context_followup_resolution(self):
        """Follow-up questions retain active conversational dataset."""
        p_prod = os.path.join(self.test_dir, "products.csv")
        pd.DataFrame({"product_name": ["A", "B"], "price": [10, 20]}).to_csv(p_prod, index=False)
        src = self.engine.ingest_excel_file(p_prod, "products.csv", workspace_id=5)

        ctx = DocumentAnalyticsContext(
            workspace_id=5,
            active_source_id=src["source_id"],
            history=[
                {"role": "user", "content": "What is the price of A?"},
                {"role": "assistant", "content": "Price of A is 10."},
            ]
        )
        res = self.resolver.resolve("What about B?", workspace_id=5, context=ctx)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    # -------------------------------------------------------------------------
    # 4. Safety, Boundary & Incompatibility Handling
    # -------------------------------------------------------------------------

    def test_missing_incompatible_schema_fields(self):
        """When query requires a metric missing from dataset, return INSUFFICIENT_SCHEMA."""
        p_sales = os.path.join(self.test_dir, "sales_simple.csv")
        pd.DataFrame({
            "order_id": [101, 102],
            "customer_name": ["Rahim", "Karim"],
            "total_revenue": [5000, 7000],
        }).to_csv(p_sales, index=False)
        self.engine.ingest_excel_file(p_sales, "sales_simple.csv", workspace_id=6)

        # Query asks for profit, but dataset only has revenue
        res = self.resolver.resolve("What was the net profit for Rahim?", workspace_id=6)
        self.assertEqual(res.status, ResolutionStatus.INSUFFICIENT_SCHEMA)
        self.assertIn("profit", res.reason.lower())

    def test_multi_category_comparison_returns_unsupported(self):
        """Cross-category queries spanning multiple domains return UNSUPPORTED_CROSS_SOURCE."""
        # Catalog
        p_prod = os.path.join(self.test_dir, "catalog.csv")
        pd.DataFrame({"product_id": [1], "price": [100]}).to_csv(p_prod, index=False)
        self.engine.ingest_excel_file(p_prod, "catalog.csv", workspace_id=7)

        # Sales
        p_sales = os.path.join(self.test_dir, "sales.csv")
        pd.DataFrame({"invoice_id": [10], "amount": [500]}).to_csv(p_sales, index=False)
        self.engine.ingest_excel_file(p_sales, "sales.csv", workspace_id=7)

        # Multi-category comparison
        res = self.resolver.resolve("Compare product prices with sales performance", workspace_id=7)
        self.assertEqual(res.status, ResolutionStatus.UNSUPPORTED_CROSS_SOURCE)
        self.assertIn("Cross-dataset", res.reason)

    def test_cross_workspace_dataset_isolation(self):
        """Datasets in workspace 10 must never be visible to workspace 20."""
        p_ws10 = os.path.join(self.test_dir, "confidential_ws10.csv")
        pd.DataFrame({"product_name": ["Secret Device"], "price": [99999]}).to_csv(p_ws10, index=False)
        src_ws10 = self.engine.ingest_excel_file(p_ws10, "confidential_ws10.csv", workspace_id=10)

        # Query from workspace 20
        res_ws20 = self.resolver.resolve("Which products have highest price?", workspace_id=20)
        self.assertEqual(res_ws20.status, ResolutionStatus.NOT_FOUND)

        # Explicit source_id probe from workspace 20
        res_probe = self.resolver.resolve("Show products", workspace_id=20, explicit_source_id=src_ws10["source_id"])
        self.assertEqual(res_probe.status, ResolutionStatus.NOT_FOUND)

    def test_deleted_source_excluded_from_resolution(self):
        """Deleting a source removes its datasets from automatic resolution."""
        p = os.path.join(self.test_dir, "temp_sales.csv")
        pd.DataFrame({"invoice_id": [1], "amount": [100]}).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "temp_sales.csv", workspace_id=8)

        # Verify it resolves prior to deletion
        res_before = self.resolver.resolve("Total sales", workspace_id=8)
        self.assertEqual(res_before.status, ResolutionStatus.RESOLVED)

        # Delete source
        self.engine.registry.delete_source(workspace_id=8, source_id=src["source_id"])

        # Subsequent resolution returns NOT_FOUND
        res_after = self.resolver.resolve("Total sales", workspace_id=8)
        self.assertEqual(res_after.status, ResolutionStatus.NOT_FOUND)

    def test_legacy_registry_backward_compatibility(self):
        """Legacy registry JSON missing 'classification' is safely hydrated on load."""
        legacy_data = {
            "source_id": "legacy_src_001",
            "workspace_id": 99,
            "original_filename": "legacy_products.xlsx",
            "file_size": 1024,
            "format": "xlsx",
            "uploaded_at": "2024-01-01T00:00:00Z",
            "sqlite_db_path": "/tmp/fake.db",
            "datasets": [
                {
                    "dataset_id": "legacy_src_001:Products",
                    "dataset_name": "Products",
                    "sheet_name": "Products",
                    "table_name": "products_t",
                    "columns": ["sku", "product_name", "unit_price", "category"],
                    "inferred_types": {"sku": "TEXT", "unit_price": "REAL"},
                    "row_count": 10,
                }
            ]
        }
        json_path = os.path.join(self.storage_dir, "ws_99_legacy_src_001.json")
        with open(json_path, "w") as f:
            json.dump(legacy_data, f)

        # Load through registry
        loaded_src = self.registry.get_source(99, "legacy_src_001")
        self.assertIsNotNone(loaded_src)
        self.assertEqual(len(loaded_src.datasets), 1)

        d = loaded_src.datasets[0]
        self.assertIsNotNone(d.classification)
        self.assertEqual(d.classification.category, DatasetCategory.PRODUCTS)
        self.assertEqual(d.classification.method, "deterministic_heuristic_v2")

    def test_plain_entity_query_fallback_for_mysql(self):
        """Plain entity queries without file indicators return NOT_FOUND so MySQL handles them (Rule 5)."""
        p_sales = os.path.join(self.test_dir, "sales.csv")
        pd.DataFrame({"seller": ["Hasan"], "sales": [1000]}).to_csv(p_sales, index=False)
        self.engine.ingest_excel_file(p_sales, "sales.csv", workspace_id=12)

        res = self.resolver.resolve("Hasan কত sales করেছে?", workspace_id=12)
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res.source)


if __name__ == "__main__":
    unittest.main()
