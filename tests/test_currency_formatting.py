import unittest
from app.analytics.excel_engine import ExcelDatabaseEngine

class TestCurrencyFormatting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ExcelDatabaseEngine()

    def test_avg_order_val_formats_as_bdt(self):
        """avg_order_val = 50833.33 -> BDT currency formatting"""
        report = self.engine._format_results(
            question="What is the average order value?",
            filename="sales.xlsx",
            rows=[{"avg_order_val": 50833.33}],
            columns=["avg_order_val"],
            latency_ms=1.2,
        )
        self.assertIn("Avg Order Val", report)
        self.assertIn("৳50,833.33", report)

    def test_total_revenue_formats_as_bdt(self):
        """total_revenue = 152500 -> BDT currency formatting"""
        report = self.engine._format_results(
            question="What is total revenue?",
            filename="sales.xlsx",
            rows=[{"total_revenue": 152500}],
            columns=["total_revenue"],
            latency_ms=1.0,
        )
        self.assertIn("Total Revenue", report)
        self.assertIn("৳152,500.00", report)

    def test_total_quantity_is_not_currency(self):
        """total_quantity = 8 -> not currency"""
        report = self.engine._format_results(
            question="How many items were sold?",
            filename="sales.xlsx",
            rows=[{"total_quantity": 8}],
            columns=["total_quantity"],
            latency_ms=0.8,
        )
        self.assertIn("Total Quantity", report)
        self.assertIn("- **Total Quantity:** 8", report)
        self.assertNotIn("৳", report)

    def test_customer_count_is_not_currency(self):
        """customer_count = 5 -> not currency"""
        report = self.engine._format_results(
            question="How many customers are there?",
            filename="customers.xlsx",
            rows=[{"customer_count": 5}],
            columns=["customer_count"],
            latency_ms=0.5,
        )
        self.assertIn("Customer Count", report)
        self.assertIn("- **Customer Count:** 5", report)
        self.assertNotIn("৳", report)

    def test_average_rating_is_not_currency(self):
        """average_rating = 4.5 -> not currency unless explicitly defined as monetary"""
        report = self.engine._format_results(
            question="What is the average product rating?",
            filename="products.xlsx",
            rows=[{"average_rating": 4.5}],
            columns=["average_rating"],
            latency_ms=0.9,
        )
        self.assertIn("Average Rating", report)
        self.assertIn("- **Average Rating:** 4.50", report)
        self.assertNotIn("৳", report)

    def test_non_monetary_value_columns_are_not_currency(self):
        """Ensure columns with 'val' or 'value' (e.g. p_val, return_val, node_value) are not treated as currency"""
        for col_name in ("p_val", "return_val", "node_value", "attr_val", "string_value"):
            self.assertFalse(
                self.engine._is_monetary_column(col_name),
                f"Column '{col_name}' was incorrectly classified as currency"
            )

    def test_existing_monetary_aliases_classified_correctly(self):
        """Verify standard monetary aliases remain correctly classified"""
        monetary_aliases = [
            "gross_revenue", "laptop_revenue", "mouse_revenue", "total_due",
            "due_amount", "dhaka_due", "unit_price", "cost", "total_sales",
            "approved_amount", "total_cost_valuation", "cart_val", "order_value"
        ]
        for col_name in monetary_aliases:
            self.assertTrue(
                self.engine._is_monetary_column(col_name),
                f"Column '{col_name}' was NOT recognized as currency"
            )

    def test_tabular_multiline_currency_and_integer_formatting(self):
        """Verify tabular formatting distinguishes currency vs quantity across multiple rows"""
        rows = [
            {"product": "Laptop", "total_sales": 150000, "qty": 3},
            {"product": "Mouse", "total_sales": 2500, "qty": 5},
        ]
        report = self.engine._format_results(
            question="Sales by product",
            filename="sales.xlsx",
            rows=rows,
            columns=["product", "total_sales", "qty"],
            latency_ms=2.1,
        )
        self.assertIn("৳150,000", report)
        self.assertIn("৳2,500", report)
        self.assertIn("| Laptop | ৳150,000 | 3 |", report)
        self.assertIn("| Mouse | ৳2,500 | 5 |", report)

    def test_profit_and_margin_are_classified_as_monetary(self):
        """profit, net_profit, margin, margin_amount, gross_margin_amount, product_worth -> currency"""
        for col in ("profit", "net_profit", "gross_profit", "margin", "margin_amount", "gross_margin_amount", "product_worth"):
            self.assertTrue(
                self.engine._is_monetary_column(col),
                f"Column '{col}' was NOT recognized as currency"
            )

    def test_rates_are_not_classified_as_monetary(self):
        """tax_rate, vat_rate, discount_rate, interest_rate, exchange_rate -> NOT currency"""
        for col in ("tax_rate", "vat_rate", "discount_rate", "interest_rate", "conversion_rate", "exchange_rate"):
            self.assertFalse(
                self.engine._is_monetary_column(col),
                f"Column '{col}' was incorrectly classified as currency"
            )

    def test_dynamic_currency_symbols(self):
        """unit_price_usd -> $, balance_eur -> €, total_revenue -> ৳"""
        report_usd = self.engine._format_results(
            question="What is the price?",
            filename="export.xlsx",
            rows=[{"unit_price_usd": 25.0}],
            columns=["unit_price_usd"],
            latency_ms=1.0,
        )
        self.assertIn("$25.00", report_usd)
        self.assertNotIn("৳", report_usd)

        report_eur = self.engine._format_results(
            question="What is the balance?",
            filename="export.xlsx",
            rows=[{"balance_eur": 150.0}],
            columns=["balance_eur"],
            latency_ms=1.0,
        )
        self.assertIn("€150.00", report_eur)
        self.assertNotIn("৳", report_eur)

    def test_formatter_is_unified_with_excel_engine(self):
        """Verify AnalyticsFormatter shares the exact classification with ExcelDatabaseEngine"""
        from app.analytics.formatter import AnalyticsFormatter
        test_fields = [
            ("profit", True),
            ("margin", True),
            ("avg_order_val", True),
            ("tax_rate", False),
            ("total_quantity", False),
            ("unit_price_usd", True),
        ]
        for field, expected in test_fields:
            self.assertEqual(
                AnalyticsFormatter.is_monetary_field(field),
                expected,
                f"AnalyticsFormatter.is_monetary_field('{field}') mismatch"
            )
            self.assertEqual(
                self.engine._is_monetary_column(field),
                expected,
                f"ExcelDatabaseEngine._is_monetary_column('{field}') mismatch"
            )

    def test_margin_percentage_vs_monetary_amount(self):
        """Verify profit_margin (ratio/percentage) is NOT monetary, while margin_amount is monetary"""
        # Non-monetary percentage/ratio metrics
        for col in ("profit_margin", "gross_margin", "net_margin", "operating_margin", "margin_pct", "margin_rate", "margin_ratio"):
            self.assertFalse(
                self.engine._is_monetary_column(col),
                f"Column '{col}' should NOT be treated as currency"
            )

        # Monetary amount metrics
        for col in ("margin_amount", "gross_margin_amount", "profit_margin_amount", "margin_val", "margin_bdt"):
            self.assertTrue(
                self.engine._is_monetary_column(col),
                f"Column '{col}' should be treated as currency"
            )

    def test_currency_symbol_metadata_override(self):
        """Verify currency_hint (from workspace/file metadata) overrides column names"""
        from app.analytics.formatter import AnalyticsFormatter
        # When metadata says USD, even generic 'total_revenue' gets '$'
        self.assertEqual(
            ExcelDatabaseEngine._get_currency_symbol("total_revenue", default="৳", currency_hint="USD"),
            "$"
        )
        self.assertEqual(
            AnalyticsFormatter.get_currency_symbol("total_revenue", default="৳", currency_hint="EUR"),
            "€"
        )
        # Fallback to BDT when no hint or hint is BDT
        self.assertEqual(
            ExcelDatabaseEngine._get_currency_symbol("total_revenue", default="৳", currency_hint=None),
            "৳"
        )

if __name__ == "__main__":
    unittest.main()

