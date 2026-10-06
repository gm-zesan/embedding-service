import unittest
import os
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


class TestSourceSelectionBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_storage = tempfile.mkdtemp(prefix="test_boundary_")
        cls.registry = DataSourceRegistry(storage_dir=cls.temp_storage)
        cls.resolver = DataSourceResolver(registry=cls.registry)
        cls.engine = ExcelDatabaseEngine(registry=cls.registry, resolver=cls.resolver)

        # Ingest sales.xlsx for Workspace 1:
        # Hasan = 1600, Karim = 1800
        p_sales = os.path.join(cls.temp_storage, "sales.xlsx")
        df_sales = pd.DataFrame({
            "seller": ["Hasan", "Karim"],
            "product": ["Rice", "Oil"],
            "qty": [20, 10],
            "revenue": [1600, 1800],
        })
        df_sales.to_excel(p_sales, index=False)
        cls.src_sales = cls.engine.ingest_excel_file(p_sales, "sales.xlsx", workspace_id=1)

    def test_01_ambiguous_query_without_file_indicators_returns_not_found(self):
        """Rule 5: When 1 file exists in workspace, an ambiguous fresh query ('Hasan কত sales করেছে?') returns NOT_FOUND so MySQL wins."""
        res = self.resolver.resolve(
            query="Hasan কত sales করেছে?",
            workspace_id=1,
        )
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res.source)

    def test_02_explicit_filename_resolves_to_uploaded_file(self):
        """Rule 1: Query explicitly mentioning 'sales.xlsx' resolves to uploaded sales.xlsx."""
        res = self.resolver.resolve(
            query="sales.xlsx-এ Hasan কত sales করেছে?",
            workspace_id=1,
        )
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source.original_filename, "sales.xlsx")
        self.assertEqual(res.source_id, self.src_sales["file_id"])

    def test_03_explicit_document_indicator_resolves_when_single_file(self):
        """Rule 1 & 4: Query with generic document indicator ('এই file-এ Hasan কত sales করেছে?') resolves to single file."""
        res = self.resolver.resolve(
            query="এই file-এ Hasan কত sales করেছে?",
            workspace_id=1,
        )
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source.original_filename, "sales.xlsx")

    def test_04_active_file_context_followup_resolves_to_file(self):
        """Rule 4: Followup with active context ('আর Karim-এরটা?') retains active file context."""
        ctx = DocumentAnalyticsContext(
            workspace_id=1,
            active_source_id=self.src_sales["file_id"],
            history=[
                {"role": "user", "content": "sales.xlsx-এ Hasan কত sales করেছে?"},
                {"role": "assistant", "content": "Hasan sales is 1600."},
            ]
        )
        res = self.resolver.resolve(
            query="আর Karim-এরটা?",
            workspace_id=1,
            context=ctx,
        )
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, self.src_sales["file_id"])

    def test_05_multiple_files_with_ambiguous_doc_query_returns_ambiguous(self):
        """Rule 6: When multiple files exist, generic document query returns AMBIGUOUS instead of latest file."""
        p_oct = os.path.join(self.temp_storage, "sales_october.csv")
        df_oct = pd.DataFrame({
            "seller": ["Hasan", "Karim"],
            "product": ["Rice", "Oil"],
            "revenue": [5000, 6000],
        })
        df_oct.to_csv(p_oct, index=False)
        self.engine.ingest_excel_file(p_oct, "sales_october.csv", workspace_id=1)

        res = self.resolver.resolve(
            query="এই file-এ Hasan কত sales করেছে?",
            workspace_id=1,
        )
        self.assertEqual(res.status, ResolutionStatus.AMBIGUOUS)
        self.assertGreaterEqual(len(res.candidates), 2)
        self.assertIn("একাধিক ফাইল আপলোড করা রয়েছে", res.clarification_message)


if __name__ == "__main__":
    unittest.main()
