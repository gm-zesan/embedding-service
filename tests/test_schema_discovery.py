import unittest
from unittest.mock import MagicMock, patch
import time
from app.analytics.schema_discovery import AutoCatalogingEngine

class TestSchemaDiscovery(unittest.TestCase):
    def setUp(self):
        self.engine = AutoCatalogingEngine(cache_ttl_seconds=60)

    def test_sensitive_columns_excluded(self):
        """Verifies passwords, tokens, and secrets are excluded from discovered dimensions and prompt catalog."""
        mock_cols = [
            {"TABLE_NAME": "users", "COLUMN_NAME": "id", "DATA_TYPE": "bigint", "COLUMN_TYPE": "bigint(20)", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "EXTRA": "auto_increment"},
            {"TABLE_NAME": "users", "COLUMN_NAME": "name", "DATA_TYPE": "varchar", "COLUMN_TYPE": "varchar(255)", "IS_NULLABLE": "NO", "COLUMN_KEY": "", "EXTRA": ""},
            {"TABLE_NAME": "users", "COLUMN_NAME": "email", "DATA_TYPE": "varchar", "COLUMN_TYPE": "varchar(255)", "IS_NULLABLE": "NO", "COLUMN_KEY": "UNI", "EXTRA": ""},
            {"TABLE_NAME": "users", "COLUMN_NAME": "password", "DATA_TYPE": "varchar", "COLUMN_TYPE": "varchar(255)", "IS_NULLABLE": "NO", "COLUMN_KEY": "", "EXTRA": ""},
            {"TABLE_NAME": "users", "COLUMN_NAME": "remember_token", "DATA_TYPE": "varchar", "COLUMN_TYPE": "varchar(100)", "IS_NULLABLE": "YES", "COLUMN_KEY": "", "EXTRA": ""},
            {"TABLE_NAME": "channel_accounts", "COLUMN_NAME": "id", "DATA_TYPE": "bigint", "COLUMN_TYPE": "bigint(20)", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "EXTRA": ""},
            {"TABLE_NAME": "channel_accounts", "COLUMN_NAME": "access_token", "DATA_TYPE": "text", "COLUMN_TYPE": "text", "IS_NULLABLE": "YES", "COLUMN_KEY": "", "EXTRA": ""},
            {"TABLE_NAME": "channel_accounts", "COLUMN_NAME": "refresh_token", "DATA_TYPE": "text", "COLUMN_TYPE": "text", "IS_NULLABLE": "YES", "COLUMN_KEY": "", "EXTRA": ""},
        ]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [mock_cols, []] # cols, fks
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(self.engine, '_get_connection', return_value=mock_conn):
            catalog = self.engine.discover_and_catalog(force_refresh=True)

        # Assert tables discovered
        self.assertIn("users", catalog["tables"])
        self.assertIn("channel_accounts", catalog["tables"])

        # Assert dimensions include name, email, id
        user_dims = catalog["tables"]["users"]["dimensions"]
        self.assertIn("name", user_dims)
        self.assertIn("email", user_dims)
        self.assertIn("id", user_dims)

        # Assert sensitive columns are strictly excluded
        self.assertNotIn("password", user_dims)
        self.assertNotIn("remember_token", user_dims)
        self.assertNotIn("access_token", catalog["tables"]["channel_accounts"]["dimensions"])
        self.assertNotIn("refresh_token", catalog["tables"]["channel_accounts"]["dimensions"])
        self.assertNotIn("password", catalog["prompt_catalog"])
        self.assertNotIn("access_token", catalog["prompt_catalog"])

    def test_measures_and_currency_classification(self):
        """Verifies numeric columns are auto-classified into currency vs numeric measures."""
        mock_cols = [
            {"TABLE_NAME": "analytics_orders", "COLUMN_NAME": "id", "DATA_TYPE": "bigint", "COLUMN_TYPE": "bigint", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "EXTRA": ""},
            {"TABLE_NAME": "analytics_orders", "COLUMN_NAME": "total_amount", "DATA_TYPE": "decimal", "COLUMN_TYPE": "decimal(12,2)", "IS_NULLABLE": "NO", "COLUMN_KEY": "", "EXTRA": ""},
            {"TABLE_NAME": "analytics_orders", "COLUMN_NAME": "item_count", "DATA_TYPE": "int", "COLUMN_TYPE": "int", "IS_NULLABLE": "NO", "COLUMN_KEY": "", "EXTRA": ""},
            {"TABLE_NAME": "analytics_orders", "COLUMN_NAME": "order_date", "DATA_TYPE": "date", "COLUMN_TYPE": "date", "IS_NULLABLE": "NO", "COLUMN_KEY": "", "EXTRA": ""},
        ]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [mock_cols, []]
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(self.engine, '_get_connection', return_value=mock_conn):
            catalog = self.engine.discover_and_catalog(force_refresh=True)

        self.assertIn("total_amount", catalog["measures"])
        self.assertEqual(catalog["measures"]["total_amount"]["data_type"], "currency")
        self.assertIn("item_count", catalog["measures"])
        self.assertEqual(catalog["measures"]["item_count"]["data_type"], "numeric")
        self.assertIn("order_date", catalog["tables"]["analytics_orders"]["dates"])

    def test_cache_ttl_behavior(self):
        """Verifies discover_and_catalog serves from memory cache when TTL is valid."""
        mock_cols = [
            {"TABLE_NAME": "items", "COLUMN_NAME": "id", "DATA_TYPE": "int", "COLUMN_TYPE": "int", "IS_NULLABLE": "NO", "COLUMN_KEY": "PRI", "EXTRA": ""},
        ]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [mock_cols, []]
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(self.engine, '_get_connection', return_value=mock_conn) as conn_mock:
            res1 = self.engine.discover_and_catalog(force_refresh=True)
            self.assertEqual(conn_mock.call_count, 1)

            # Second call without force_refresh should hit cache without calling db
            res2 = self.engine.discover_and_catalog(force_refresh=False)
            self.assertEqual(conn_mock.call_count, 1)
            self.assertEqual(res1, res2)

    def test_empty_schema_fallback(self):
        """Verifies empty or failing schema connection falls back gracefully without crashing."""
        with patch.object(self.engine, '_get_connection', side_effect=Exception("DB Connection Timeout")):
            catalog = self.engine.discover_and_catalog(force_refresh=True)
            self.assertIsInstance(catalog, dict)
            self.assertIn("tables", catalog)
            self.assertEqual(catalog["tables"], {})

if __name__ == '__main__':
    unittest.main()
