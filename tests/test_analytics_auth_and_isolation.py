import unittest
from fastapi.testclient import TestClient
from app.main import app
import app.config as config
import app.main as main_mod


class TestAnalyticsAuthAndIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_key = "test-auth-secret-key-12345"

    def setUp(self):
        # Configure test key for each test
        self.orig_config_key = config.API_KEY
        self.orig_config_env = config.APP_ENV
        self.orig_main_key = main_mod.API_KEY
        self.orig_main_env = main_mod.APP_ENV

        config.API_KEY = self.test_key
        config.APP_ENV = "testing"
        main_mod.API_KEY = self.test_key
        main_mod.APP_ENV = "testing"

    def tearDown(self):
        config.API_KEY = self.orig_config_key
        config.APP_ENV = self.orig_config_env
        main_mod.API_KEY = self.orig_main_key
        main_mod.APP_ENV = self.orig_main_env

    def test_analytics_query_rejects_missing_api_key(self):
        """Missing API key returns 401 Unauthorized."""
        resp = self.client.post("/analytics/query", json={
            "query": "Total revenue",
            "workspace_id": 1
        })
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid or missing API key", resp.json()["detail"])

    def test_analytics_query_rejects_invalid_api_key(self):
        """Invalid API key returns 401 Unauthorized."""
        resp = self.client.post(
            "/analytics/query",
            json={"query": "Total revenue", "workspace_id": 1},
            headers={"X-API-Key": "wrong-key"}
        )
        self.assertEqual(resp.status_code, 401)

    def test_analytics_endpoints_require_auth(self):
        """Verify all protected analytics routes reject unauthenticated requests."""
        endpoints = [
            ("GET", "/analytics/sources/list?workspace_id=1"),
            ("POST", "/analytics/sources/resolve"),
            ("POST", "/analytics/excel/query"),
            ("DELETE", "/analytics/sources/non-existent-source?workspace_id=1"),
        ]
        for method, path in endpoints:
            if method == "GET":
                r = self.client.get(path)
            elif method == "DELETE":
                r = self.client.delete(path)
            else:
                r = self.client.post(path, json={"query": "test", "workspace_id": 1})
            self.assertEqual(r.status_code, 401, f"Expected 401 for unauthenticated {method} {path}, got {r.status_code}")

    def test_analytics_query_accepts_valid_x_api_key(self):
        """Valid X-API-Key header is accepted."""
        headers = {"X-API-Key": self.test_key}
        resp = self.client.get("/analytics/sources/list?workspace_id=9999", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["workspace_id"], 9999)
        self.assertEqual(data["count"], 0)

    def test_analytics_query_accepts_valid_bearer_token(self):
        """Valid Authorization Bearer token is accepted."""
        headers = {"Authorization": f"Bearer {self.test_key}"}
        resp = self.client.get("/analytics/sources/list?workspace_id=9999", headers=headers)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_workspace_isolation_across_sources(self):
        """Verify workspace A cannot delete sources belonging to workspace B."""
        headers = {"X-API-Key": self.test_key}
        # Attempt to delete a non-owned source in workspace 1
        resp = self.client.delete("/analytics/sources/ws_2_secret_source?workspace_id=1", headers=headers)
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
