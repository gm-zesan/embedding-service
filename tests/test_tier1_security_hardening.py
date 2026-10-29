import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
import app.config as config
import app.main as main_mod


class TestTier1SecurityHardening(unittest.TestCase):
    """
    Tier 1 Security Verification Suite:
    - Authentication defaults and environment enforcement
    - Missing secrets handling
    - Opt-in local dev override controls
    - Restrictive CORS origin filtering
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_secret = "test-tier1-super-secret-key-999"

    def setUp(self):
        # Snapshot configuration state
        self.orig_config_key = config.API_KEY
        self.orig_config_env = config.APP_ENV
        self.orig_config_optin = config.ALLOW_UNAUTHENTICATED_DEV
        self.orig_main_key = main_mod.API_KEY
        self.orig_main_env = main_mod.APP_ENV
        self.orig_main_optin = main_mod.ALLOW_UNAUTHENTICATED_DEV

    def tearDown(self):
        # Restore configuration state
        config.API_KEY = self.orig_config_key
        config.APP_ENV = self.orig_config_env
        config.ALLOW_UNAUTHENTICATED_DEV = self.orig_config_optin
        main_mod.API_KEY = self.orig_main_key
        main_mod.APP_ENV = self.orig_main_env
        main_mod.ALLOW_UNAUTHENTICATED_DEV = self.orig_main_optin

    # ── 1. Authentication Defaults & Secret Enforcement ───────────────────────

    def test_missing_secret_in_production_fails_closed_with_500(self):
        """Missing API_KEY in production must fail closed with HTTP 500 (server misconfiguration)."""
        config.API_KEY = ""
        config.APP_ENV = "production"
        config.ALLOW_UNAUTHENTICATED_DEV = False
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "production"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = False

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Server misconfiguration", resp.json()["detail"])

    def test_missing_secret_in_prod_alias_with_optin_fails_closed_with_500(self):
        """Even with ALLOW_UNAUTHENTICATED_DEV=True, APP_ENV='prod' must fail closed with HTTP 500."""
        config.API_KEY = ""
        config.APP_ENV = "prod"
        config.ALLOW_UNAUTHENTICATED_DEV = True
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "prod"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = True

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Server misconfiguration", resp.json()["detail"])

    def test_missing_secret_in_live_alias_with_optin_fails_closed_with_500(self):
        """Even with ALLOW_UNAUTHENTICATED_DEV=True, APP_ENV='live' must fail closed with HTTP 500."""
        config.API_KEY = ""
        config.APP_ENV = "live"
        config.ALLOW_UNAUTHENTICATED_DEV = True
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "live"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = True

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Server misconfiguration", resp.json()["detail"])

    def test_missing_secret_in_staging_fails_closed_with_500(self):
        """Missing API_KEY in staging must fail closed with HTTP 500."""
        config.API_KEY = ""
        config.APP_ENV = "staging"
        config.ALLOW_UNAUTHENTICATED_DEV = False
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "staging"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = False

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Server misconfiguration", resp.json()["detail"])

    def test_missing_secret_in_local_without_optin_fails_closed_with_500(self):
        """Missing API_KEY in local development fails closed with 500 unless ALLOW_UNAUTHENTICATED_DEV=True."""
        config.API_KEY = ""
        config.APP_ENV = "local"
        config.ALLOW_UNAUTHENTICATED_DEV = False
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "local"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = False

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Unauthenticated access is strictly forbidden outside local development", resp.json()["detail"])

    def test_optin_dev_flag_is_strictly_ignored_in_production(self):
        """Even if an operator mistakenly sets ALLOW_UNAUTHENTICATED_DEV=True in production, it MUST fail closed."""
        config.API_KEY = ""
        config.APP_ENV = "production"
        config.ALLOW_UNAUTHENTICATED_DEV = True  # Attempted override in production
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "production"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = True

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Server misconfiguration", resp.json()["detail"])

    def test_optin_dev_flag_permits_unauthenticated_access_only_in_local(self):
        """In local development with explicit opt-in, unauthenticated access is permitted."""
        config.API_KEY = ""
        config.APP_ENV = "local"
        config.ALLOW_UNAUTHENTICATED_DEV = True
        main_mod.API_KEY = ""
        main_mod.APP_ENV = "local"
        main_mod.ALLOW_UNAUTHENTICATED_DEV = True

        with patch("app.main.search_knowledge_base", return_value={"query": "test", "total_found": 0, "results": []}):
            resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
            self.assertEqual(resp.status_code, 200)

    # ── 2. Credential Verification & Constant-Time Checking ───────────────────

    def test_configured_api_key_rejects_missing_header(self):
        """When API_KEY is set, requests lacking auth headers return 401."""
        config.API_KEY = self.test_secret
        config.APP_ENV = "production"
        main_mod.API_KEY = self.test_secret
        main_mod.APP_ENV = "production"

        resp = self.client.post("/api/v1/search", json={"query": "test", "workspace_id": 1})
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["detail"], "Invalid or missing API key")

    def test_configured_api_key_rejects_invalid_key(self):
        """When API_KEY is set, invalid key returns 401 without secret leakage."""
        config.API_KEY = self.test_secret
        config.APP_ENV = "production"
        main_mod.API_KEY = self.test_secret
        main_mod.APP_ENV = "production"

        resp = self.client.post(
            "/api/v1/search",
            json={"query": "test", "workspace_id": 1},
            headers={"X-API-Key": "incorrect-token-abc"},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["detail"], "Invalid or missing API key")

    def test_configured_api_key_accepts_valid_x_api_key(self):
        """Valid X-API-Key header is accepted."""
        config.API_KEY = self.test_secret
        config.APP_ENV = "production"
        main_mod.API_KEY = self.test_secret
        main_mod.APP_ENV = "production"

        with patch("app.main.search_knowledge_base", return_value={"query": "test", "total_found": 0, "results": []}):
            resp = self.client.post(
                "/api/v1/search",
                json={"query": "test", "workspace_id": 1},
                headers={"X-API-Key": self.test_secret},
            )
            self.assertEqual(resp.status_code, 200)

    def test_configured_api_key_accepts_valid_bearer_token(self):
        """Valid Authorization: Bearer <key> header is accepted."""
        config.API_KEY = self.test_secret
        config.APP_ENV = "production"
        main_mod.API_KEY = self.test_secret
        main_mod.APP_ENV = "production"

        with patch("app.main.search_knowledge_base", return_value={"query": "test", "total_found": 0, "results": []}):
            resp = self.client.post(
                "/api/v1/search",
                json={"query": "test", "workspace_id": 1},
                headers={"Authorization": f"Bearer {self.test_secret}"},
            )
            self.assertEqual(resp.status_code, 200)

    # ── 3. Restrictive CORS Origin Filtering ──────────────────────────────────

    def test_cors_allows_configured_trusted_origin(self):
        """OPTIONS preflight from an explicitly allowed origin receives Access-Control-Allow-Origin."""
        resp = self.client.options(
            "/api/v1/search",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-API-Key",
            },
        )
        # Should permit configured origin
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://localhost:8000")

    def test_cors_rejects_untrusted_external_origin(self):
        """OPTIONS preflight from an untrusted origin does NOT receive Access-Control-Allow-Origin."""
        resp = self.client.options(
            "/api/v1/search",
            headers={
                "Origin": "https://malicious-attacker.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # Untrusted origin must never be reflected in Access-Control-Allow-Origin
        self.assertIsNone(resp.headers.get("access-control-allow-origin"))

    def test_cors_never_returns_wildcard_origin(self):
        """CORS header must never return wildcard '*'."""
        resp = self.client.options(
            "/api/v1/search",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertNotEqual(resp.headers.get("access-control-allow-origin"), "*")


if __name__ == "__main__":
    unittest.main()
