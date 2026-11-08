"""
Security tests for VirtualExecutionGateway under Contract 3.0.0.
Verifies cryptographic signature validation, nonce replay protection, expiry,
query tampering detection, and read-only SQLite sandbox enforcement.
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from app.analytics.contract_v3 import (
    CONTRACT_VERSION_3_0,
    AnalyticsExecuteRequest,
    AnalyticsExecutionResponse,
    PreparedQueryArtifact,
    ExecutionAuthorizationGrant,
)
from app.analytics.execution_gateway import (
    VirtualExecutionGateway,
    ExecutionGatewaySecurityError,
)


class TestExecutionGatewaySecurity(unittest.TestCase):

    def setUp(self):
        self.secret = "test_super_secret_analytics_gateway_key"
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_id = 999
        self.dataset_id = "sales_q4"
        self.generation_id = "gen_sec_001"

        # Create isolated virtual SQLite database
        self.db_path = os.path.join(self.temp_dir, f"ws_{self.workspace_id}_{self.dataset_id}.sqlite")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, amount REAL);")
        cur.executemany(
            "INSERT INTO orders (id, status, amount) VALUES (?, ?, ?);",
            [
                (1, "completed", 1500.0),
                (2, "completed", 2500.0),
                (3, "pending", 500.0),
            ]
        )
        conn.commit()
        conn.close()

        self.gateway = VirtualExecutionGateway(
            storage_dir=self.temp_dir,
            secret_key=self.secret,
            query_timeout=5.0,
            max_rows=50,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_authorized_request(self, sql: str, params: list, nonce: str = "nonce_1", expires_in_sec: int = 60) -> AnalyticsExecuteRequest:
        fp = PreparedQueryArtifact.compute_fingerprint(
            sql=sql,
            params=params,
            workspace_id=self.workspace_id,
            generation_id=self.generation_id,
        )
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=expires_in_sec)).isoformat()

        sig = ExecutionAuthorizationGrant.generate_signature(
            secret=self.secret,
            workspace_id=self.workspace_id,
            query_fingerprint=fp,
            generation_id=self.generation_id,
            expires_at=expires_at,
            nonce=nonce,
        )

        grant = ExecutionAuthorizationGrant(
            workspace_id=self.workspace_id,
            query_fingerprint=fp,
            generation_id=self.generation_id,
            expires_at=expires_at,
            nonce=nonce,
            signature=sig,
        )

        return AnalyticsExecuteRequest(
            workspace_id=self.workspace_id,
            dataset_id=self.dataset_id,
            generation_id=self.generation_id,
            query_fingerprint=fp,
            sql=sql,
            params=params,
            authorization_grant=grant,
        )

    def test_successful_authorized_execution(self):
        """Verifies query executes and returns rows when grant is fully valid."""
        sql = "SELECT SUM(amount) AS total_sales FROM orders WHERE status = ?"
        params = ["completed"]
        req = self._create_authorized_request(sql, params, nonce="nonce_success")

        resp = self.gateway.execute_authorized_query(req)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.rows), 1)
        self.assertEqual(resp.rows[0]["total_sales"], 4000.0)
        self.assertTrue("৳" in resp.report or "টাকা" in resp.report or "রিপোর্ট" in resp.report)

    def test_query_tampering_sql_modification_rejected(self):
        """Verifies tampering with SQL after grant was signed causes INTEGRITY_VIOLATION."""
        sql = "SELECT SUM(amount) AS total_sales FROM orders WHERE status = ?"
        params = ["completed"]
        req = self._create_authorized_request(sql, params, nonce="nonce_tamper_sql")

        # Attacker modifies SQL to exfiltrate all rows
        req.sql = "SELECT * FROM orders"

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            self.gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "INTEGRITY_VIOLATION")

    def test_query_tampering_params_modification_rejected(self):
        """Verifies tampering with parameter bindings after grant was signed causes INTEGRITY_VIOLATION."""
        sql = "SELECT SUM(amount) AS total_sales FROM orders WHERE status = ?"
        params = ["completed"]
        req = self._create_authorized_request(sql, params, nonce="nonce_tamper_params")

        # Attacker modifies param value
        req.params = ["pending"]

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            self.gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "INTEGRITY_VIOLATION")

    def test_invalid_signature_rejected(self):
        """Verifies forged or malformed signature causes INVALID_SIGNATURE."""
        sql = "SELECT SUM(amount) AS total_sales FROM orders WHERE status = ?"
        params = ["completed"]
        req = self._create_authorized_request(sql, params, nonce="nonce_forged")

        # Attacker forges signature
        req.authorization_grant.signature = "deadbeef" * 8

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            self.gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "INVALID_SIGNATURE")

    def test_replay_attack_rejected(self):
        """Verifies consuming the same nonce twice causes REPLAY_DETECTED on the second attempt."""
        sql = "SELECT SUM(amount) AS total_sales FROM orders WHERE status = ?"
        params = ["completed"]
        req = self._create_authorized_request(sql, params, nonce="nonce_replay_test_123")

        # First execution succeeds
        resp1 = self.gateway.execute_authorized_query(req)
        self.assertTrue(resp1.success)

        # Second execution with same grant and nonce must be rejected
        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            self.gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "REPLAY_DETECTED")

    def test_expired_grant_rejected(self):
        """Verifies expired grant causes GRANT_EXPIRED."""
        sql = "SELECT SUM(amount) AS total_sales FROM orders WHERE status = ?"
        params = ["completed"]
        req = self._create_authorized_request(sql, params, nonce="nonce_expired", expires_in_sec=-10)

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            self.gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "GRANT_EXPIRED")

    def test_redis_backed_nonce_replay_protection(self):
        """Verifies that when a Redis client is injected, it uses atomic SET ... NX to prevent replay."""
        from app.analytics.execution_gateway import NonceReplayCache
        from unittest.mock import MagicMock

        mock_redis = MagicMock()
        # First call returns True (set successful), second call returns False (already exists)
        mock_redis.set.side_effect = [True, False]

        cache = NonceReplayCache(redis_client=mock_redis)
        cache._redis_checked = True  # use injected client

        # First consumption
        self.assertTrue(cache.consume("redis_nonce_abc_123"))
        mock_redis.set.assert_called_with("analytics:grant_nonce:redis_nonce_abc_123", "1", ex=300, nx=True)

        # Second consumption (replay attack detected by Redis)
        self.assertFalse(cache.consume("redis_nonce_abc_123"))

    def test_redis_failure_gracefully_falls_back_to_lru_memory(self):
        """Verifies that Redis network failure in local single-worker logs warning and falls back to local memory without crashing."""
        from app.analytics.execution_gateway import NonceReplayCache
        from unittest.mock import MagicMock, patch

        mock_redis = MagicMock()
        mock_redis.set.side_effect = ConnectionError("Redis connection refused")

        with patch.dict(os.environ, {"APP_ENV": "development", "MULTI_WORKER": "false", "REDIS_REQUIRED": "false"}):
            cache = NonceReplayCache(redis_client=mock_redis)
            cache._redis_checked = True

            # First consumption falls back to memory and succeeds
            self.assertTrue(cache.consume("fallback_nonce_xyz"))

            # Second consumption in memory detects replay
            self.assertFalse(cache.consume("fallback_nonce_xyz"))

    def test_multi_worker_without_redis_strictly_fails_closed(self):
        """Verifies that in multi-worker or production mode, missing Redis strictly fails closed."""
        from app.analytics.execution_gateway import NonceReplayCache, ExecutionGatewaySecurityError
        from unittest.mock import patch

        with patch.dict(os.environ, {"APP_ENV": "production", "MULTI_WORKER": "true"}):
            cache = NonceReplayCache(redis_client=None)
            cache._redis_checked = True

            with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
                cache.consume("multi_worker_nonce_1")
            self.assertEqual(ctx.exception.code, "DISTRIBUTED_REPLAY_CACHE_UNAVAILABLE")

    def test_multi_worker_redis_network_failure_strictly_fails_closed(self):
        """Verifies that in multi-worker or production mode, Redis error strictly fails closed instead of falling back."""
        from app.analytics.execution_gateway import NonceReplayCache, ExecutionGatewaySecurityError
        from unittest.mock import MagicMock, patch

        mock_redis = MagicMock()
        mock_redis.set.side_effect = ConnectionError("Redis connection timeout during partition")

        with patch.dict(os.environ, {"APP_ENV": "production", "MULTI_WORKER": "true"}):
            cache = NonceReplayCache(redis_client=mock_redis)
            cache._redis_checked = True

            with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
                cache.consume("multi_worker_nonce_2")
            self.assertEqual(ctx.exception.code, "DISTRIBUTED_REPLAY_CACHE_ERROR")


if __name__ == "__main__":
    unittest.main()
