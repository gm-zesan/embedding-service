"""
Test Legacy Excel Route Retirement & Contract 3.0 Virtual Gateway Enforcement.
Verifies P0-C security guarantees:
1. Legacy /excel/query and /sources/query cannot execute SQL.
2. Direct requests return explicit non-success retired responses.
3. excel_engine.query_excel is never called by legacy endpoints.
4. Contract 3.0 /analytics/v3/execute remains the sole governed execution boundary.
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.analytics.contract_v3 import (
    ExecutionAuthorizationGrant,
    AnalyticsExecuteRequest,
    CONTRACT_VERSION_3_0,
    PreparedQueryArtifact,
)
from app.analytics.execution_gateway import (
    VirtualExecutionGateway,
    ExecutionGatewaySecurityError,
)
import time
import os
import sqlite3
import tempfile


class TestLegacyExcelRouteRetirement(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.secret_key = "test-secret-key-remediation-123"

    def _make_grant(self, workspace_id: int, query_fingerprint: str, generation_id: str, secret: str = None, expires_in_sec: int = 60, nonce: str = "nonce_1"):
        now = time.time()
        expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + expires_in_sec))
        s = secret or self.secret_key
        sig = ExecutionAuthorizationGrant.generate_signature(
            secret=s,
            workspace_id=workspace_id,
            query_fingerprint=query_fingerprint,
            generation_id=generation_id,
            expires_at=expires_at,
            nonce=nonce,
        )
        return ExecutionAuthorizationGrant(
            workspace_id=workspace_id,
            query_fingerprint=query_fingerprint,
            generation_id=generation_id,
            expires_at=expires_at,
            nonce=nonce,
            signature=sig,
        )

    @patch("app.analytics.excel_engine.excel_engine.query_excel")
    def test_legacy_excel_query_route_cannot_execute_sql(self, mock_query_excel):
        """
        Verify POST /analytics/excel/query returns retired status and never calls query_excel.
        """
        response = self.client.post(
            "/analytics/excel/query",
            headers={"X-API-Key": "test-key"},
            json={
                "workspace_id": 99,
                "question": "What is total salary?",
                "file_id": "file_test_123",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["status"], "retired")
        self.assertEqual(data["error"], "ENDPOINT_RETIRED")
        self.assertIn("retired", data["report"])
        self.assertEqual(data["rows"], [])

        # Crucial security assertion: query_excel was NEVER called!
        mock_query_excel.assert_not_called()

    @patch("app.analytics.excel_engine.excel_engine.query_excel")
    def test_legacy_sources_query_route_cannot_execute_sql(self, mock_query_excel):
        """
        Verify POST /analytics/sources/query returns retired status and never calls query_excel.
        """
        response = self.client.post(
            "/analytics/sources/query",
            headers={"X-API-Key": "test-key"},
            json={
                "workspace_id": 99,
                "question": "Show all margins",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["status"], "retired")
        self.assertEqual(data["error"], "ENDPOINT_RETIRED")

        mock_query_excel.assert_not_called()

    @patch("app.analytics.excel_engine.excel_engine.query_excel")
    def test_query_analytics_rejects_direct_excel_engine_dispatch(self, mock_query_excel):
        """
        Verify POST /analytics/query with engine='excel' returns security rejection and does not call query_excel.
        """
        response = self.client.post(
            "/analytics/query",
            headers={"X-API-Key": "test-key"},
            json={
                "workspace_id": 99,
                "query": "Show me sheet revenue",
                "engine": "excel",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertTrue(data.get("is_security_rejection"))
        self.assertIn("retired", data.get("report", ""))

        mock_query_excel.assert_not_called()

    def test_contract_v3_valid_signed_grant_executes_successfully(self):
        """
        Verify Stage C /analytics/v3/execute succeeds when grant is valid and signed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws_id = 42
            ds_id = "dataset_valid_test"
            gen_id = "gen_uuid_42"
            db_path = os.path.join(tmp_dir, f"ws_{ws_id}_{ds_id}.sqlite")

            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE sales (id INTEGER, amount REAL);")
            conn.execute("INSERT INTO sales VALUES (1, 100.5), (2, 200.0);")
            conn.commit()
            conn.close()

            sql = "SELECT SUM(amount) AS total FROM sales"
            fingerprint = PreparedQueryArtifact.compute_fingerprint(
                sql=sql, params=[], workspace_id=ws_id, generation_id=gen_id
            )

            grant = self._make_grant(
                workspace_id=ws_id,
                query_fingerprint=fingerprint,
                generation_id=gen_id,
                nonce="nonce_test_valid_123",
            )

            gateway = VirtualExecutionGateway(storage_dir=tmp_dir, secret_key=self.secret_key)
            req = AnalyticsExecuteRequest(
                contract_version=CONTRACT_VERSION_3_0,
                request_id="req_valid_01",
                workspace_id=ws_id,
                dataset_id=ds_id,
                generation_id=gen_id,
                query_fingerprint=fingerprint,
                sql=sql,
                params=[],
                authorization_grant=grant,
            )

            resp = gateway.execute_authorized_query(req)
            self.assertEqual(resp.row_count, 1)
            self.assertEqual(resp.rows[0]["total"], 300.5)

    def test_contract_v3_invalid_signature_is_rejected(self):
        """
        Verify grant with forged signature raises INVALID_SIGNATURE.
        """
        ws_id = 42
        ds_id = "ds_tamper"
        gen_id = "gen_uuid"
        sql = "SELECT * FROM orders"
        fingerprint = PreparedQueryArtifact.compute_fingerprint(
            sql=sql, params=[], workspace_id=ws_id, generation_id=gen_id
        )

        grant = self._make_grant(
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id=gen_id,
            secret="different-secret-key",
            nonce="nonce_tamper_123",
        )

        gateway = VirtualExecutionGateway(secret_key=self.secret_key)
        req = AnalyticsExecuteRequest(
            contract_version=CONTRACT_VERSION_3_0,
            request_id="req_tamper",
            workspace_id=ws_id,
            dataset_id=ds_id,
            generation_id=gen_id,
            query_fingerprint=fingerprint,
            sql=sql,
            params=[],
            authorization_grant=grant,
        )

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "INVALID_SIGNATURE")

    def test_contract_v3_expired_grant_is_rejected(self):
        """
        Verify expired grant raises GRANT_EXPIRED.
        """
        ws_id = 42
        ds_id = "ds_expired"
        gen_id = "gen_uuid"
        sql = "SELECT 1"
        fingerprint = PreparedQueryArtifact.compute_fingerprint(
            sql=sql, params=[], workspace_id=ws_id, generation_id=gen_id
        )

        grant = self._make_grant(
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id=gen_id,
            expires_in_sec=-10,  # Expired in past
            nonce="nonce_expired_123",
        )

        gateway = VirtualExecutionGateway(secret_key=self.secret_key)
        req = AnalyticsExecuteRequest(
            contract_version=CONTRACT_VERSION_3_0,
            request_id="req_exp",
            workspace_id=ws_id,
            dataset_id=ds_id,
            generation_id=gen_id,
            query_fingerprint=fingerprint,
            sql=sql,
            params=[],
            authorization_grant=grant,
        )

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "GRANT_EXPIRED")

    def test_contract_v3_wrong_workspace_is_rejected(self):
        """
        Verify workspace mismatch between request and grant raises WORKSPACE_MISMATCH.
        """
        ws_id = 42
        gen_id = "gen_uuid"
        sql = "SELECT 1"
        fingerprint = PreparedQueryArtifact.compute_fingerprint(
            sql=sql, params=[], workspace_id=ws_id, generation_id=gen_id
        )

        grant = self._make_grant(
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id=gen_id,
            nonce="nonce_ws_mismatch",
        )

        gateway = VirtualExecutionGateway(secret_key=self.secret_key)
        req = AnalyticsExecuteRequest(
            contract_version=CONTRACT_VERSION_3_0,
            request_id="req_ws",
            workspace_id=999,  # Mismatched!
            dataset_id="ds_mismatch",
            generation_id=gen_id,
            query_fingerprint=fingerprint,
            sql=sql,
            params=[],
            authorization_grant=grant,
        )

        with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
            gateway.execute_authorized_query(req)
        self.assertEqual(ctx.exception.code, "WORKSPACE_MISMATCH")

    def test_contract_v3_replayed_nonce_is_rejected(self):
        """
        Verify replaying the same nonce raises REPLAY_DETECTED.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws_id = 42
            ds_id = "ds_replay"
            gen_id = "gen_uuid"
            db_path = os.path.join(tmp_dir, f"ws_{ws_id}_{ds_id}.sqlite")

            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE t (x INT);")
            conn.execute("INSERT INTO t VALUES (1);")
            conn.commit()
            conn.close()

            sql = "SELECT x FROM t"
            fingerprint = PreparedQueryArtifact.compute_fingerprint(
                sql=sql, params=[], workspace_id=ws_id, generation_id=gen_id
            )

            grant = self._make_grant(
                workspace_id=ws_id,
                query_fingerprint=fingerprint,
                generation_id=gen_id,
                nonce="nonce_single_use_test_999",
            )

            gateway = VirtualExecutionGateway(storage_dir=tmp_dir, secret_key=self.secret_key)
            req = AnalyticsExecuteRequest(
                contract_version=CONTRACT_VERSION_3_0,
                request_id="req_replay",
                workspace_id=ws_id,
                dataset_id=ds_id,
                generation_id=gen_id,
                query_fingerprint=fingerprint,
                sql=sql,
                params=[],
                authorization_grant=grant,
            )

            # First execution succeeds
            res1 = gateway.execute_authorized_query(req)
            self.assertEqual(res1.row_count, 1)

            # Replay same request -> rejected!
            with self.assertRaises(ExecutionGatewaySecurityError) as ctx:
                gateway.execute_authorized_query(req)
            self.assertEqual(ctx.exception.code, "REPLAY_DETECTED")


if __name__ == "__main__":
    unittest.main()
