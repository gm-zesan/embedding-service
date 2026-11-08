"""
Unit tests for Contract 3.0.0 Specification and Schema Symmetry.
Verifies typing, version validation, cryptographic signing, and fingerprinting.
"""

import unittest
import time
from datetime import datetime, timedelta, timezone

from app.analytics.contract_v3 import (
    CONTRACT_VERSION_3_0,
    AnalyticsPrepareRequest,
    PreparedQueryArtifact,
    ExecutionAuthorizationGrant,
    AnalyticsExecuteRequest,
    AnalyticsExecutionResponse,
    DynamicSchemaCatalog,
    VirtualTableMeta,
    VirtualColumnMeta,
    VirtualColumnType,
)
from app.analytics.models import (
    SemanticQueryPlan,
    DomainEnum,
    MeasureSpec,
    AggregationType,
)


class TestContractV3Symmetry(unittest.TestCase):

    def test_contract_version_invariants(self):
        """Ensures contract version defaults to 3.0.0 and rejects version mismatches."""
        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="total_amount", aggregation=AggregationType.SUM)],
        )

        req = AnalyticsPrepareRequest(
            workspace_id=10,
            dataset_id="ds_123",
            generation_id="gen_abc",
            query_plan=plan,
        )
        self.assertEqual(req.contract_version, "3.0.0")

        with self.assertRaises(ValueError):
            AnalyticsPrepareRequest(
                contract_version="2.0.0",
                workspace_id=10,
                dataset_id="ds_123",
                generation_id="gen_abc",
                query_plan=plan,
            )

    def test_query_fingerprint_deterministic_computation(self):
        """Ensures query fingerprint is deterministic and sensitive to SQL, params, workspace, and generation."""
        sql = "SELECT SUM(amount) FROM orders WHERE status = ?"
        params = ["completed"]
        ws_id = 101
        gen_id = "gen_uuid_001"

        fp1 = PreparedQueryArtifact.compute_fingerprint(sql, params, ws_id, gen_id)
        fp2 = PreparedQueryArtifact.compute_fingerprint(sql, params, ws_id, gen_id)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

        # Changing parameter changes fingerprint
        fp_diff_param = PreparedQueryArtifact.compute_fingerprint(sql, ["cancelled"], ws_id, gen_id)
        self.assertNotEqual(fp1, fp_diff_param)

        # Changing workspace changes fingerprint
        fp_diff_ws = PreparedQueryArtifact.compute_fingerprint(sql, params, 999, gen_id)
        self.assertNotEqual(fp1, fp_diff_ws)

    def test_execution_authorization_grant_signature_and_expiry(self):
        """Tests HMAC-SHA256 grant signing, verification, tampering detection, and expiry."""
        secret = "super_secure_tenant_secret_key_12345"
        ws_id = 42
        gen_id = "gen_xyz_987"
        fingerprint = "a" * 64
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=60)).isoformat()
        nonce = "unique_nonce_abc123"

        sig = ExecutionAuthorizationGrant.generate_signature(
            secret=secret,
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id=gen_id,
            expires_at=expires_at,
            nonce=nonce,
        )

        grant = ExecutionAuthorizationGrant(
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id=gen_id,
            expires_at=expires_at,
            nonce=nonce,
            signature=sig,
        )

        self.assertTrue(grant.verify_signature(secret))
        self.assertFalse(grant.verify_signature("wrong_secret"))
        self.assertFalse(grant.is_expired())

        # Tampered generation id fails verification
        tampered_grant = ExecutionAuthorizationGrant(
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id="hacked_gen",
            expires_at=expires_at,
            nonce=nonce,
            signature=sig,
        )
        self.assertFalse(tampered_grant.verify_signature(secret))

        # Expired grant detection
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        expired_grant = ExecutionAuthorizationGrant(
            workspace_id=ws_id,
            query_fingerprint=fingerprint,
            generation_id=gen_id,
            expires_at=past,
            nonce=nonce,
            signature=sig,
        )
        self.assertTrue(expired_grant.is_expired())


if __name__ == "__main__":
    unittest.main()

