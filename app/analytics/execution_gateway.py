"""
Execution Gateway for Phase 3 Contract 3.0.0.
Enforces the mandatory security barrier:
1. Validates cryptographically signed ExecutionAuthorizationGrant.
2. Checks grant expiry (max 60 seconds).
3. Verifies single-use nonce for strict replay protection.
4. Verifies query fingerprint matches the authorized SQL and parameters.
5. Executes parameterized SQL against isolated virtual SQLite database in read-only mode.
6. Enforces timeout (15s) and row bounds (50 rows).
"""

import os
import sqlite3
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import OrderedDict

from app.analytics.contract_v3 import (
    CONTRACT_VERSION_3_0,
    AnalyticsExecuteRequest,
    AnalyticsExecutionResponse,
    PreparedQueryArtifact,
    ExecutionAuthorizationGrant,
)
from app.analytics.formatter import AnalyticsFormatter
from app import config

logger = logging.getLogger(__name__)


class ExecutionGatewaySecurityError(Exception):
    """Raised when an authorization grant or security check fails at execution time."""
    def __init__(self, message: str, code: str = "SECURITY_VIOLATION"):
        super().__init__(message)
        self.code = code


def is_multi_worker_environment() -> bool:
    """
    Determines if the runtime environment is multi-worker or production,
    where single-instance in-memory caching is unsafe for nonce replay protection.
    """
    env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower()
    if env in ("production", "staging", "prod"):
        return True
    if os.getenv("MULTI_WORKER", "").lower() in ("true", "1", "yes"):
        return True
    if os.getenv("REDIS_REQUIRED", "").lower() in ("true", "1", "yes"):
        return True
    try:
        workers = int(os.getenv("WEB_CONCURRENCY", os.getenv("WORKERS", "1")))
        if workers > 1:
            return True
    except (ValueError, TypeError):
        pass
    return False


class NonceReplayCache:
    """
    Hybrid Nonce Replay Cache:
    1. Multi-Worker / Production: Redis is strictly required. If Redis is unavailable or errors,
       fails closed (raises ExecutionGatewaySecurityError) to prevent cross-worker replay attacks.
    2. Local Development (Single-Worker): In-memory bounded LRU cache with monotonic TTL,
       with optional graceful fallback from Redis if configured.
    Guarantees strict single-use nonce consumption across single or multi-worker deployments.
    """
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300.0, redis_client: Any = None):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._nonces: OrderedDict[str, float] = OrderedDict()
        self._redis = redis_client
        self._redis_checked = False

    def _get_redis(self):
        if self._redis_checked:
            return self._redis

        self._redis_checked = True
        redis_url = os.getenv("REDIS_URL")
        redis_host = os.getenv("REDIS_HOST")

        if not redis_url and not redis_host:
            return None

        try:
            import redis
            if redis_url:
                self._redis = redis.from_url(redis_url, socket_timeout=2.0)
            else:
                port = int(os.getenv("REDIS_PORT", 6379))
                password = os.getenv("REDIS_PASSWORD")
                self._redis = redis.Redis(host=redis_host, port=port, password=password, socket_timeout=2.0)
            self._redis.ping()
            logger.info("[NonceReplayCache] Connected to distributed Redis for multi-worker nonce replay protection.")
        except Exception as e:
            if is_multi_worker_environment():
                logger.error(f"[NonceReplayCache] Redis connection failed in multi-worker environment: {e}.")
            else:
                logger.warning(f"[NonceReplayCache] Redis unavailable ({e}). Falling back to in-memory LRU cache in local single-worker mode.")
            self._redis = None

        return self._redis

    def consume(self, nonce: str) -> bool:
        """
        Attempts to consume a nonce.
        - In multi-worker / production: Requires Redis atomic SET NX. If Redis is missing or fails,
          FAILS CLOSED immediately by raising ExecutionGatewaySecurityError.
        - In local single-worker dev: Uses Redis if available, otherwise safely falls back to local LRU.
        Returns True if fresh and consumed; returns False if already seen (replay detected).
        """
        multi_worker = is_multi_worker_environment()
        r = self._get_redis()

        if multi_worker:
            if r is None:
                logger.error("[NonceReplayCache] Multi-worker environment requires Redis for nonce replay protection, but Redis is unavailable. Failing closed.")
                raise ExecutionGatewaySecurityError(
                    "Distributed Redis replay protection is unavailable in multi-worker environment. Execution failed closed.",
                    code="DISTRIBUTED_REPLAY_CACHE_UNAVAILABLE"
                )
            try:
                key = f"analytics:grant_nonce:{nonce}"
                acquired = r.set(key, "1", ex=int(self.ttl), nx=True)
                return bool(acquired)
            except Exception as e:
                logger.error(f"[NonceReplayCache] Redis error during nonce consumption in multi-worker environment: {e}. Failing closed.")
                raise ExecutionGatewaySecurityError(
                    f"Distributed replay protection failed: {e}",
                    code="DISTRIBUTED_REPLAY_CACHE_ERROR"
                )

        # Local Single-Worker Development Mode:
        if r is not None:
            try:
                key = f"analytics:grant_nonce:{nonce}"
                acquired = r.set(key, "1", ex=int(self.ttl), nx=True)
                return bool(acquired)
            except Exception as e:
                logger.warning(f"[NonceReplayCache] Redis error during nonce consumption in local mode: {e}. Falling back to memory.")

        now = time.monotonic()
        self._prune(now)

        if nonce in self._nonces:
            return False  # Replay attack

        self._nonces[nonce] = now + self.ttl
        if len(self._nonces) > self.max_size:
            self._nonces.popitem(last=False)
        return True

    def _prune(self, now: float) -> None:
        expired = [n for n, exp in self._nonces.items() if exp < now]
        for n in expired:
            del self._nonces[n]


# Global gateway nonce cache
_replay_cache = NonceReplayCache()


class VirtualExecutionGateway:
    """
    Executes pre-authorized analytical queries against workspace-isolated SQLite databases.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        secret_key: Optional[str] = None,
        query_timeout: Optional[float] = None,
        max_rows: int = 50,
        replay_cache: Optional[NonceReplayCache] = None,
    ):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.storage_dir = storage_dir or os.path.join(repo_root, "storage", "excel_databases")
        self.secret_key = secret_key or os.getenv("ANALYTICS_SECRET_KEY") or os.getenv("API_KEY") or "fallback_analytics_secret"
        self.query_timeout = query_timeout or getattr(config, "ANALYTICS_QUERY_TIMEOUT_SECONDS", 15.0)
        self.max_rows = max_rows
        self.formatter = AnalyticsFormatter()
        self.replay_cache = replay_cache or _replay_cache

    def execute_authorized_query(self, req: AnalyticsExecuteRequest) -> AnalyticsExecutionResponse:
        """
        Main execution entry point for Stage C.
        """
        t_start = time.perf_counter()
        grant = req.authorization_grant

        # 1. Verify Contract Version
        if req.contract_version != CONTRACT_VERSION_3_0:
            raise ExecutionGatewaySecurityError(
                f"Unsupported contract version '{req.contract_version}'. Expected '{CONTRACT_VERSION_3_0}'.",
                code="VERSION_MISMATCH"
            )

        # 2. Verify Grant Scope matches Request
        if grant.workspace_id != req.workspace_id:
            raise ExecutionGatewaySecurityError(
                f"Authorization grant workspace mismatch: grant={grant.workspace_id}, request={req.workspace_id}",
                code="WORKSPACE_MISMATCH"
            )

        if grant.generation_id != req.generation_id:
            raise ExecutionGatewaySecurityError(
                f"Authorization grant generation mismatch: grant={grant.generation_id}, request={req.generation_id}",
                code="STALE_GENERATION"
            )

        # 3. Verify Query Fingerprint (Integrity Check)
        expected_fingerprint = PreparedQueryArtifact.compute_fingerprint(
            sql=req.sql,
            params=req.params,
            workspace_id=req.workspace_id,
            generation_id=req.generation_id,
        )

        if req.query_fingerprint != expected_fingerprint or grant.query_fingerprint != expected_fingerprint:
            logger.critical(
                "[ExecutionGateway] Query tampering detected! Computed=%s, Req=%s, Grant=%s",
                expected_fingerprint, req.query_fingerprint, grant.query_fingerprint
            )
            raise ExecutionGatewaySecurityError(
                "Query integrity violation: SQL or parameters have been modified after authorization.",
                code="INTEGRITY_VIOLATION"
            )

        # 4. Verify Cryptographic Signature
        if not grant.verify_signature(self.secret_key):
            logger.warning("[ExecutionGateway] Invalid grant signature for workspace %s", req.workspace_id)
            raise ExecutionGatewaySecurityError("Invalid authorization grant signature.", code="INVALID_SIGNATURE")

        # 5. Check Expiry
        if grant.is_expired():
            raise ExecutionGatewaySecurityError("Authorization grant has expired.", code="GRANT_EXPIRED")

        # 6. Single-Use Nonce Check (Replay Prevention)
        if not self.replay_cache.consume(grant.nonce):
            logger.warning("[ExecutionGateway] Replay attack detected for nonce %s", grant.nonce)
            raise ExecutionGatewaySecurityError("Authorization grant has already been used (replay rejected).", code="REPLAY_DETECTED")

        # 7. Locate Workspace-Isolated Virtual SQLite DB
        db_path = self._resolve_virtual_db_path(req.workspace_id, req.dataset_id)
        if not os.path.exists(db_path):
            raise ExecutionGatewaySecurityError(
                f"Virtual database for workspace {req.workspace_id} and dataset {req.dataset_id} not found.",
                code="DATABASE_NOT_FOUND"
            )

        # 8. Execute Parameterized Query in Read-Only Sandbox
        rows, columns = self._execute_sqlite(db_path=db_path, sql=req.sql, params=req.params)

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 9. Format Normalized Report (BDT standard)
        report = self._format_normalized_report(req.dataset_id, rows, columns, elapsed_ms)

        return AnalyticsExecutionResponse(
            success=True,
            contract_version=CONTRACT_VERSION_3_0,
            request_id=req.request_id,
            workspace_id=req.workspace_id,
            dataset_id=req.dataset_id,
            generation_id=req.generation_id,
            rows=rows,
            columns=columns,
            row_count=len(rows),
            report=report,
            latency_ms=elapsed_ms,
        )

    def _resolve_virtual_db_path(self, workspace_id: int, dataset_id: str) -> str:
        """Finds database file matching ws_{workspace_id}_{dataset_id}.sqlite."""
        # Sanitize dataset_id to prevent path traversal
        clean_dataset = "".join(c for c in dataset_id if c.isalnum() or c in ("-", "_"))[:64]
        exact_path = os.path.join(self.storage_dir, f"ws_{workspace_id}_{clean_dataset}.sqlite")
        if os.path.exists(exact_path):
            return exact_path

        # If clean_dataset is shorter or file was named with hash prefix, search storage_dir strictly with workspace_id prefix
        prefix = f"ws_{workspace_id}_"
        if os.path.exists(self.storage_dir):
            for fname in os.listdir(self.storage_dir):
                if fname.startswith(prefix) and fname.endswith(".sqlite"):
                    if clean_dataset in fname:
                        return os.path.join(self.storage_dir, fname)

        return exact_path

    def _execute_sqlite(self, db_path: str, sql: str, params: List[Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Executes read-only parameterized SQL query with progress handler timeout.
        """
        try:
            db_uri = f"{Path(os.path.abspath(db_path)).as_uri()}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        except Exception:
            conn = sqlite3.connect(db_path, timeout=5.0)

        conn.row_factory = sqlite3.Row
        deadline = time.monotonic() + self.query_timeout
        timed_out = False

        def progress_handler() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        conn.set_progress_handler(progress_handler, 1000)

        try:
            # Enforce read-only pragma
            conn.execute("PRAGMA query_only = ON;")
            cursor = conn.cursor()
            cursor.execute(sql, params or [])
            results = cursor.fetchall()
            columns = list(results[0].keys()) if results else []
            rows = [dict(r) for r in results[:self.max_rows]]
            return rows, columns
        except sqlite3.OperationalError as e:
            if timed_out or "interrupted" in str(e).lower():
                raise TimeoutError(f"Database query exceeded the {self.query_timeout}s timeout limit.") from e
            raise
        finally:
            try:
                conn.set_progress_handler(None, 0)
            except Exception:
                pass
            conn.close()

    def _format_normalized_report(self, dataset_name: str, rows: List[Dict[str, Any]], columns: List[str], elapsed_ms: float) -> str:
        """Formats tabular rows into standard BDT analytics markdown."""
        if not rows:
            return "📊 **বিশ্লেষণ ফলাফল**: প্রদত্ত শর্ত অনুযায়ী কোনো রেকর্ড পাওয়া যায়নি।"

        lines = [
            f"📊 **ব্যবসায়িক অ্যানালিটিক্স রিপোর্ট** (`{dataset_name}`)",
            f"• মোট ফলাফল: {len(rows)} টি রেকর্ড (সময়: {elapsed_ms}ms)",
            "----------------------------------------"
        ]

        # If scalar single value
        if len(rows) == 1 and len(columns) == 1:
            col = columns[0]
            val = rows[0][col]
            formatted_val = self._format_value(col, val)
            lines.append(f"• **{col}**: {formatted_val}")
            return "\n".join(lines)

        # Build clean Markdown table (up to 10 preview rows)
        preview_rows = rows[:10]
        header_line = "| " + " | ".join(columns) + " |"
        sep_line = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.append(header_line)
        lines.append(sep_line)

        for r in preview_rows:
            row_vals = [str(self._format_value(c, r.get(c))) for c in columns]
            lines.append("| " + " | ".join(row_vals) + " |")

        if len(rows) > 10:
            lines.append(f"\n*(আরও {len(rows) - 10} টি রেকর্ড অন্তর্ভুক্ত রয়েছে)*")

        return "\n".join(lines)

    @staticmethod
    def _format_value(col: str, val: Any) -> str:
        if val is None:
            return "-"
        if isinstance(val, (int, float)):
            lower_col = col.lower()
            if any(k in lower_col for k in ("price", "cost", "amount", "salary", "due", "revenue", "sales", "total", "fee", "charge")):
                return f"৳{val:,.2f}"
            return f"{val:,}" if isinstance(val, int) else f"{val:,.2f}"
        return str(val)


# Global gateway instance
virtual_execution_gateway = VirtualExecutionGateway()

