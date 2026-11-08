"""
Contract 3.0.0 - Enterprise AI Analytics Specification.
Defines versioned, strictly-typed Pydantic contracts between Laravel and Python
for Virtual Relational Database and Cross-Table SQL Synthesis.

Invariants:
1. LLM must never generate executable SQL.
2. Two-Stage Protocol: Stage A (Prepare) -> Stage B (Authorize) -> Stage C (Execute).
3. Cryptographically signed, single-use ExecutionAuthorizationGrant is required for execution.
4. Parameters are strictly bound (? placeholders), zero string interpolation.
5. Confirmed relationships (state == 'CONFIRMED') only.
"""

from __future__ import annotations

import hmac
import hashlib
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from app.analytics.models import (
    SemanticQueryPlan,
    DomainEnum,
    AggregationType,
    OperatorType,
    FilterSpec,
    MeasureSpec,
    DimensionSpec,
    OrderBySpec,
    TimeRangeSpec,
)
from app.analytics.semantic_schema import (
    SpreadsheetRelationshipGraph,
    DiscoveredRelationshipEdge,
    RelationshipLifecycleState,
    RelationshipCardinality,
)

CONTRACT_VERSION_3_0 = "3.0.0"


class VirtualColumnType(str, Enum):
    """Physical data types supported by virtual SQLite engine."""
    TEXT = "TEXT"
    NUMERIC = "NUMERIC"
    INTEGER = "INTEGER"
    REAL = "REAL"
    DATE = "DATE"


class VirtualColumnMeta(BaseModel):
    """Metadata for a virtual table column."""
    name: str                           # Safe sanitized identifier (e.g. col_0_product_name)
    original_header: str                # Raw human header (e.g. "Product Name", "পণ্যের নাম")
    physical_type: VirtualColumnType = VirtualColumnType.TEXT
    semantic_role: str = "attribute"
    sensitivity: str = "restricted"
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_nullable: bool = True


class VirtualTableMeta(BaseModel):
    """Metadata for a virtual table representing an uploaded worksheet."""
    name: str                           # Safe sanitized table identifier (e.g. tbl_orders)
    sheet_name: str                     # Original sheet name (e.g. "Orders", "অর্ডারসমূহ")
    row_count: int = 0
    columns: Dict[str, VirtualColumnMeta] = Field(default_factory=dict)
    primary_key: List[str] = Field(default_factory=list)


class DynamicSchemaCatalog(BaseModel):
    """Dynamic multi-sheet schema catalog for a dataset."""
    workspace_id: int
    dataset_id: str
    generation_id: str
    schema_fingerprint: str
    tables: Dict[str, VirtualTableMeta] = Field(default_factory=dict)
    relationships: Optional[SpreadsheetRelationshipGraph] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Stage A: Prepare Contract
# ---------------------------------------------------------------------------

class AnalyticsPrepareRequest(BaseModel):
    """
    Stage A Request: Sent by Laravel to Python to compile deterministic SQL
    without executing it.
    """
    contract_version: str = CONTRACT_VERSION_3_0
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: int = Field(..., gt=0, description="Mandatory workspace ID (>0)")
    dataset_id: str = Field(..., min_length=1, description="Dataset or document ID")
    generation_id: str = Field(..., min_length=1, description="Dataset generation UUID")
    schema_fingerprint: Optional[str] = None
    natural_query: Optional[str] = None
    query_plan: SemanticQueryPlan
    relationship_graph: Optional[SpreadsheetRelationshipGraph] = None
    source_descriptors: Optional[Dict[str, Any]] = Field(default_factory=dict)
    catalog: Optional[DynamicSchemaCatalog] = None

    @field_validator("contract_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v != CONTRACT_VERSION_3_0:
            raise ValueError(f"Contract version mismatch: expected '{CONTRACT_VERSION_3_0}', got '{v}'")
        return v


class PreparedQueryArtifact(BaseModel):
    """
    Stage A Response: Returned by Python to Laravel containing compiled parameterized SQL,
    parameter bindings, and dependency metadata for independent AST authorization.
    NOT EXECUTED AT THIS STAGE.
    """
    contract_version: str = CONTRACT_VERSION_3_0
    request_id: str
    workspace_id: int
    dataset_id: str
    generation_id: str
    query_fingerprint: str              # SHA-256(sql_template + params + workspace_id + generation_id)
    schema_fingerprint: str
    sql_template: str                   # Parameterized SQL with '?' placeholders
    params: List[Any] = Field(default_factory=list)
    referenced_tables: List[str] = Field(default_factory=list)
    referenced_columns: List[str] = Field(default_factory=list)
    confirmed_edge_ids: List[str] = Field(default_factory=list)
    compiler_version: str = "3.0.0"
    status: Literal["prepared"] = "prepared"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def compute_fingerprint(sql: str, params: List[Any], workspace_id: int, generation_id: str) -> str:
        canonical_str = f"{sql.strip()}|{repr(params)}|{workspace_id}|{generation_id.strip()}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage B: Authorization Grant Contract
# ---------------------------------------------------------------------------

class ExecutionAuthorizationGrant(BaseModel):
    """
    Stage B Output: Issued by Laravel after validating AST, column sensitivities,
    and workspace ownership. Cryptographically signed using HMAC-SHA256.
    """
    grant_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: int
    query_fingerprint: str
    generation_id: str
    authorized_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str                     # ISO8601, typically authorized_at + 60s
    nonce: str = Field(default_factory=lambda: uuid.uuid4().hex)
    signature: str                      # HMAC-SHA256(secret, workspace_id:query_fingerprint:generation_id:expires_at:nonce)

    @staticmethod
    def generate_signature(secret: str, workspace_id: int, query_fingerprint: str, generation_id: str, expires_at: str, nonce: str) -> str:
        message = f"{workspace_id}:{query_fingerprint.strip()}:{generation_id.strip()}:{expires_at.strip()}:{nonce.strip()}"
        return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_signature(self, secret: str) -> bool:
        expected = self.generate_signature(
            secret=secret,
            workspace_id=self.workspace_id,
            query_fingerprint=self.query_fingerprint,
            generation_id=self.generation_id,
            expires_at=self.expires_at,
            nonce=self.nonce,
        )
        return hmac.compare_digest(self.signature, expected)

    def is_expired(self) -> bool:
        try:
            # Parse ISO8601 with timezone
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > exp
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Stage C: Execute Contract
# ---------------------------------------------------------------------------

class AnalyticsExecuteRequest(BaseModel):
    """
    Stage C Request: Sent by Laravel to Python with the authorized prepared query
    and signed ExecutionAuthorizationGrant.
    """
    contract_version: str = CONTRACT_VERSION_3_0
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: int = Field(..., gt=0)
    dataset_id: str
    generation_id: str
    query_fingerprint: str
    sql: str
    params: List[Any] = Field(default_factory=list)
    authorization_grant: ExecutionAuthorizationGrant

    @field_validator("contract_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v != CONTRACT_VERSION_3_0:
            raise ValueError(f"Contract version mismatch: expected '{CONTRACT_VERSION_3_0}', got '{v}'")
        return v


class AnalyticsExecutionResponse(BaseModel):
    """
    Stage C Response: Normalized tabular results with BDT formatting and latency metrics.
    """
    success: bool
    contract_version: str = CONTRACT_VERSION_3_0
    request_id: str
    workspace_id: int
    dataset_id: str
    generation_id: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    report: str = ""                    # Formatted Markdown report in Bengali / BDT standard
    latency_ms: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None

