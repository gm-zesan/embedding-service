"""
Domain-Agnostic Semantic Schema Definitions and Profiler.
Provides metadata models for representing dataset purpose, entity types,
extensible column semantic roles, units, candidate relationships, and schema fingerprints
without hardcoding a specific business domain or e-commerce lexicon.
"""

import re
import json
import hashlib
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ColumnSemanticRole(str, Enum):
    """Extensible semantic roles that columns can fulfill across arbitrary business domains."""
    IDENTIFIER = "identifier"        # Unique keys, IDs, codes (e.g. employee_id, student_id, batch_no)
    ENTITY_NAME = "entity_name"      # Names/labels for primary entities (e.g. client_name, course_title)
    MEASURE = "measure"              # Continuous numerical metric (e.g. billable_hours, yield_pct, temperature)
    CURRENCY = "currency"            # Monetary numerical value (e.g. hourly_rate, total_cost, budget, fee)
    QUANTITY = "quantity"            # Count or discrete integer amounts (e.g. units_produced, student_count)
    DATE_TIME = "date_time"          # Timestamp, calendar date, semester, year (e.g. visit_date, start_time)
    STATUS = "status"                # State, progress stage, letter grade (e.g. milestone_status, grade, pass_fail)
    CATEGORY = "category"            # Discrete groupings (e.g. department, trial_arm, machine_type)
    FREE_TEXT = "free_text"          # Unstructured narrative, descriptions, feedback (e.g. notes, clinical_summary)
    ATTRIBUTE = "attribute"          # Secondary properties (e.g. email, phone, location, color, version)


class ColumnSemanticDescriptor(BaseModel):
    """Rich semantic description of a single column."""
    name: str
    physical_type: str = "TEXT"      # INTEGER, REAL, TEXT, DATE
    semantic_role: ColumnSemanticRole = ColumnSemanticRole.ATTRIBUTE
    entity_type: Optional[str] = None  # e.g. "student", "consultant", "patient", "project", "batch", "product"
    business_meaning: Optional[str] = None # Plain-language explanation
    unit: Optional[str] = None       # "USD", "BDT", "hours", "kg", "celsius", "%", "credits"
    is_nullable: bool = False
    uniqueness_ratio: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class DatasetSemanticSchema(BaseModel):
    """
    Domain-agnostic representation of tabular dataset semantics.
    Supports any industry: healthcare, education, legal, manufacturing, consulting, e-commerce, etc.
    """
    dataset_id: str
    domain: str = "general_business" # e.g. "education", "healthcare", "consulting", "manufacturing", "e_commerce"
    purpose: str                     # Natural language summary of dataset's role
    primary_entity: Optional[str] = None # Central noun: "student_grade", "consulting_timesheet", "production_batch"
    columns: Dict[str, ColumnSemanticDescriptor] = Field(default_factory=dict)
    measures: List[str] = Field(default_factory=list)      # Quantifiable measures available for aggregations
    dimensions: List[str] = Field(default_factory=list)    # Categorical fields available for filtering/grouping
    date_fields: List[str] = Field(default_factory=list)   # Temporal columns
    candidate_relationships: List[Dict[str, str]] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    schema_fingerprint: str           # SHA-256 hash of columns, types, and shape
    confidence: float = 1.0
    unresolved_ambiguity: List[str] = Field(default_factory=list)
    interpreter_version: str = "2.1.0"


class ColumnProfile(BaseModel):
    """Statistical and structural profile of a column computed deterministically without saving raw rows."""
    name: str
    physical_type: str
    null_count: int = 0
    null_ratio: float = 0.0
    unique_count: int = 0
    uniqueness_ratio: float = 0.0
    sample_values: List[str] = Field(default_factory=list) # Maximum 3 bounded, sanitized representative values
    inferred_date_pattern: Optional[str] = None
    min_num: Optional[float] = None
    max_num: Optional[float] = None


class DatasetProfile(BaseModel):
    """Sanitized dataset structural profile used as input to semantic interpreters."""
    dataset_name: str
    sheet_name: Optional[str] = None
    original_filename: Optional[str] = None
    workspace_id: Optional[int] = None
    row_count: int = 0
    column_profiles: Dict[str, ColumnProfile] = Field(default_factory=dict)
    schema_fingerprint: str


def compute_schema_fingerprint(
    columns: List[str],
    data_types: Dict[str, str],
    row_count: int = 0,
) -> str:
    """Computes a stable deterministic fingerprint of schema structure."""
    sorted_items = sorted([(c, data_types.get(c, "TEXT")) for c in columns])
    # Group row count by magnitude (e.g. 0-10, 10-100, 100-1000, 1000+) to prevent minor row inserts from busting cache
    magnitude = len(str(row_count))
    payload = json.dumps({"schema": sorted_items, "mag": magnitude}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def profile_dataset_columns(
    columns: List[str],
    data_types: Dict[str, str],
    raw_sample_data: Optional[List[Dict[str, Any]]] = None,
    row_count: int = 0,
    dataset_name: str = "",
    sheet_name: Optional[str] = None,
    original_filename: Optional[str] = None,
    workspace_id: Optional[int] = None,
) -> DatasetProfile:
    """
    Profiles tabular data deterministically.
    Extracts column nullability, uniqueness, data type patterns, and strictly bounded sanitized samples.
    Avoids storing raw user rows or sensitive full-row records. Redacts PII and credentials.
    """
    samples = raw_sample_data or []
    col_profiles: Dict[str, ColumnProfile] = {}

    for col in columns:
        col_type = data_types.get(col, "TEXT").upper()
        c_lower = col.lower()
        values = [row.get(col) for row in samples if col in row and row.get(col) is not None]
        null_count = len(samples) - len(values)
        null_ratio = round(null_count / max(1, len(samples)), 3)

        # Exclude secrets/credentials from sample values entirely
        is_secret_col = any(k in c_lower for k in ("password", "secret", "token", "api_key", "pin", "ssn", "nid", "cvv"))
        if is_secret_col:
            col_profiles[col] = ColumnProfile(
                name=col,
                physical_type=col_type,
                null_count=null_count,
                null_ratio=null_ratio,
                unique_count=len(values),
                uniqueness_ratio=1.0 if values else 0.0,
                sample_values=["[REDACTED_SECRET]"],
            )
            continue

        # Distinct sanitized string values (bounded to 3 samples max, strings truncated to 50 chars)
        clean_distinct: List[str] = []
        seen = set()
        for v in values:
            v_str = str(v).strip()
            # Redact credit cards or 16-digit national IDs
            if re.match(r"^\d{16}$", v_str) or re.match(r"^\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}$", v_str):
                v_str = "XXXX-XXXX-XXXX-" + re.sub(r"\D", "", v_str)[-4:]
            # Redact email addresses
            elif re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", v_str):
                v_str = "[EMAIL_REDACTED]"
            # Redact phone numbers (international or Bangladeshi 11-digit)
            elif re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", v_str) or re.match(r"^01[3-9]\d{8}$", v_str):
                v_str = "[PHONE_REDACTED]"
            elif len(v_str) > 50:
                v_str = v_str[:47] + "..."

            if v_str not in seen:
                seen.add(v_str)
                clean_distinct.append(v_str)
            if len(clean_distinct) >= 3:
                break

        unique_count = len(seen)
        uniqueness_ratio = round(unique_count / max(1, len(values)), 3) if values else 0.0

        # Numeric stats if available
        min_num, max_num = None, None
        if col_type in ("INTEGER", "REAL", "NUMERIC"):
            num_vals = []
            for v in values:
                try:
                    num_vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            if num_vals:
                min_num = min(num_vals)
                max_num = max(num_vals)

        # Date pattern detection
        inferred_date = None
        if col_type == "DATE" or "date" in col.lower() or "time" in col.lower():
            for v in clean_distinct:
                if re.match(r"^\d{4}-\d{2}-\d{2}", str(v)):
                    inferred_date = "YYYY-MM-DD"
                    break
                elif re.match(r"^\d{2}/\d{2}/\d{4}", str(v)):
                    inferred_date = "MM/DD/YYYY"
                    break

        col_profiles[col] = ColumnProfile(
            name=col,
            physical_type=col_type,
            null_count=null_count,
            null_ratio=null_ratio,
            unique_count=unique_count,
            uniqueness_ratio=uniqueness_ratio,
            sample_values=clean_distinct,
            inferred_date_pattern=inferred_date,
            min_num=min_num,
            max_num=max_num,
        )

    fingerprint = compute_schema_fingerprint(columns, data_types, row_count)

    return DatasetProfile(
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        original_filename=original_filename,
        workspace_id=workspace_id,
        row_count=row_count,
        column_profiles=col_profiles,
        schema_fingerprint=fingerprint,
    )
