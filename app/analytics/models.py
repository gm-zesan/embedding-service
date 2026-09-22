"""
Phase 3.x Generic Semantic Analytics Engine - Data Contracts & Pydantic Models.
Defines strict semantic AST types, operator allowlists, typed filter validation,
and formal relationship specifications.
"""

from enum import Enum
from typing import List, Optional, Union, Literal, Any
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator, model_validator


class DomainEnum(str, Enum):
    """Authoritative business analytics execution domains."""
    SALES = "sales"
    PAYMENTS = "payments"
    DUE = "due"
    PRODUCT = "product"
    DUE_ASSIGNMENT = "due_assignment"


class AggregationType(str, Enum):
    """Allowed semantic aggregation functions."""
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT_DISTINCT = "count_distinct"


class OperatorType(str, Enum):
    """Allowed comparison and membership operators."""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "IN"
    BETWEEN = "BETWEEN"
    LIKE = "LIKE"
    IS_NULL = "IS NULL"
    NOT_NULL = "NOT NULL"


class CardinalityType(str, Enum):
    """Schema relationship cardinality."""
    MANY_TO_ONE = "many_to_one"
    ONE_TO_MANY = "one_to_many"
    ONE_TO_ONE = "one_to_one"
    MANY_TO_MANY = "many_to_many"


class FanoutRisk(str, Enum):
    """
    Join risk categorization for Grain Analysis:
    - LOW: Dimension lookup (many_to_one), safe to join directly.
    - HIGH: Child table expansion (one_to_many), requires pre-aggregation.
    - EXTREME_CARTESIAN: Cross-grain event tables (e.g. orders x payments), direct join strictly forbidden.
    """
    LOW = "LOW"
    HIGH = "HIGH"
    EXTREME_CARTESIAN = "EXTREME_CARTESIAN"


class RelationshipSpec(BaseModel):
    """Formal schema relationship specification between two business entities."""
    source_entity: str
    target_entity: str
    source_key: str
    target_key: str
    cardinality: CardinalityType
    semantic_role: str
    fanout_risk: FanoutRisk
    requires_pre_aggregation: bool = False


DEFAULT_MEASURE_AGGREGATIONS = {
    "sales_amount": AggregationType.SUM,
    "order_count": AggregationType.COUNT,
    "collection_amount": AggregationType.SUM,
    "payment_count": AggregationType.COUNT,
    "product_quantity": AggregationType.SUM,
    "product_revenue": AggregationType.SUM,
    "active_assignment_count": AggregationType.COUNT,
    "average_order_value": AggregationType.AVG,
    "due_amount": AggregationType.SUM,
}


class MeasureSpec(BaseModel):
    """Semantic measure specification (e.g., sales_amount, collection_amount)."""
    name: str
    aggregation: Optional[AggregationType] = None
    alias: Optional[str] = None

    @model_validator(mode="after")
    def populate_default_aggregation(self) -> "MeasureSpec":
        if self.aggregation is None:
            self.aggregation = DEFAULT_MEASURE_AGGREGATIONS.get(self.name, AggregationType.SUM)
        return self


class DimensionSpec(BaseModel):
    """Semantic dimension specification (e.g., salesperson, customer, product)."""
    name: str
    alias: Optional[str] = None


# Strongly typed filter value union: primitives, numeric, or homogeneous collections
FilterValue = Union[str, int, float, bool, List[str], List[int], List[float]]


class FilterSpec(BaseModel):
    """
    Typed filter specification with operator validation and syntactic guardrails.
    Full type matching (e.g. checking if a numeric field receives a non-numeric string)
    is enforced by the SemanticValidator / CapabilityMatrix.
    """
    field: str
    operator: OperatorType = OperatorType.EQUALS
    value: Optional[FilterValue] = None

    @field_validator("operator", mode="before")
    @classmethod
    def normalize_operator(cls, v: Any) -> OperatorType:
        if isinstance(v, str):
            clean = v.strip().upper()
            for op in OperatorType:
                if op.value == clean or op.name == clean:
                    return op
        return v

    @model_validator(mode="after")
    def validate_operator_value_consistency(self) -> "FilterSpec":
        op = self.operator
        val = self.value

        # Null checks
        if op in (OperatorType.IS_NULL, OperatorType.NOT_NULL):
            # value is not required for IS NULL / NOT NULL
            return self

        if val is None:
            raise ValueError(f"Filter on '{self.field}' with operator '{op.value}' requires a non-null value.")

        # BETWEEN requires a 2-element collection
        if op == OperatorType.BETWEEN:
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"BETWEEN operator for '{self.field}' requires a 2-element list [start, end], got {val}")

        # IN requires a non-empty list
        if op == OperatorType.IN:
            if not isinstance(val, (list, tuple)) or len(val) == 0:
                raise ValueError(f"IN operator for '{self.field}' requires a non-empty list, got {val}")

        return self


class TimeRangeSpec(BaseModel):
    """Structured relative or absolute temporal filter."""
    type: Literal[
        "today",
        "yesterday",
        "last_7_days",
        "this_month",
        "last_month",
        "lifetime",
        "last_n_days",
        "custom_range",
    ]
    n_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @model_validator(mode="after")
    def validate_time_range_params(self) -> "TimeRangeSpec":
        if self.type == "last_n_days" and (self.n_days is None or self.n_days <= 0):
            raise ValueError("Time range type 'last_n_days' requires positive integer 'n_days'.")
        if self.type == "custom_range" and (not self.start_date or not self.end_date):
            raise ValueError("Time range type 'custom_range' requires both 'start_date' and 'end_date'.")
        return self


class OrderBySpec(BaseModel):
    """Sort specification for ranking and tabular outputs."""
    field: str
    direction: Literal["asc", "desc"] = "desc"


class DerivedMetricType(str, Enum):
    """Allowlisted compound derived metrics computed across tables or aggregate grains."""
    OUTSTANDING_DUE = "outstanding_due"
    COLLECTION_RATE = "collection_rate"
    AVERAGE_ORDER_VALUE = "average_order_value"
    PERIOD_DIFFERENCE = "period_difference"
    PERIOD_GROWTH_PERCENT = "period_growth_percent"


class DerivedMetricSpec(BaseModel):
    """Specification of an allowed compound business metric."""
    type: DerivedMetricType
    alias: Optional[str] = None
    target_dimension: Optional[str] = None  # e.g., "customer" for customer-level due balance


class SemanticQueryPlan(BaseModel):
    """
    Structured Semantic Query Plan (AST) generated by SemanticPlanner v2.
    Fully decoupled from physical schema, table names, SQL clauses, and workspace scope.
    """
    thought_process: Optional[str] = None
    domain: DomainEnum
    measures: List[MeasureSpec] = Field(default_factory=list)
    dimensions: List[DimensionSpec] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    time_range: Optional[TimeRangeSpec] = None
    order_by: List[OrderBySpec] = Field(default_factory=list)
    limit: Optional[int] = None
    derived_metrics: List[DerivedMetricSpec] = Field(default_factory=list)

    # Legacy intent metadata for backwards compatibility / telemetry only (zero compiler influence)
    legacy_intent: Optional[str] = None

    # Clarification / conversational guardrails
    needs_clarification: bool = False
    clarification_options: List[str] = Field(default_factory=list)
    clarification_message: Optional[str] = None

    # Security rejection handling (e.g. injection, prompt leakage attempt)
    is_security_rejection: bool = False
    rejection_reason: Optional[str] = None
