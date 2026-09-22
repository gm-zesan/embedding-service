"""
Phase 3.x Generic Semantic Analytics Engine - Semantic Validator v2.
Enforces Capability Matrix rules, Grain Analysis, Join Fanout Risk assessment,
and Pre-Aggregation Planning prior to SQL compilation.
"""

from typing import List, Optional, Tuple, Dict, Set
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from app.analytics.models_v2 import (
    DomainEnum,
    FanoutRisk,
    CardinalityType,
    DerivedMetricType,
    SemanticQueryPlan,
)
from app.analytics.registries import (
    CAPABILITY_MATRIX,
    get_measure,
    get_dimension,
    get_relationship,
    get_derived_metric,
    get_fanout_risk,
    validate_filter_value_type,
    validate_plan_semantics,
)


class GrainAnalysisResult(BaseModel):
    """Output of semantic grain analysis."""
    primary_grain: str  # "global_scalar", "salesperson", "customer", "product", "category", etc.
    requires_pre_aggregation: bool = False
    pre_aggregation_entities: List[str] = Field(default_factory=list)
    fanout_risk: FanoutRisk = FanoutRisk.LOW
    notes: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Validation report with grain analysis for Compiler v2."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    grain_analysis: Optional[GrainAnalysisResult] = None
    sanitized_plan: Optional[SemanticQueryPlan] = None


class SemanticValidator:
    """
    Validates SemanticQueryPlan instances before they reach the deterministic SQL compiler.
    Acts as a hard security and semantic boundary against hallucinated joins,
    incompatible measures/dimensions, and dangerous Cartesian fanouts.
    """

    @classmethod
    def analyze_grain(cls, plan: SemanticQueryPlan) -> GrainAnalysisResult:
        """
        Determines the output grain and checks if fanout-prone relationships
        (e.g., orders x order_items or orders x payments) mandate pre-aggregated CTEs.
        """
        notes: List[str] = []
        requires_pre_agg = False
        pre_agg_entities: List[str] = []
        max_fanout_risk = FanoutRisk.LOW

        # 1. Determine primary aggregation grain
        if not plan.group_by:
            primary_grain = "global_scalar"
            notes.append("No group_by specified: Query executes as single-row global scalar aggregation.")
        elif len(plan.group_by) == 1:
            primary_grain = plan.group_by[0]
            notes.append(f"Single-dimension grain: {primary_grain}")
        else:
            primary_grain = "_".join(plan.group_by)
            notes.append(f"Composite grain: {primary_grain}")

        # 2. Check for Cross-Domain Dynamic Balance (e.g. DUE domain: orders vs payments)
        is_due_domain = plan.domain == DomainEnum.DUE
        has_due_metric = any(
            dm.type == DerivedMetricType.OUTSTANDING_DUE for dm in plan.derived_metrics
        )
        if is_due_domain or has_due_metric:
            requires_pre_agg = True
            pre_agg_entities = ["orders", "payments"]
            max_fanout_risk = FanoutRisk.EXTREME_CARTESIAN
            notes.append(
                "Cross-domain balance detected (orders vs payments). "
                "Extreme Cartesian risk: Compiler must generate independent pre-aggregated CTEs "
                "before outer merging at customer/workspace grain."
            )
            return GrainAnalysisResult(
                primary_grain=primary_grain,
                requires_pre_aggregation=requires_pre_agg,
                pre_aggregation_entities=pre_agg_entities,
                fanout_risk=max_fanout_risk,
                notes=notes,
            )

        # 3. Check for Order Items (Line Items) joined to Orders
        has_product_measure = any(
            m.name in ("product_quantity", "product_revenue") for m in plan.measures
        )
        has_order_level_measure = any(
            m.name in ("sales_amount", "order_count") for m in plan.measures
        )
        if has_product_measure and has_order_level_measure:
            requires_pre_agg = True
            pre_agg_entities = ["order_items"]
            max_fanout_risk = FanoutRisk.HIGH
            notes.append(
                "Query references both order-level and order-item-level measures. "
                "One-to-many relationship creates high fanout risk: "
                "Order items must be pre-aggregated to order grain before joining."
            )

        return GrainAnalysisResult(
            primary_grain=primary_grain,
            requires_pre_aggregation=requires_pre_agg,
            pre_aggregation_entities=pre_agg_entities,
            fanout_risk=max_fanout_risk,
            notes=notes,
        )

    @classmethod
    def validate_plan(cls, plan: SemanticQueryPlan) -> ValidationResult:
        """Alias for validate."""
        return cls.validate(plan)

    @classmethod
    def validate(cls, plan: SemanticQueryPlan) -> ValidationResult:
        """
        Complete validation entrypoint.
        Executes schema compatibility checks, operator/type checks, and grain analysis.
        """
        # Guard: If plan is already flagged for clarification or security rejection, preserve it
        if plan.needs_clarification:
            return ValidationResult(
                is_valid=True,
                errors=[],
                sanitized_plan=plan,
            )

        if plan.is_security_rejection:
            return ValidationResult(
                is_valid=False,
                errors=[plan.rejection_reason or "Security rejection triggered."],
                sanitized_plan=plan,
            )

        # 1. Base semantic validation against CapabilityMatrix
        semantic_errors = validate_plan_semantics(plan)
        if semantic_errors:
            return ValidationResult(
                is_valid=False,
                errors=semantic_errors,
                sanitized_plan=plan,
            )

        # 2. Must request at least one measure, derived metric, or dimension/filter
        if not plan.measures and not plan.derived_metrics and not plan.dimensions and not plan.filters:
            return ValidationResult(
                is_valid=False,
                errors=["Semantic query plan must request at least one measure, derived metric, or dimension/filter."],
                sanitized_plan=plan,
            )

        # 3. Grain Analysis
        grain_analysis = cls.analyze_grain(plan)

        return ValidationResult(
            is_valid=True,
            errors=[],
            grain_analysis=grain_analysis,
            sanitized_plan=plan,
        )
