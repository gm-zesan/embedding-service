"""
Phase 3.x Generic Semantic Analytics Engine - Semantic Registries & Capability Matrix.
Provides central catalog for measures, dimensions, relationships, derived metrics,
grain analysis rules, and typed field validation.
"""

from typing import Dict, List, Optional, Tuple, Any, Set, Union
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import re

from app.analytics.models_v2 import (
    DomainEnum,
    AggregationType,
    OperatorType,
    CardinalityType,
    FanoutRisk,
    RelationshipSpec,
    DerivedMetricType,
    FilterSpec,
    SemanticQueryPlan,
)


class MeasureDefinition(BaseModel):
    """Metadata for registered business measures."""
    name: str
    domain: DomainEnum
    source_table: str
    source_column: str
    default_aggregation: AggregationType
    allowed_aggregations: List[AggregationType]
    default_scope_filter: Optional[str] = None
    data_type: str  # "currency", "integer", "decimal"
    description: str


class DimensionDefinition(BaseModel):
    """Metadata for registered business dimensions."""
    name: str
    domain: DomainEnum
    source_table: str
    id_column: str
    label_column: str
    data_type: str  # "string", "date", "integer"
    description: str


class RelationshipDefinition(BaseModel):
    """Formal schema relationship graph node and edge metadata."""
    name: str
    source_entity: str
    target_entity: str
    source_table: str
    target_table: str
    source_key: str
    target_key: str
    cardinality: CardinalityType
    semantic_role: str
    fanout_risk: FanoutRisk
    join_clause: str
    requires_pre_aggregation: bool = False


class DerivedMetricDefinition(BaseModel):
    """Metadata for compound derived metrics computed across tables or grains."""
    type: DerivedMetricType
    required_domains: List[DomainEnum]
    base_measures: List[str]
    formula_description: str
    canonical_formula: str
    grain_strategy: str
    optimized_single_table_expr: Optional[str] = None


# =====================================================================
# 1. MEASURE REGISTRY
# =====================================================================
MEASURE_REGISTRY: Dict[str, MeasureDefinition] = {
    "sales_amount": MeasureDefinition(
        name="sales_amount",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        source_column="net_amount",
        default_aggregation=AggregationType.SUM,
        allowed_aggregations=[
            AggregationType.SUM,
            AggregationType.AVG,
            AggregationType.MIN,
            AggregationType.MAX,
        ],
        default_scope_filter="status = 'completed'",
        data_type="currency",
        description="Total net sales amount for completed orders",
    ),
    "order_count": MeasureDefinition(
        name="order_count",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[
            AggregationType.COUNT,
            AggregationType.COUNT_DISTINCT,
        ],
        default_scope_filter="status = 'completed'",
        data_type="integer",
        description="Count of completed orders",
    ),
    "collection_amount": MeasureDefinition(
        name="collection_amount",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_payments",
        source_column="amount",
        default_aggregation=AggregationType.SUM,
        allowed_aggregations=[
            AggregationType.SUM,
            AggregationType.AVG,
            AggregationType.MIN,
            AggregationType.MAX,
        ],
        default_scope_filter=None,
        data_type="currency",
        description="Total cash collected from customer payments",
    ),
    "payment_count": MeasureDefinition(
        name="payment_count",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_payments",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[
            AggregationType.COUNT,
            AggregationType.COUNT_DISTINCT,
        ],
        default_scope_filter=None,
        data_type="integer",
        description="Count of individual payment transactions",
    ),
    "product_quantity": MeasureDefinition(
        name="product_quantity",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_order_items",
        source_column="quantity",
        default_aggregation=AggregationType.SUM,
        allowed_aggregations=[
            AggregationType.SUM,
            AggregationType.AVG,
            AggregationType.MAX,
            AggregationType.MIN,
        ],
        default_scope_filter=None,
        data_type="integer",
        description="Total quantity of products sold",
    ),
    "product_revenue": MeasureDefinition(
        name="product_revenue",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_order_items",
        source_column="subtotal",
        default_aggregation=AggregationType.SUM,
        allowed_aggregations=[
            AggregationType.SUM,
            AggregationType.AVG,
            AggregationType.MAX,
            AggregationType.MIN,
        ],
        default_scope_filter=None,
        data_type="currency",
        description="Total line item revenue from products",
    ),
    "active_assignment_count": MeasureDefinition(
        name="active_assignment_count",
        domain=DomainEnum.DUE_ASSIGNMENT,
        source_table="analytics_due_assignments",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[AggregationType.COUNT],
        default_scope_filter="status IN ('assigned', 'in_progress')",
        data_type="integer",
        description="Total count of active due recovery assignments",
    ),
    "average_order_value": MeasureDefinition(
        name="average_order_value",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        source_column="net_amount",
        default_aggregation=AggregationType.AVG,
        allowed_aggregations=[AggregationType.AVG],
        default_scope_filter="status = 'completed'",
        data_type="currency",
        description="Average order value (canonical derived: SUM(sales_amount) / COUNT(order_count))",
    ),
    "due_amount": MeasureDefinition(
        name="due_amount",
        domain=DomainEnum.DUE,
        source_table="analytics_orders",
        source_column="net_amount",
        default_aggregation=AggregationType.SUM,
        allowed_aggregations=[AggregationType.SUM],
        default_scope_filter=None,
        data_type="currency",
        description="Alias for outstanding due amount",
    ),
}


# =====================================================================
# 2. DIMENSION REGISTRY
# =====================================================================
DIMENSION_REGISTRY: Dict[str, DimensionDefinition] = {
    "salesperson": DimensionDefinition(
        name="salesperson",
        domain=DomainEnum.SALES,
        source_table="analytics_salespersons",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Sales representative or field officer name",
    ),
    "customer": DimensionDefinition(
        name="customer",
        domain=DomainEnum.SALES,
        source_table="analytics_customers",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Customer or business client name",
    ),
    "product": DimensionDefinition(
        name="product",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Product catalog item name",
    ),
    "category": DimensionDefinition(
        name="category",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        id_column="category",
        label_column="category",
        data_type="string",
        description="Product classification category",
    ),
    "payment_method": DimensionDefinition(
        name="payment_method",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_payments",
        id_column="payment_method",
        label_column="payment_method",
        data_type="string",
        description="Payment channel: cash, bank, bkash, nagad, etc.",
    ),
    "collector": DimensionDefinition(
        name="collector",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_salespersons",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Sales representative collecting payment",
    ),
    "status": DimensionDefinition(
        name="status",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        id_column="status",
        label_column="status",
        data_type="string",
        description="Order status: completed, pending, cancelled",
    ),
    "assignment_status": DimensionDefinition(
        name="assignment_status",
        domain=DomainEnum.DUE_ASSIGNMENT,
        source_table="analytics_due_assignments",
        id_column="status",
        label_column="status",
        data_type="string",
        description="Recovery task status: assigned, in_progress, collected, escalated",
    ),
    "assigned_collector": DimensionDefinition(
        name="assigned_collector",
        domain=DomainEnum.DUE_ASSIGNMENT,
        source_table="analytics_salespersons",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Assigned debt recovery agent",
    ),
    "order_date": DimensionDefinition(
        name="order_date",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        id_column="order_date",
        label_column="order_date",
        data_type="date",
        description="Date when order was placed",
    ),
    "order_month": DimensionDefinition(
        name="order_month",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        id_column="order_date",
        label_column="order_date",
        data_type="string",
        description="Month when order was placed (YYYY-MM)",
    ),
    "collected_date": DimensionDefinition(
        name="collected_date",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_payments",
        id_column="collected_at",
        label_column="collected_at",
        data_type="date",
        description="Date when payment was recorded",
    ),
}


# =====================================================================
# 3. RELATIONSHIP REGISTRY (Schema Join Graph & Fanout Governance)
# =====================================================================
RELATIONSHIP_REGISTRY: Dict[Tuple[str, str], RelationshipDefinition] = {
    ("orders", "customers"): RelationshipDefinition(
        name="orders_to_customers",
        source_entity="orders",
        target_entity="customers",
        source_table="analytics_orders",
        target_table="analytics_customers",
        source_key="customer_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="buyer",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_orders.customer_id = analytics_customers.id",
        requires_pre_aggregation=False,
    ),
    ("orders", "salespersons"): RelationshipDefinition(
        name="orders_to_salespersons",
        source_entity="orders",
        target_entity="salespersons",
        source_table="analytics_orders",
        target_table="analytics_salespersons",
        source_key="salesperson_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="seller",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_orders.salesperson_id = analytics_salespersons.id",
        requires_pre_aggregation=False,
    ),
    ("orders", "order_items"): RelationshipDefinition(
        name="orders_to_order_items",
        source_entity="orders",
        target_entity="order_items",
        source_table="analytics_orders",
        target_table="analytics_order_items",
        source_key="id",
        target_key="order_id",
        cardinality=CardinalityType.ONE_TO_MANY,
        semantic_role="line_items",
        fanout_risk=FanoutRisk.HIGH,
        join_clause="analytics_orders.id = analytics_order_items.order_id",
        requires_pre_aggregation=True,
    ),
    ("order_items", "products"): RelationshipDefinition(
        name="order_items_to_products",
        source_entity="order_items",
        target_entity="products",
        source_table="analytics_order_items",
        target_table="analytics_products",
        source_key="product_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="catalog_item",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_order_items.product_id = analytics_products.id",
        requires_pre_aggregation=False,
    ),
    ("payments", "customers"): RelationshipDefinition(
        name="payments_to_customers",
        source_entity="payments",
        target_entity="customers",
        source_table="analytics_payments",
        target_table="analytics_customers",
        source_key="customer_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="payer",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_payments.customer_id = analytics_customers.id",
        requires_pre_aggregation=False,
    ),
    ("payments", "salespersons"): RelationshipDefinition(
        name="payments_to_salespersons",
        source_entity="payments",
        target_entity="salespersons",
        source_table="analytics_payments",
        target_table="analytics_salespersons",
        source_key="salesperson_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="collector",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_payments.salesperson_id = analytics_salespersons.id",
        requires_pre_aggregation=False,
    ),
    ("orders", "payments"): RelationshipDefinition(
        name="orders_to_payments_cross_grain",
        source_entity="orders",
        target_entity="payments",
        source_table="analytics_orders",
        target_table="analytics_payments",
        source_key="customer_id",
        target_key="customer_id",
        cardinality=CardinalityType.MANY_TO_MANY,
        semantic_role="cross_domain_balance",
        fanout_risk=FanoutRisk.EXTREME_CARTESIAN,
        join_clause="orders_cte.customer_id = payments_cte.customer_id",
        requires_pre_aggregation=True,
    ),
    ("due_assignments", "customers"): RelationshipDefinition(
        name="due_assignments_to_customers",
        source_entity="due_assignments",
        target_entity="customers",
        source_table="analytics_due_assignments",
        target_table="analytics_customers",
        source_key="customer_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="due_debtor",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_due_assignments.customer_id = analytics_customers.id",
        requires_pre_aggregation=False,
    ),
    ("due_assignments", "salespersons"): RelationshipDefinition(
        name="due_assignments_to_salespersons",
        source_entity="due_assignments",
        target_entity="salespersons",
        source_table="analytics_due_assignments",
        target_table="analytics_salespersons",
        source_key="assigned_salesperson_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="recovery_assignee",
        fanout_risk=FanoutRisk.LOW,
        join_clause="analytics_due_assignments.assigned_salesperson_id = analytics_salespersons.id",
        requires_pre_aggregation=False,
    ),
}


# =====================================================================
# 4. DERIVED METRIC REGISTRY
# =====================================================================
DERIVED_METRIC_REGISTRY: Dict[DerivedMetricType, DerivedMetricDefinition] = {
    DerivedMetricType.OUTSTANDING_DUE: DerivedMetricDefinition(
        type=DerivedMetricType.OUTSTANDING_DUE,
        required_domains=[DomainEnum.DUE, DomainEnum.SALES, DomainEnum.PAYMENTS],
        base_measures=["sales_amount", "collection_amount"],
        formula_description="Total net completed orders minus total collections",
        canonical_formula="SUM(orders.net_amount) - SUM(payments.amount)",
        grain_strategy="pre_aggregated_cte_merge",
        optimized_single_table_expr=None,
    ),
    DerivedMetricType.COLLECTION_RATE: DerivedMetricDefinition(
        type=DerivedMetricType.COLLECTION_RATE,
        required_domains=[DomainEnum.PAYMENTS, DomainEnum.SALES, DomainEnum.DUE],
        base_measures=["collection_amount", "sales_amount"],
        formula_description="Total collections divided by total sales multiplied by 100",
        canonical_formula="(SUM(payments.amount) / SUM(orders.net_amount)) * 100",
        grain_strategy="cross_domain_ratio",
        optimized_single_table_expr=None,
    ),
    DerivedMetricType.AVERAGE_ORDER_VALUE: DerivedMetricDefinition(
        type=DerivedMetricType.AVERAGE_ORDER_VALUE,
        required_domains=[DomainEnum.SALES],
        base_measures=["sales_amount", "order_count"],
        formula_description="Total completed net sales divided by count of completed orders",
        canonical_formula="SUM(sales_amount) / COUNT(orders)",
        grain_strategy="ratio",
        optimized_single_table_expr="AVG(net_amount)",
    ),
    DerivedMetricType.PERIOD_DIFFERENCE: DerivedMetricDefinition(
        type=DerivedMetricType.PERIOD_DIFFERENCE,
        required_domains=[DomainEnum.SALES, DomainEnum.PAYMENTS],
        base_measures=[],
        formula_description="Value in current period minus value in previous period",
        canonical_formula="current_period_val - prior_period_val",
        grain_strategy="temporal_slice_difference",
        optimized_single_table_expr=None,
    ),
    DerivedMetricType.PERIOD_GROWTH_PERCENT: DerivedMetricDefinition(
        type=DerivedMetricType.PERIOD_GROWTH_PERCENT,
        required_domains=[DomainEnum.SALES, DomainEnum.PAYMENTS],
        base_measures=[],
        formula_description="Percentage growth from prior period: ((current - prior) / prior) * 100",
        canonical_formula="((current_val - prior_val) / prior_val) * 100",
        grain_strategy="temporal_growth_ratio",
        optimized_single_table_expr=None,
    ),
}


# =====================================================================
# 5. SEMANTIC CAPABILITY MATRIX
# =====================================================================
CAPABILITY_MATRIX: Dict[DomainEnum, Dict[str, Set[str]]] = {
    DomainEnum.SALES: {
        "measures": {
            "sales_amount",
            "order_count",
            "average_order_value",
        },
        "dimensions": {
            "salesperson",
            "customer",
            "product",
            "category",
            "order_date",
            "order_month",
            "status",
        },
        "group_by": {
            "salesperson",
            "customer",
            "product",
            "category",
            "order_date",
            "order_month",
        },
        "derived_metrics": {
            "average_order_value",
            "period_difference",
            "period_growth_percent",
        },
    },
    DomainEnum.PAYMENTS: {
        "measures": {
            "collection_amount",
            "payment_count",
        },
        "dimensions": {
            "collector",
            "salesperson",
            "customer",
            "payment_method",
            "collected_date",
        },
        "group_by": {
            "collector",
            "salesperson",
            "customer",
            "payment_method",
            "collected_date",
        },
        "derived_metrics": {
            "period_difference",
            "period_growth_percent",
        },
    },
    DomainEnum.DUE: {
        "measures": {
            "sales_amount",
            "collection_amount",
            "outstanding_due",
            "due_amount",
        },
        "dimensions": {
            "customer",
            "assigned_collector",
            "salesperson",
        },
        "group_by": {
            "customer",
            "assigned_collector",
            "salesperson",
        },
        "derived_metrics": {
            "outstanding_due",
            "collection_rate",
        },
    },
    DomainEnum.PRODUCT: {
        "measures": {
            "product_quantity",
            "product_revenue",
            "sales_amount",
        },
        "dimensions": {
            "product",
            "category",
            "salesperson",
            "customer",
        },
        "group_by": {
            "product",
            "category",
        },
        "derived_metrics": {
            "period_difference",
            "period_growth_percent",
        },
    },
    DomainEnum.DUE_ASSIGNMENT: {
        "measures": {
            "active_assignment_count",
        },
        "dimensions": {
            "customer",
            "assigned_collector",
            "status",
            "assignment_status",
        },
        "group_by": {
            "assigned_collector",
            "status",
            "assignment_status",
            "customer",
        },
        "derived_metrics": set(),
    },
}


# =====================================================================
# 6. FIELD DATA TYPE MAPPING (For Typed Filter Value Validation)
# =====================================================================
FIELD_TYPE_REGISTRY: Dict[str, str] = {
    # Numeric Measures
    "sales_amount": "numeric",
    "order_count": "numeric",
    "collection_amount": "numeric",
    "payment_count": "numeric",
    "product_quantity": "numeric",
    "product_revenue": "numeric",
    "active_assignment_count": "numeric",
    "outstanding_due": "numeric",
    "due_amount": "numeric",
    "average_order_value": "numeric",
    # String Dimensions
    "salesperson": "string",
    "customer": "string",
    "product": "string",
    "category": "string",
    "payment_method": "string",
    "collector": "string",
    "status": "string",
    "assignment_status": "string",
    "assigned_collector": "string",
    # Date Dimensions
    "order_date": "date",
    "collected_date": "date",
}


# =====================================================================
# 7. REGISTRY QUERY HELPERS & VALIDATION ENGINE
# =====================================================================
def get_measure(name: str) -> Optional[MeasureDefinition]:
    """Retrieve measure definition by registered token."""
    return MEASURE_REGISTRY.get(name)


def get_dimension(name: str) -> Optional[DimensionDefinition]:
    """Retrieve dimension definition by registered token."""
    return DIMENSION_REGISTRY.get(name)


def get_relationship(source_entity: str, target_entity: str) -> Optional[RelationshipDefinition]:
    """Retrieve relationship definition between two schema entities."""
    return (
        RELATIONSHIP_REGISTRY.get((source_entity, target_entity))
        or RELATIONSHIP_REGISTRY.get((target_entity, source_entity))
    )


def get_derived_metric(metric_type: Union[DerivedMetricType, str]) -> Optional[DerivedMetricDefinition]:
    """Retrieve derived metric definition."""
    if isinstance(metric_type, str):
        try:
            metric_type = DerivedMetricType(metric_type)
        except ValueError:
            return None
    return DERIVED_METRIC_REGISTRY.get(metric_type)


def get_fanout_risk(source_entity: str, target_entity: str) -> FanoutRisk:
    """Assess fanout risk between two entities. Defaults to HIGH if unspecified."""
    rel = get_relationship(source_entity, target_entity)
    if rel:
        return rel.fanout_risk
    return FanoutRisk.HIGH


def validate_filter_value_type(field: str, operator: OperatorType, value: Any) -> Tuple[bool, Optional[str]]:
    """
    Validates that a filter's value matches the registered semantic data type of the field.
    Prevents invalid queries (e.g. sales_amount > 'Hasan') from reaching the compiler.
    """
    expected_type = FIELD_TYPE_REGISTRY.get(field)
    if not expected_type:
        return False, f"Unknown filter field '{field}'. Not registered in FIELD_TYPE_REGISTRY."

    # Operator: IS NULL / NOT NULL doesn't need value validation
    if operator in (OperatorType.IS_NULL, OperatorType.NOT_NULL):
        return True, None

    # Helper: Check if scalar is numeric
    def is_scalar_numeric(v: Any) -> bool:
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, str):
            try:
                float(v)
                return True
            except ValueError:
                return False
        return False

    # 1. Numeric Field Validation
    if expected_type == "numeric":
        if operator == OperatorType.BETWEEN:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False, f"BETWEEN operator on '{field}' requires a 2-element list [start, end], got {value}"
            if not (is_scalar_numeric(value[0]) and is_scalar_numeric(value[1])):
                return False, f"BETWEEN operator on '{field}' requires numeric range boundaries, got {value}"
            return True, None

        if operator == OperatorType.IN:
            if not isinstance(value, (list, tuple)):
                return False, f"IN operator on '{field}' requires a list, got {type(value).__name__}"
            if not all(is_scalar_numeric(item) for item in value):
                return False, f"IN operator on numeric field '{field}' contains non-numeric items: {value}"
            return True, None

        # Scalar comparisons (=, !=, >, >=, <, <=)
        if not is_scalar_numeric(value):
            return False, f"Filter field '{field}' expects a numeric value, got '{value}' (type: {type(value).__name__})"
        return True, None

    # 2. String Field Validation
    if expected_type == "string":
        if operator == OperatorType.IN:
            if not isinstance(value, (list, tuple)):
                return False, f"IN operator on '{field}' requires a list, got {type(value).__name__}"
            return True, None

        if operator == OperatorType.BETWEEN:
            return False, f"BETWEEN operator is not supported for string field '{field}'"

        if isinstance(value, (list, tuple, dict)):
            return False, f"String field '{field}' requires scalar string value, got {type(value).__name__}"
        return True, None

    # 3. Date Field Validation
    if expected_type == "date":
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if operator == OperatorType.BETWEEN:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False, f"BETWEEN operator on '{field}' requires 2 date strings, got {value}"
            if not (isinstance(value[0], str) and isinstance(value[1], str)):
                return False, f"BETWEEN operator on '{field}' requires date strings 'YYYY-MM-DD', got {value}"
            return True, None

        if operator == OperatorType.IN:
            if not isinstance(value, (list, tuple)):
                return False, f"IN operator on '{field}' requires a list of dates"
            return True, None

        if not isinstance(value, str):
            return False, f"Date field '{field}' requires date string 'YYYY-MM-DD', got {value}"
        return True, None

    return True, None


def validate_plan_semantics(plan: SemanticQueryPlan) -> List[str]:
    """
    Validates complete SemanticQueryPlan against the Capability Matrix,
    checking domain compatibility, group_by grain, and filter types.
    Returns an empty list if valid, or a list of descriptive semantic errors.
    """
    errors: List[str] = []
    matrix = CAPABILITY_MATRIX.get(plan.domain)

    if not matrix:
        errors.append(f"Domain '{plan.domain}' is not registered in CapabilityMatrix.")
        return errors

    allowed_measures = matrix["measures"]
    allowed_dimensions = matrix["dimensions"]
    allowed_group_by = matrix["group_by"]
    allowed_derived = matrix["derived_metrics"]

    # 1. Validate Measures
    for m in plan.measures:
        if m.name not in allowed_measures:
            errors.append(
                f"Measure '{m.name}' is not allowed in domain '{plan.domain.value}'. "
                f"Allowed measures: {sorted(list(allowed_measures))}"
            )
        else:
            m_def = get_measure(m.name)
            if m_def and m.aggregation not in m_def.allowed_aggregations:
                errors.append(
                    f"Aggregation '{m.aggregation.value}' is not allowed for measure '{m.name}'. "
                    f"Allowed: {[a.value for a in m_def.allowed_aggregations]}"
                )

    # 2. Validate Dimensions
    for d in plan.dimensions:
        if d.name not in allowed_dimensions:
            errors.append(
                f"Dimension '{d.name}' is not allowed in domain '{plan.domain.value}'. "
                f"Allowed dimensions: {sorted(list(allowed_dimensions))}"
            )

    # 3. Validate Group By
    for g in plan.group_by:
        if g not in allowed_group_by:
            errors.append(
                f"Group by field '{g}' is not allowed in domain '{plan.domain.value}'. "
                f"Allowed groupings: {sorted(list(allowed_group_by))}"
            )

    # 4. Validate Derived Metrics
    for dm in plan.derived_metrics:
        if dm.type.value not in allowed_derived and dm.type != DerivedMetricType.OUTSTANDING_DUE:
            errors.append(
                f"Derived metric '{dm.type.value}' is not allowed in domain '{plan.domain.value}'. "
                f"Allowed derived metrics: {sorted(list(allowed_derived))}"
            )

    # 5. Validate Filters and Filter Value Types
    for f in plan.filters:
        # Field must be either in allowed dimensions or allowed measures
        if f.field not in allowed_dimensions and f.field not in allowed_measures:
            errors.append(
                f"Filter field '{f.field}' is not allowed in domain '{plan.domain.value}'."
            )
        else:
            is_valid, err = validate_filter_value_type(f.field, f.operator, f.value)
            if not is_valid and err:
                errors.append(err)

    return errors
