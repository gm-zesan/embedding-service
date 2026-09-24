"""
Phase 3.x Generic Semantic Analytics Engine - Semantic Registries & Capability Matrix.
Provides central catalog for measures, dimensions, relationships, derived metrics,
grain analysis rules, and typed field validation across ALL business tables.
Integrates with Dynamic AutoCatalogingEngine for automatic schema discovery.
"""

from typing import Dict, List, Optional, Tuple, Any, Set, Union
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import re

from app.analytics.models import (
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
from app.analytics.schema_discovery import auto_catalog


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
    data_type: str  # "string", "date", "integer", "numeric"
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
    # --- Sales Domain ---
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
    "average_order_value": MeasureDefinition(
        name="average_order_value",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        source_column="net_amount",
        default_aggregation=AggregationType.AVG,
        allowed_aggregations=[AggregationType.AVG],
        default_scope_filter="status = 'completed'",
        data_type="currency",
        description="Average order value",
    ),

    # --- Payments Domain ---
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

    # --- Product Catalog & Performance Domain ---
    "product_count": MeasureDefinition(
        name="product_count",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[AggregationType.COUNT, AggregationType.COUNT_DISTINCT],
        default_scope_filter=None,
        data_type="integer",
        description="Count of distinct products in catalog",
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
    "unit_price": MeasureDefinition(
        name="unit_price",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        source_column="unit_price",
        default_aggregation=AggregationType.AVG,
        allowed_aggregations=[AggregationType.AVG, AggregationType.MIN, AggregationType.MAX],
        default_scope_filter=None,
        data_type="currency",
        description="Product unit price",
    ),
    "cost_price": MeasureDefinition(
        name="cost_price",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        source_column="cost_price",
        default_aggregation=AggregationType.AVG,
        allowed_aggregations=[AggregationType.AVG, AggregationType.MIN, AggregationType.MAX],
        default_scope_filter=None,
        data_type="currency",
        description="Product cost price",
    ),

    # --- Salesperson / Staff Domain ---
    "salesperson_count": MeasureDefinition(
        name="salesperson_count",
        domain=DomainEnum.SALESPERSON,
        source_table="analytics_salespersons",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[AggregationType.COUNT, AggregationType.COUNT_DISTINCT],
        default_scope_filter=None,
        data_type="integer",
        description="Headcount of sales team members",
    ),
    "target_amount": MeasureDefinition(
        name="target_amount",
        domain=DomainEnum.SALESPERSON,
        source_table="analytics_salespersons",
        source_column="target_amount",
        default_aggregation=AggregationType.SUM,
        allowed_aggregations=[AggregationType.SUM, AggregationType.AVG, AggregationType.MIN, AggregationType.MAX],
        default_scope_filter=None,
        data_type="currency",
        description="Total sales quota target amount",
    ),

    # --- Customer Domain ---
    "customer_count": MeasureDefinition(
        name="customer_count",
        domain=DomainEnum.CUSTOMER,
        source_table="analytics_customers",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[AggregationType.COUNT, AggregationType.COUNT_DISTINCT],
        default_scope_filter=None,
        data_type="integer",
        description="Headcount of registered customers",
    ),

    # --- Due & Recovery Domain ---
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

    # --- CRM Domain ---
    "contact_count": MeasureDefinition(
        name="contact_count",
        domain=DomainEnum.CRM,
        source_table="crm_contacts",
        source_column="id",
        default_aggregation=AggregationType.COUNT,
        allowed_aggregations=[AggregationType.COUNT],
        default_scope_filter=None,
        data_type="integer",
        description="Total count of CRM contacts",
    ),
}


# =====================================================================
# 2. DIMENSION REGISTRY
# =====================================================================
DIMENSION_REGISTRY: Dict[str, DimensionDefinition] = {
    # Salesperson / Staff
    "salesperson": DimensionDefinition(
        name="salesperson",
        domain=DomainEnum.SALESPERSON,
        source_table="analytics_salespersons",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Sales representative or field officer name",
    ),
    "employee_code": DimensionDefinition(
        name="employee_code",
        domain=DomainEnum.SALESPERSON,
        source_table="analytics_salespersons",
        id_column="employee_code",
        label_column="employee_code",
        data_type="string",
        description="Employee ID or identification code",
    ),

    # Customer
    "customer": DimensionDefinition(
        name="customer",
        domain=DomainEnum.CUSTOMER,
        source_table="analytics_customers",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Customer or business client name",
    ),
    "address": DimensionDefinition(
        name="address",
        domain=DomainEnum.CUSTOMER,
        source_table="analytics_customers",
        id_column="address",
        label_column="address",
        data_type="string",
        description="Customer physical address or location",
    ),

    # Contact Info
    "phone": DimensionDefinition(
        name="phone",
        domain=DomainEnum.CUSTOMER,
        source_table="analytics_customers",
        id_column="phone",
        label_column="phone",
        data_type="string",
        description="Phone number or mobile contact",
    ),
    "email": DimensionDefinition(
        name="email",
        domain=DomainEnum.CUSTOMER,
        source_table="analytics_customers",
        id_column="email",
        label_column="email",
        data_type="string",
        description="Email address",
    ),

    # Product Catalog
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
    "unit_price": DimensionDefinition(
        name="unit_price",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        id_column="unit_price",
        label_column="unit_price",
        data_type="numeric",
        description="Product selling unit price",
    ),
    "cost_price": DimensionDefinition(
        name="cost_price",
        domain=DomainEnum.PRODUCT,
        source_table="analytics_products",
        id_column="cost_price",
        label_column="cost_price",
        data_type="numeric",
        description="Product wholesale cost price",
    ),

    # Payments & Transactions
    "payment_method": DimensionDefinition(
        name="payment_method",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_payments",
        id_column="payment_method",
        label_column="payment_method",
        data_type="string",
        description="Payment channel: cash, bank, bkash, nagad, etc.",
    ),
    "transaction_ref": DimensionDefinition(
        name="transaction_ref",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_payments",
        id_column="transaction_ref",
        label_column="transaction_ref",
        data_type="string",
        description="Transaction or payment reference code",
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
    "assigned_collector": DimensionDefinition(
        name="assigned_collector",
        domain=DomainEnum.PAYMENTS,
        source_table="analytics_salespersons",
        id_column="id",
        label_column="name",
        data_type="string",
        description="Assigned collector",
    ),

    # Order Details
    "order_number": DimensionDefinition(
        name="order_number",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        id_column="order_number",
        label_column="order_number",
        data_type="string",
        description="Unique order invoice code",
    ),
    "status": DimensionDefinition(
        name="status",
        domain=DomainEnum.SALES,
        source_table="analytics_orders",
        id_column="status",
        label_column="status",
        data_type="string",
        description="Entity or transaction status",
    ),

    # Due Recovery
    "assignment_status": DimensionDefinition(
        name="assignment_status",
        domain=DomainEnum.DUE_ASSIGNMENT,
        source_table="analytics_due_assignments",
        id_column="status",
        label_column="status",
        data_type="string",
        description="Recovery task status",
    ),
    "notes": DimensionDefinition(
        name="notes",
        domain=DomainEnum.DUE_ASSIGNMENT,
        source_table="analytics_due_assignments",
        id_column="notes",
        label_column="notes",
        data_type="string",
        description="Notes and descriptions",
    ),

    # Dates
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
        description="Date when payment was received",
    ),
}


# =====================================================================
# 3. RELATIONSHIP REGISTRY
# =====================================================================
RELATIONSHIP_REGISTRY: Dict[Tuple[str, str], RelationshipDefinition] = {
    ("sales", "salesperson"): RelationshipDefinition(
        name="order_salesperson",
        source_entity="sales",
        target_entity="salesperson",
        source_table="analytics_orders",
        target_table="analytics_salespersons",
        source_key="salesperson_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="account_executive",
        fanout_risk=FanoutRisk.LOW,
        join_clause="JOIN analytics_salespersons s ON s.id = o.salesperson_id AND s.workspace_id = ?",
    ),
    ("sales", "customer"): RelationshipDefinition(
        name="order_customer",
        source_entity="sales",
        target_entity="customer",
        source_table="analytics_orders",
        target_table="analytics_customers",
        source_key="customer_id",
        target_key="id",
        cardinality=CardinalityType.MANY_TO_ONE,
        semantic_role="buyer",
        fanout_risk=FanoutRisk.LOW,
        join_clause="JOIN analytics_customers c ON c.id = o.customer_id AND c.workspace_id = ?",
    ),
    ("sales", "order_item"): RelationshipDefinition(
        name="order_items_rel",
        source_entity="sales",
        target_entity="order_item",
        source_table="analytics_orders",
        target_table="analytics_order_items",
        source_key="id",
        target_key="order_id",
        cardinality=CardinalityType.ONE_TO_MANY,
        semantic_role="line_items",
        fanout_risk=FanoutRisk.HIGH,
        join_clause="JOIN analytics_order_items oi ON oi.order_id = o.id",
        requires_pre_aggregation=True,
    ),
}


# =====================================================================
# 4. DERIVED METRICS REGISTRY
# =====================================================================
DERIVED_METRIC_REGISTRY: Dict[DerivedMetricType, DerivedMetricDefinition] = {
    DerivedMetricType.AVERAGE_ORDER_VALUE: DerivedMetricDefinition(
        type=DerivedMetricType.AVERAGE_ORDER_VALUE,
        required_domains=[DomainEnum.SALES],
        base_measures=["sales_amount", "order_count"],
        formula_description="Sum of net sales divided by count of orders",
        canonical_formula="SUM(net_amount) / COUNT(id)",
        grain_strategy="single_table_aggregate",
        optimized_single_table_expr="ROUND(COALESCE(AVG(net_amount), 0), 2)",
    ),
    DerivedMetricType.OUTSTANDING_DUE: DerivedMetricDefinition(
        type=DerivedMetricType.OUTSTANDING_DUE,
        required_domains=[DomainEnum.SALES, DomainEnum.PAYMENTS],
        base_measures=["sales_amount", "collection_amount"],
        formula_description="Total sales amount minus total cash collections",
        canonical_formula="COALESCE(sales.sales_amount, 0) - COALESCE(payments.collection_amount, 0)",
        grain_strategy="pre_aggregated_cte_join",
    ),
}


# =====================================================================
# 5. CAPABILITY MATRIX
# =====================================================================
CAPABILITY_MATRIX: Dict[DomainEnum, Dict[str, Set[str]]] = {
    DomainEnum.SALES: {
        "measures": {
            "sales_amount",
            "order_count",
            "average_order_value",
            "salesperson_count",
            "customer_count",
            "product_quantity",
            "product_revenue",
        },
        "dimensions": {
            "salesperson",
            "customer",
            "product",
            "category",
            "order_date",
            "order_month",
            "order_number",
            "status",
            "phone",
            "email",
        },
        "group_by": {
            "salesperson",
            "customer",
            "product",
            "category",
            "order_date",
            "order_month",
            "order_number",
            "status",
        },
        "derived_metrics": {
            "average_order_value",
            "period_difference",
            "period_growth_percent",
        },
    },
    DomainEnum.SALESPERSON: {
        "measures": {
            "salesperson_count",
            "target_amount",
            "sales_amount",
            "order_count",
        },
        "dimensions": {
            "salesperson",
            "phone",
            "email",
            "employee_code",
            "target_amount",
            "status",
        },
        "group_by": {
            "salesperson",
            "phone",
            "email",
            "status",
            "employee_code",
            "target_amount",
        },
        "derived_metrics": set(),
    },
    DomainEnum.CUSTOMER: {
        "measures": {
            "customer_count",
            "due_amount",
            "sales_amount",
            "order_count",
        },
        "dimensions": {
            "customer",
            "phone",
            "email",
            "address",
            "status",
        },
        "group_by": {
            "customer",
            "phone",
            "email",
            "status",
            "address",
        },
        "derived_metrics": {
            "outstanding_due",
        },
    },
    DomainEnum.PRODUCT: {
        "measures": {
            "product_count",
            "product_quantity",
            "product_revenue",
            "unit_price",
            "cost_price",
        },
        "dimensions": {
            "product",
            "category",
            "unit_price",
            "cost_price",
            "status",
        },
        "group_by": {
            "product",
            "category",
            "unit_price",
            "cost_price",
            "status",
        },
        "derived_metrics": set(),
    },
    DomainEnum.PAYMENTS: {
        "measures": {
            "collection_amount",
            "payment_count",
        },
        "dimensions": {
            "collector",
            "salesperson",
            "assigned_collector",
            "customer",
            "payment_method",
            "transaction_ref",
            "collected_date",
            "status",
        },
        "group_by": {
            "collector",
            "salesperson",
            "assigned_collector",
            "customer",
            "payment_method",
            "collected_date",
            "status",
        },
        "derived_metrics": {
            "collection_rate",
            "period_difference",
            "period_growth_percent",
        },
    },
    DomainEnum.DUE: {
        "measures": {
            "due_amount",
            "sales_amount",
            "collection_amount",
        },
        "dimensions": {
            "customer",
            "salesperson",
            "phone",
            "status",
        },
        "group_by": {
            "customer",
            "salesperson",
            "status",
        },
        "derived_metrics": {
            "outstanding_due",
        },
    },
    DomainEnum.DUE_ASSIGNMENT: {
        "measures": {
            "active_assignment_count",
        },
        "dimensions": {
            "assigned_collector",
            "assignment_status",
            "status",
            "notes",
            "due_date",
        },
        "group_by": {
            "assigned_collector",
            "assignment_status",
            "status",
            "due_date",
        },
        "derived_metrics": set(),
    },
    DomainEnum.CRM: {
        "measures": {
            "contact_count",
        },
        "dimensions": {
            "name",
            "phone",
            "email",
            "source",
            "status",
        },
        "group_by": {
            "name",
            "source",
            "status",
        },
        "derived_metrics": set(),
    },
}


# =====================================================================
# 6. FIELD TYPE REGISTRY
# =====================================================================
FIELD_TYPE_REGISTRY: Dict[str, str] = {
    # String fields
    "salesperson": "string",
    "salesperson_name": "string",
    "customer": "string",
    "customer_name": "string",
    "product": "string",
    "product_name": "string",
    "category": "string",
    "payment_method": "string",
    "status": "string",
    "order_number": "string",
    "transaction_ref": "string",
    "address": "string",
    "email": "string",
    "phone": "string",
    "collector": "string",
    "assigned_collector": "string",
    "assignment_status": "string",
    "notes": "string",
    "source": "string",
    "name": "string",
    "employee_code": "string",

    # Numeric fields
    "sales_amount": "numeric",
    "order_count": "numeric",
    "collection_amount": "numeric",
    "payment_count": "numeric",
    "due_amount": "numeric",
    "outstanding_due": "numeric",
    "average_order_value": "numeric",
    "target_amount": "numeric",
    "product_count": "numeric",
    "customer_count": "numeric",
    "salesperson_count": "numeric",
    "product_quantity": "numeric",
    "product_revenue": "numeric",
    "active_assignment_count": "numeric",
    "contact_count": "numeric",
    "unit_price": "numeric",
    "cost_price": "numeric",

    # Date fields
    "order_date": "date",
    "collected_date": "date",
    "due_date": "date",
    "order_month": "string",
}


# =====================================================================
# 7. REGISTRY QUERY HELPERS & VALIDATION ENGINE
# =====================================================================
def get_measure(name: str) -> Optional[MeasureDefinition]:
    """Retrieve measure definition by registered token or dynamic catalog."""
    if name in MEASURE_REGISTRY:
        return MEASURE_REGISTRY[name]
    dynamic_measures = auto_catalog.discover_and_catalog().get("measures", {})
    if name in dynamic_measures:
        m = dynamic_measures[name]
        return MeasureDefinition(
            name=m["name"],
            domain=DomainEnum.SALES,
            source_table=m["table"],
            source_column=m["column"],
            default_aggregation=AggregationType.SUM,
            allowed_aggregations=[AggregationType.SUM, AggregationType.AVG, AggregationType.COUNT],
            data_type=m["data_type"],
            description=f"Auto-cataloged measure {name}",
        )
    return None


def get_dimension(name: str) -> Optional[DimensionDefinition]:
    """Retrieve dimension definition by registered token or dynamic catalog."""
    if name in DIMENSION_REGISTRY:
        return DIMENSION_REGISTRY[name]
    dynamic_dims = auto_catalog.discover_and_catalog().get("dimensions", {})
    if name in dynamic_dims:
        d = dynamic_dims[name]
        return DimensionDefinition(
            name=d["name"],
            domain=DomainEnum.SALES,
            source_table=d["table"],
            id_column=d["column"],
            label_column=d["column"],
            data_type=d["data_type"],
            description=f"Auto-cataloged dimension {name}",
        )
    return None


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
    """Assess fanout risk between two entities."""
    rel = get_relationship(source_entity, target_entity)
    if rel:
        return rel.fanout_risk
    return FanoutRisk.LOW


def validate_filter_value_type(field: str, operator: OperatorType, value: Any) -> Tuple[bool, Optional[str]]:
    """
    Validates that a filter's value matches the registered semantic data type of the field.
    """
    expected_type = FIELD_TYPE_REGISTRY.get(field)
    if not expected_type:
        dynamic_types = auto_catalog.discover_and_catalog().get("field_types", {})
        expected_type = dynamic_types.get(field, "string")

    if operator in (OperatorType.IS_NULL, OperatorType.NOT_NULL):
        return True, None

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

    if expected_type == "numeric":
        if operator == OperatorType.BETWEEN:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False, f"BETWEEN operator on '{field}' requires a 2-element list [start, end]"
            return True, None

        if operator == OperatorType.IN:
            if not isinstance(value, (list, tuple)):
                return False, f"IN operator on '{field}' requires a list"
            return True, None

        if not is_scalar_numeric(value):
            return False, f"Filter field '{field}' expects a numeric value, got '{value}'"
        return True, None

    if expected_type == "date":
        if operator == OperatorType.BETWEEN:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False, f"BETWEEN operator on '{field}' requires 2 date strings"
            return True, None
        return True, None

    return True, None


def validate_plan_semantics(plan: SemanticQueryPlan) -> List[str]:
    """
    Validates complete SemanticQueryPlan against the Capability Matrix and Dynamic Catalog.
    """
    errors: List[str] = []
    matrix = CAPABILITY_MATRIX.get(plan.domain)

    # Dynamic catalog fallback
    dynamic_catalog = auto_catalog.discover_and_catalog()
    dynamic_measures = set(dynamic_catalog.get("measures", {}).keys())
    dynamic_dimensions = set(dynamic_catalog.get("dimensions", {}).keys())

    allowed_measures = matrix["measures"] if matrix else dynamic_measures
    allowed_dimensions = matrix["dimensions"] if matrix else dynamic_dimensions
    allowed_group_by = matrix["group_by"] if matrix else dynamic_dimensions
    allowed_derived = matrix["derived_metrics"] if matrix else set()

    # 1. Validate Measures
    for m in plan.measures:
        if m.name not in allowed_measures and m.name not in dynamic_measures:
            errors.append(
                f"Measure '{m.name}' is not allowed in domain '{plan.domain.value}'."
            )

    # 2. Validate Dimensions
    for d in plan.dimensions:
        if d.name not in allowed_dimensions and d.name not in dynamic_dimensions:
            errors.append(
                f"Dimension '{d.name}' is not allowed in domain '{plan.domain.value}'."
            )

    # 3. Validate Group By
    for g in plan.group_by:
        if g not in allowed_group_by and g not in dynamic_dimensions:
            errors.append(
                f"Group by field '{g}' is not allowed in domain '{plan.domain.value}'."
            )

    # 4. Validate Derived Metrics
    for dm in plan.derived_metrics:
        if dm.type.value not in allowed_derived and dm.type != DerivedMetricType.OUTSTANDING_DUE:
            errors.append(
                f"Derived metric '{dm.type.value}' is not allowed in domain '{plan.domain.value}'."
            )

    # 5. Validate Filters
    for f in plan.filters:
        if f.field not in allowed_dimensions and f.field not in allowed_measures and f.field not in dynamic_dimensions and f.field not in dynamic_measures:
            errors.append(
                f"Filter field '{f.field}' is not allowed in domain '{plan.domain.value}'."
            )
        else:
            is_valid, err = validate_filter_value_type(f.field, f.operator, f.value)
            if not is_valid and err:
                errors.append(err)

    return errors
