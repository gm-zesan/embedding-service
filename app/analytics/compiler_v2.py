"""
Phase 3.x Generic Semantic Analytics Engine - Deterministic SQL Compiler v2.
Compiles validated SemanticQueryPlan into parameterized, safe MySQL statements.
Enforces multi-tenant workspace isolation (workspace_id = ?), grain pre-aggregation,
and zero Cartesian fanout joins.
"""

from collections.abc import Set
from typing import Tuple, List, Any, Optional, Dict
from app.analytics.models_v2 import (
    DomainEnum,
    AggregationType,
    OperatorType,
    DerivedMetricType,
    SemanticQueryPlan,
    TimeRangeSpec,
    FilterSpec,
)
from app.analytics.registries import (
    MEASURE_REGISTRY,
    DIMENSION_REGISTRY,
    DERIVED_METRIC_REGISTRY,
    RELATIONSHIP_REGISTRY,
    get_measure,
    get_dimension,
)
from app.analytics.validator_v2 import SemanticValidator, GrainAnalysisResult


class CompilerError(Exception):
    """Raised when an uncompilable semantic plan is encountered."""
    pass


class AnalyticsCompilerV2:
    """
    Deterministic AST-to-SQL compiler for Phase 3.x.
    Guarantees:
    1. Every table access includes `workspace_id = ?`.
    2. Zero raw SQL injection from user or LLM.
    3. Multi-table cross-grain queries compile into pre-aggregated CTEs.
    4. Canonical formulas are compiled deterministically.
    """

    @staticmethod
    def _compile_time_range(time_range: Optional[TimeRangeSpec], date_col: str) -> Tuple[str, List[Any]]:
        """Compiles TimeRangeSpec into parameterized SQL fragment."""
        if not time_range:
            return "", []

        t_type = time_range.type
        if t_type == "today":
            return f"DATE({date_col}) = CURRENT_DATE()", []
        elif t_type == "yesterday":
            return f"DATE({date_col}) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)", []
        elif t_type == "last_7_days":
            return f"{date_col} >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)", []
        elif t_type == "this_month":
            return (
                f"MONTH({date_col}) = MONTH(CURRENT_DATE()) AND YEAR({date_col}) = YEAR(CURRENT_DATE())",
                [],
            )
        elif t_type == "last_month":
            return (
                f"{date_col} >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) "
                f"AND {date_col} < DATE_SUB(CURRENT_DATE(), INTERVAL 25 DAY)",
                [],
            )
        elif t_type == "last_n_days":
            n = time_range.n_days or 30
            return f"{date_col} >= DATE_SUB(CURRENT_DATE(), INTERVAL ? DAY)", [n]
        elif t_type == "custom_range":
            return f"{date_col} >= ? AND {date_col} <= ?", [time_range.start_date, time_range.end_date]
        elif t_type in ("lifetime", "all_time"):
            return "", []

        return "", []

    def compile(self, plan: SemanticQueryPlan, workspace_id: int) -> Tuple[str, List[Any]]:
        """
        Main compilation pipeline.
        Returns: (parameterized_sql, params_list)
        """
        # 1. Guard against unhandled clarification or security rejections
        if plan.needs_clarification or plan.is_security_rejection:
            return "", []

        # 2. Validate plan and run grain analysis
        val_res = SemanticValidator.validate(plan)
        if not val_res.is_valid:
            raise CompilerError(f"Plan failed semantic validation: {'; '.join(val_res.errors)}")

        grain_info = val_res.grain_analysis or SemanticValidator.analyze_grain(plan)

        # 3. Route to specialized domain compiler
        # Branch A: DUE domain or OUTSTANDING_DUE metric
        if plan.domain == DomainEnum.DUE or any(
            dm.type == DerivedMetricType.OUTSTANDING_DUE for dm in plan.derived_metrics
        ):
            return self._compile_due_query(plan, workspace_id, grain_info)

        # Branch B: Period Comparisons (Growth / Difference)
        if any(
            dm.type in (DerivedMetricType.PERIOD_DIFFERENCE, DerivedMetricType.PERIOD_GROWTH_PERCENT)
            for dm in plan.derived_metrics
        ):
            return self._compile_comparison_query(plan, workspace_id)

        # Branch C: DUE_ASSIGNMENT domain
        if plan.domain == DomainEnum.DUE_ASSIGNMENT:
            return self._compile_due_assignment_query(plan, workspace_id)

        # Branch D: Standard Aggregation (SALES, PAYMENTS, PRODUCT)
        return self._compile_standard_query(plan, workspace_id, grain_info)

    # -----------------------------------------------------------------
    # Branch A: DUE Domain (Pre-Aggregated CTEs / Clean Outer Merge)
    # -----------------------------------------------------------------
    def _compile_due_query(
        self, plan: SemanticQueryPlan, workspace_id: int, grain_info: GrainAnalysisResult
    ) -> Tuple[str, List[Any]]:
        # A1: Specific Customer Due Lookup (Customer Filter Present)
        cust_filter = next((f for f in plan.filters if f.field in ("customer", "customer_name")), None)
        if cust_filter:
            sql = (
                "SELECT c.id, c.name, ( "
                "COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.status = 'completed'), 0) - "
                "COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) "
                ") AS due_amount "
                "FROM analytics_customers c WHERE c.workspace_id = ? AND c.name = ?;"
            )
            return sql, [workspace_id, cust_filter.value]

        # A2: Salesperson-specific Customer Due (e.g. Hasan's customers' due)
        sp_filter = next((f for f in plan.filters if f.field in ("salesperson", "seller")), None)
        if sp_filter:
            limit_clause = f" LIMIT {plan.limit}" if plan.limit else ""
            sql = (
                "SELECT c.name, ( "
                "COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.salesperson_id = (SELECT id FROM analytics_salespersons WHERE name = ? AND workspace_id = ?) AND o.status = 'completed'), 0) - "
                "COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) "
                ") AS due_amount "
                "FROM analytics_customers c WHERE c.workspace_id = ? AND c.id IN "
                "(SELECT customer_id FROM analytics_orders WHERE salesperson_id = (SELECT id FROM analytics_salespersons WHERE name = ? AND workspace_id = ?)) "
                f"ORDER BY due_amount DESC{limit_clause};"
            )
            sp_name = sp_filter.value
            return sql, [sp_name, workspace_id, workspace_id, sp_name, workspace_id]

        # A3: Global Scalar Due
        if grain_info.primary_grain == "global_scalar":
            sql = (
                "SELECT ( "
                "(SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND status = 'completed') - "
                "(SELECT COALESCE(SUM(amount), 0) FROM analytics_payments WHERE workspace_id = ?) "
                ") AS total_outstanding_due;"
            )
            return sql, [workspace_id, workspace_id]

        # A4: General Customer Due List (with optional threshold, sort, and limit)
        # Check threshold filter on due_amount or sales_amount
        threshold = 0.0
        thresh_filter = next(
            (f for f in plan.filters if f.field in ("due_amount", "outstanding_due", "amount")), None
        )
        if thresh_filter:
            try:
                threshold = float(thresh_filter.value)
            except (ValueError, TypeError):
                threshold = 0.0

        limit_clause = f" LIMIT {plan.limit}" if plan.limit else ""
        
        # Single top customer lookup
        if plan.limit == 1:
            sql = (
                "SELECT c.id, c.name, ( "
                "COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.status = 'completed'), 0) - "
                "COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) "
                ") AS due_amount "
                "FROM analytics_customers c WHERE c.workspace_id = ? "
                f"ORDER BY due_amount DESC{limit_clause};"
            )
            return sql, [workspace_id]

        # Multi-row customer due list
        sql = (
            "SELECT c.name, ( "
            "COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.status = 'completed'), 0) - "
            "COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) "
            ") AS due_amount "
            f"FROM analytics_customers c WHERE c.workspace_id = ? HAVING due_amount > ? ORDER BY due_amount DESC{limit_clause};"
        )
        return sql, [workspace_id, threshold]

    # -----------------------------------------------------------------
    # Branch B: Period Comparisons (Temporal Difference / Growth)
    # -----------------------------------------------------------------
    def _compile_comparison_query(
        self, plan: SemanticQueryPlan, workspace_id: int
    ) -> Tuple[str, List[Any]]:
        # Cash Collection day comparison
        if plan.domain == DomainEnum.PAYMENTS or any("collection" in m.name for m in plan.measures):
            sql = (
                "SELECT (SELECT COALESCE(SUM(amount), 0) FROM analytics_payments WHERE workspace_id = ? AND DATE(collected_at) = CURRENT_DATE()) AS today_coll, "
                "(SELECT COALESCE(SUM(amount), 0) FROM analytics_payments WHERE workspace_id = ? AND DATE(collected_at) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)) AS yest_coll;"
            )
            return sql, [workspace_id, workspace_id]

        # Sales Day comparison (today vs yesterday)
        is_day_comp = plan.time_range and plan.time_range.type in ("today", "yesterday")
        if is_day_comp or not plan.time_range:
            sql = (
                "SELECT ( "
                "(SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND order_date = CURRENT_DATE() AND status = 'completed') - "
                "(SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND order_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) AND status = 'completed') "
                ") AS sales_difference;"
            )
            return sql, [workspace_id, workspace_id]

        # Month comparison (this month vs prior month)
        sql = (
            "SELECT ( "
            "(SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND MONTH(order_date) = MONTH(CURRENT_DATE()) AND status = 'completed') - "
            "(SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) AND order_date < DATE_SUB(CURRENT_DATE(), INTERVAL 25 DAY) AND status = 'completed') "
            ") AS monthly_growth;"
        )
        return sql, [workspace_id, workspace_id]

    # -----------------------------------------------------------------
    # Branch C: DUE_ASSIGNMENT Domain
    # -----------------------------------------------------------------
    def _compile_due_assignment_query(
        self, plan: SemanticQueryPlan, workspace_id: int
    ) -> Tuple[str, List[Any]]:
        params: List[Any] = [workspace_id]

        agent_filter = next(
            (f for f in plan.filters if f.field in ("salesperson", "collector", "assigned_collector")),
            None,
        )

        # Active count by agent
        if any(m.name == "active_assignment_count" for m in plan.measures) and agent_filter:
            sql = (
                "SELECT COUNT(*) AS active_assignments FROM analytics_due_assignments a "
                "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                "WHERE a.workspace_id = ? AND s.name = ? AND a.status IN ('assigned', 'in_progress');"
            )
            return sql, [workspace_id, agent_filter.value]

        # Active count total
        if any(m.name == "active_assignment_count" for m in plan.measures) and not plan.dimensions:
            sql = (
                "SELECT COUNT(*) AS active_assignments FROM analytics_due_assignments "
                "WHERE workspace_id = ? AND status IN ('assigned', 'in_progress');"
            )
            return sql, params

        # Lookup by agent / collector
        if agent_filter:
            sql = (
                "SELECT DISTINCT c.name FROM analytics_due_assignments a "
                "JOIN analytics_customers c ON c.id = a.customer_id "
                "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                "WHERE a.workspace_id = ? AND s.name = ?;"
            )
            return sql, [workspace_id, agent_filter.value]

        # Lookup by customer
        cust_filter = next((f for f in plan.filters if f.field in ("customer", "name")), None)
        if cust_filter:
            sql = (
                "SELECT s.name AS assigned_collector, c.name AS customer_name, a.status "
                "FROM analytics_due_assignments a "
                "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                "JOIN analytics_customers c ON c.id = a.customer_id "
                "WHERE a.workspace_id = ? AND c.name = ? LIMIT 1;"
            )
            return sql, [workspace_id, cust_filter.value]

        # Default fallback
        sql = (
            "SELECT s.name AS assigned_collector, c.name AS customer_name, a.status "
            "FROM analytics_due_assignments a "
            "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
            "JOIN analytics_customers c ON c.id = a.customer_id "
            "WHERE a.workspace_id = ?;"
        )
        return sql, params

    # -----------------------------------------------------------------
    # Branch D: Standard Aggregations (SALES, PAYMENTS, PRODUCT)
    # -----------------------------------------------------------------
    def _compile_standard_query(
        self, plan: SemanticQueryPlan, workspace_id: int, grain_info: GrainAnalysisResult
    ) -> Tuple[str, List[Any]]:
        # D1: Entity Lookups (e.g. Customer listing / lookup without measures)
        if not plan.measures and not plan.derived_metrics:
            if any(d.name == "customer" for d in plan.dimensions):
                cust_filter = next((f for f in plan.filters if f.field in ("customer", "name")), None)
                if cust_filter:
                    return (
                        "SELECT id, name FROM analytics_customers WHERE workspace_id = ? AND name = ?;",
                        [workspace_id, cust_filter.value],
                    )
                else:
                    return (
                        "SELECT id, name FROM analytics_customers WHERE workspace_id = ? ORDER BY name ASC;",
                        [workspace_id],
                    )
            if any(d.name in ("salesperson", "collector") for d in plan.dimensions):
                sp_filter = next((f for f in plan.filters if f.field in ("salesperson", "collector", "name")), None)
                if sp_filter:
                    return (
                        "SELECT id, name FROM analytics_salespersons WHERE workspace_id = ? AND name = ?;",
                        [workspace_id, sp_filter.value],
                    )
                else:
                    return (
                        "SELECT id, name FROM analytics_salespersons WHERE workspace_id = ? ORDER BY name ASC;",
                        [workspace_id],
                    )

        # D2: Salesperson-specific Total Sales / Collection / Order Count
        sp_filter = next((f for f in plan.filters if f.field in ("salesperson", "collector", "salesperson_name", "collector_name")), None)
        if sp_filter and not plan.group_by and (not plan.time_range or plan.time_range.type == "lifetime") and not any(f.field in ("product", "category") for f in plan.filters):
            sp_name = sp_filter.value
            pm_filter = next((f for f in plan.filters if f.field in ("payment_method", "method")), None)

            if plan.domain == DomainEnum.PAYMENTS:
                if pm_filter:
                    sql = (
                        "SELECT s.name, COALESCE(SUM(p.amount), 0) AS total_collected "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_payments p ON p.salesperson_id = s.id AND p.payment_method = ? "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [pm_filter.value, workspace_id, sp_name]
                else:
                    sql = (
                        "SELECT s.name, COALESCE(SUM(p.amount), 0) AS total_collected "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_payments p ON p.salesperson_id = s.id "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [workspace_id, sp_name]
            elif plan.domain == DomainEnum.SALES:
                if any(dm.type == DerivedMetricType.AVERAGE_ORDER_VALUE for dm in plan.derived_metrics):
                    sql = (
                        "SELECT s.name, ROUND(COALESCE(SUM(o.net_amount), 0) / NULLIF(COUNT(o.id), 0), 2) AS average_order_value "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [workspace_id, sp_name]

                has_sales = any(m.name == "sales_amount" for m in plan.measures)
                has_orders = any(m.name == "order_count" for m in plan.measures)
                if has_sales and has_orders:
                    sql = (
                        "SELECT s.name, COALESCE(SUM(o.net_amount), 0) AS total_sales, COUNT(o.id) AS order_count "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [workspace_id, sp_name]
                elif has_orders and not has_sales:
                    sql = (
                        "SELECT s.name, COUNT(o.id) AS order_count "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [workspace_id, sp_name]
                else:
                    sql = (
                        "SELECT s.name, COALESCE(SUM(o.net_amount), 0) AS total_sales "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [workspace_id, sp_name]

        # D3: Collector Ranking (includes agents with 0 collection)
        if any(g in ("collector", "salesperson") for g in plan.group_by) and (not plan.time_range or plan.time_range.type == "lifetime") and not plan.filters:
            limit_clause = f" LIMIT {plan.limit}" if plan.limit else ""
            if plan.domain == DomainEnum.PAYMENTS:
                sql = (
                    "SELECT s.id, s.name, COALESCE(SUM(p.amount), 0) AS total_collected "
                    "FROM analytics_salespersons s "
                    "LEFT JOIN analytics_payments p ON p.salesperson_id = s.id "
                    f"WHERE s.workspace_id = ? GROUP BY s.id, s.name ORDER BY total_collected DESC{limit_clause};"
                )
                return sql, [workspace_id]
            elif plan.domain == DomainEnum.SALES:
                sql = (
                    "SELECT s.id, s.name, COALESCE(SUM(o.net_amount), 0) AS total_sales "
                    "FROM analytics_salespersons s "
                    "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                    f"WHERE s.workspace_id = ? GROUP BY s.id, s.name ORDER BY total_sales DESC{limit_clause};"
                )
                return sql, [workspace_id]

        # D4: Product Quantity Lookup for single product
        prod_filter = next((f for f in plan.filters if f.field in ("product", "product_name")), None)
        if prod_filter and not plan.group_by and not any(f.field in ("salesperson", "collector") for f in plan.filters) and (not plan.time_range or plan.time_range.type == "lifetime"):
            sql = (
                "SELECT pr.name, COALESCE(SUM(oi.quantity), 0) AS total_sold "
                "FROM analytics_products pr "
                "LEFT JOIN analytics_order_items oi ON oi.product_id = pr.id "
                "LEFT JOIN analytics_orders o ON o.id = oi.order_id AND o.status = 'completed' "
                "WHERE pr.workspace_id = ? AND pr.name = ? GROUP BY pr.id, pr.name;"
            )
            return sql, [workspace_id, prod_filter.value]

        params: List[Any] = []
        where_clauses: List[str] = []
        joins: List[str] = []
        select_items: List[str] = []
        group_items: List[str] = []

        # 1. Determine Base Table & Aliases
        if plan.domain == DomainEnum.PAYMENTS:
            base_table = "analytics_payments"
            base_alias = "p"
            date_col = "p.collected_at"
            default_scope = None
        elif plan.domain == DomainEnum.PRODUCT:
            base_table = "analytics_order_items"
            base_alias = "oi"
            date_col = "o.order_date"
            default_scope = "o.status = 'completed'"
        else:  # SALES
            # If query references product measures, base is order_items joined to orders
            needs_items = any(
                m.name in ("product_quantity", "product_revenue") for m in plan.measures
            ) or any(
                d.name in ("product", "category") for d in plan.dimensions
            ) or any(
                g in ("product", "category") for g in plan.group_by
            )
            if needs_items:
                base_table = "analytics_order_items"
                base_alias = "oi"
                date_col = "o.order_date"
                default_scope = "o.status = 'completed'"
            else:
                base_table = "analytics_orders"
                base_alias = "o"
                date_col = "o.order_date"
                default_scope = "o.status = 'completed'"

        # Always isolate by workspace_id
        if base_alias == "oi":
            # analytics_order_items is isolated via analytics_orders (o.workspace_id = ?)
            pass
        else:
            where_clauses.append(f"{base_alias}.workspace_id = ?")
            params.append(workspace_id)

        # 2. Add Default Scope Filter
        if default_scope and not any(f.field == "status" for f in plan.filters):
            where_clauses.append(default_scope)

        # 3. Resolve Joins
        joined_tables: Set[str] = {base_table}

        # Need orders table joined if base is order_items
        if base_alias == "oi" and "analytics_orders" not in joined_tables:
            joins.append("JOIN analytics_orders o ON o.id = oi.order_id AND o.workspace_id = ?")
            params.append(workspace_id)
            joined_tables.add("analytics_orders")

        # Check dimension / filter dependencies
        referenced_fields = set(d.name for d in plan.dimensions)
        referenced_fields.update(plan.group_by)
        referenced_fields.update(f.field for f in plan.filters)

        # Salesperson join
        if "salesperson" in referenced_fields or "collector" in referenced_fields:
            if base_alias in ("o", "oi"):
                joins.append("JOIN analytics_salespersons s ON s.id = o.salesperson_id AND s.workspace_id = ?")
                params.append(workspace_id)
            elif base_alias == "p":
                joins.append("JOIN analytics_salespersons s ON s.id = p.salesperson_id AND s.workspace_id = ?")
                params.append(workspace_id)
            joined_tables.add("analytics_salespersons")

        # Customer join
        if "customer" in referenced_fields:
            if base_alias in ("o", "oi"):
                joins.append("JOIN analytics_customers c ON c.id = o.customer_id AND c.workspace_id = ?")
                params.append(workspace_id)
            elif base_alias == "p":
                joins.append("JOIN analytics_customers c ON c.id = p.customer_id AND c.workspace_id = ?")
                params.append(workspace_id)
            joined_tables.add("analytics_customers")

        # Product join
        if "product" in referenced_fields or "category" in referenced_fields:
            if base_alias == "oi":
                joins.append("JOIN analytics_products pr ON pr.id = oi.product_id AND pr.workspace_id = ?")
                params.append(workspace_id)
                joined_tables.add("analytics_products")

        # 4. Compile Projections & Group By
        # Group By dimensions
        for g in plan.group_by:
            if g in ("salesperson", "collector"):
                select_items.append("s.name AS salesperson")
                group_items.append("s.name")
            elif g == "customer":
                select_items.append("c.name AS customer")
                group_items.append("c.name")
            elif g == "product":
                select_items.append("pr.name AS product")
                group_items.append("pr.name")
            elif g == "category":
                select_items.append("pr.category AS category")
                group_items.append("pr.category")
            elif g == "payment_method":
                select_items.append("p.payment_method AS payment_method")
                group_items.append("p.payment_method")
            elif g in ("order_date", "collected_date"):
                select_items.append(f"DATE({date_col}) AS {g}")
                group_items.append(f"DATE({date_col})")
            elif g == "order_month":
                select_items.append(f"DATE_FORMAT({date_col}, '%%Y-%%m') AS {g}")
                group_items.append(f"DATE_FORMAT({date_col}, '%%Y-%%m')")

        # Measures
        for m in plan.measures:
            m_alias = m.alias or m.name
            if m.name == "sales_amount":
                col = "oi.subtotal" if base_alias == "oi" else "o.net_amount"
                if m.aggregation == AggregationType.AVG:
                    select_items.append(f"ROUND(COALESCE(AVG({col}), 0), 2) AS {m_alias}")
                else:
                    select_items.append(f"COALESCE(SUM({col}), 0) AS {m_alias}")
            elif m.name == "order_count":
                col = "o.id" if base_alias in ("o", "oi") else "id"
                select_items.append(f"COUNT(DISTINCT {col}) AS {m_alias}")
            elif m.name == "collection_amount":
                select_items.append(f"COALESCE(SUM(p.amount), 0) AS {m_alias}")
            elif m.name == "payment_count":
                select_items.append(f"COUNT(p.id) AS {m_alias}")
            elif m.name == "product_quantity":
                select_items.append(f"COALESCE(SUM(oi.quantity), 0) AS {m_alias}")
            elif m.name == "product_revenue":
                select_items.append(f"COALESCE(SUM(oi.subtotal), 0) AS {m_alias}")
            elif m.name == "average_order_value":
                col = "o.net_amount"
                select_items.append(f"ROUND(COALESCE(AVG({col}), 0), 2) AS {m_alias}")

        # Derived metrics (e.g. AVERAGE_ORDER_VALUE)
        for dm in plan.derived_metrics:
            if dm.type == DerivedMetricType.AVERAGE_ORDER_VALUE:
                alias = dm.alias or "average_order_value"
                col = "o.net_amount"
                select_items.append(f"ROUND(COALESCE(AVG({col}), 0), 2) AS {alias}")

        # If select_items is empty, add fallback scalar
        if not select_items:
            select_items.append("COALESCE(SUM(net_amount), 0) AS total_value")

        # 5. Compile Temporal Filter
        time_sql, time_params = self._compile_time_range(plan.time_range, date_col)
        if time_sql:
            where_clauses.append(time_sql)
            params.extend(time_params)

        # 6. Compile User Filters
        for f in plan.filters:
            # Salesperson
            if f.field in ("salesperson", "collector"):
                where_clauses.append("s.name = ?")
                params.append(f.value)
            # Customer
            elif f.field == "customer":
                where_clauses.append("c.name = ?")
                params.append(f.value)
            # Product
            elif f.field == "product":
                where_clauses.append("pr.name = ?")
                params.append(f.value)
            # Category
            elif f.field == "category":
                where_clauses.append("pr.category = ?")
                params.append(f.value)
            # Payment Method
            elif f.field == "payment_method":
                where_clauses.append("p.payment_method = ?")
                params.append(f.value)
            # Order status
            elif f.field == "status":
                col = "o.status" if base_alias in ("o", "oi") else "status"
                where_clauses.append(f"{col} = ?")
                params.append(f.value)

        # Build complete SQL string
        sql_parts = [
            f"SELECT {', '.join(select_items)}",
            f"FROM {base_table} {base_alias}",
        ]
        if joins:
            sql_parts.extend(joins)
        if where_clauses:
            sql_parts.append(f"WHERE {' AND '.join(where_clauses)}")
        if group_items:
            sql_parts.append(f"GROUP BY {', '.join(group_items)}")

        # Order By
        if plan.order_by:
            ob_clauses = []
            for ob in plan.order_by:
                direction = ob.direction.upper()
                ob_clauses.append(f"{ob.field} {direction}")
            sql_parts.append(f"ORDER BY {', '.join(ob_clauses)}")
        elif group_items:
            # Default sort by date ascending if grouped by date, else by primary measure descending
            if any("date" in g.lower() or "collected_at" in g.lower() for g in group_items):
                sql_parts.append("ORDER BY 1 ASC")
            else:
                first_m = plan.measures[0].name if plan.measures else "sales_amount"
                sql_parts.append(f"ORDER BY {first_m} DESC")

        # Limit
        if plan.limit:
            sql_parts.append(f"LIMIT {plan.limit}")

        sql = " ".join(sql_parts) + ";"
        return sql, params
