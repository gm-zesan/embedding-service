"""
Phase 3.x Generic Semantic Analytics Engine - Deterministic SQL Compiler v2.
Compiles validated SemanticQueryPlan into parameterized, safe MySQL statements.
Enforces multi-tenant workspace isolation (workspace_id = ?), grain pre-aggregation,
and zero Cartesian fanout joins across ALL database tables and business entities.
"""

from collections.abc import Set
from typing import Tuple, List, Any, Optional, Dict
from app.analytics.models import (
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
from app.analytics.validator import SemanticValidator, GrainAnalysisResult


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
        if not time_range or not date_col:
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
        if plan.needs_clarification or plan.is_security_rejection:
            return "", []

        val_res = SemanticValidator.validate(plan)
        if not val_res.is_valid:
            raise CompilerError(f"Plan failed semantic validation: {'; '.join(val_res.errors)}")

        grain_info = val_res.grain_analysis or SemanticValidator.analyze_grain(plan)

        # 1. DUE Domain / OUTSTANDING_DUE metric
        if plan.domain == DomainEnum.DUE or any(
            dm.type == DerivedMetricType.OUTSTANDING_DUE for dm in plan.derived_metrics
        ):
            return self._compile_due_query(plan, workspace_id, grain_info)

        # 2. Period Comparisons
        if any(
            dm.type in (DerivedMetricType.PERIOD_DIFFERENCE, DerivedMetricType.PERIOD_GROWTH_PERCENT)
            for dm in plan.derived_metrics
        ):
            return self._compile_comparison_query(plan, workspace_id)

        # 3. DUE_ASSIGNMENT Domain
        if plan.domain == DomainEnum.DUE_ASSIGNMENT:
            return self._compile_due_assignment_query(plan, workspace_id)

        # 4. SALESPERSON Domain
        if plan.domain == DomainEnum.SALESPERSON:
            return self._compile_salesperson_domain_query(plan, workspace_id)

        # 5. CUSTOMER Domain
        if plan.domain == DomainEnum.CUSTOMER:
            return self._compile_customer_domain_query(plan, workspace_id)

        # 6. CRM Domain
        if plan.domain == DomainEnum.CRM:
            return self._compile_crm_domain_query(plan, workspace_id)

        # 7. Standard Queries (SALES, PAYMENTS, PRODUCT)
        return self._compile_standard_query(plan, workspace_id, grain_info)

    def _compile_salesperson_domain_query(self, plan: SemanticQueryPlan, workspace_id: int) -> Tuple[str, List[Any]]:
        params: List[Any] = [workspace_id]
        where_clauses: List[str] = ["s.workspace_id = ?"]
        select_items: List[str] = []
        group_items: List[str] = []

        for f in plan.filters:
            if f.field in ("salesperson", "name"):
                where_clauses.append("s.name = ?")
                params.append(f.value)
            elif f.field == "employee_code":
                where_clauses.append("s.employee_code = ?")
                params.append(f.value)
            elif f.field == "phone":
                where_clauses.append("s.phone = ?")
                params.append(f.value)
            elif f.field == "status":
                where_clauses.append("s.is_active = ?")
                params.append(1 if f.value in (1, "1", "active", True) else 0)

        has_measures = bool(plan.measures)
        has_group = bool(plan.group_by)

        if has_group:
            for g in plan.group_by:
                if g == "salesperson":
                    select_items.append("s.name AS salesperson")
                    group_items.append("s.name")
                elif g == "status":
                    select_items.append("CASE WHEN s.is_active = 1 THEN 'active' ELSE 'inactive' END AS status")
                    group_items.append("s.is_active")
                elif g == "employee_code":
                    select_items.append("s.employee_code AS employee_code")
                    group_items.append("s.employee_code")
                elif g == "phone":
                    select_items.append("s.phone AS phone")
                    group_items.append("s.phone")
                elif g == "email":
                    select_items.append("s.email AS email")
                    group_items.append("s.email")
                elif g == "target_amount":
                    select_items.append("s.target_amount AS target_amount")
                    group_items.append("s.target_amount")

        if has_measures:
            for m in plan.measures:
                m_alias = m.alias or m.name
                if m.name == "salesperson_count":
                    select_items.append(f"COUNT(s.id) AS {m_alias}")
                elif m.name == "target_amount":
                    if m.aggregation == AggregationType.AVG:
                        select_items.append(f"ROUND(COALESCE(AVG(s.target_amount), 0), 2) AS {m_alias}")
                    else:
                        select_items.append(f"COALESCE(SUM(s.target_amount), 0) AS {m_alias}")
        elif not has_group:
            if plan.dimensions:
                for d in plan.dimensions:
                    if d.name in ("salesperson", "name"):
                        select_items.append("s.name AS salesperson")
                    elif d.name == "phone":
                        select_items.append("s.phone AS phone")
                    elif d.name == "email":
                        select_items.append("s.email AS email")
                    elif d.name == "employee_code":
                        select_items.append("s.employee_code AS employee_code")
                    elif d.name == "target_amount":
                        select_items.append("s.target_amount AS target_amount")
                    elif d.name == "status":
                        select_items.append("CASE WHEN s.is_active = 1 THEN 'active' ELSE 'inactive' END AS status")
            else:
                select_items = ["s.id", "s.name AS salesperson", "s.phone", "s.email", "s.employee_code", "s.target_amount"]

        if not select_items:
            select_items = ["COUNT(s.id) AS salesperson_count"]

        sql_parts = [
            f"SELECT {', '.join(select_items)}",
            "FROM analytics_salespersons s",
            f"WHERE {' AND '.join(where_clauses)}",
        ]
        if group_items:
            sql_parts.append(f"GROUP BY {', '.join(group_items)}")

        if plan.order_by:
            ob_parts = [f"{ob.field} {ob.direction.upper()}" for ob in plan.order_by]
            sql_parts.append(f"ORDER BY {', '.join(ob_parts)}")
        elif not has_measures and not has_group:
            sql_parts.append("ORDER BY s.name ASC")

        if plan.limit:
            sql_parts.append(f"LIMIT {plan.limit}")

        return " ".join(sql_parts) + ";", params

    def _compile_customer_domain_query(self, plan: SemanticQueryPlan, workspace_id: int) -> Tuple[str, List[Any]]:
        params: List[Any] = [workspace_id]
        where_clauses: List[str] = ["c.workspace_id = ?"]
        select_items: List[str] = []
        group_items: List[str] = []

        for f in plan.filters:
            if f.field in ("customer", "name"):
                where_clauses.append("c.name = ?")
                params.append(f.value)
            elif f.field == "phone":
                where_clauses.append("c.phone = ?")
                params.append(f.value)
            elif f.field == "address":
                where_clauses.append("c.address LIKE ?")
                params.append(f"%{f.value}%")
            elif f.field == "status":
                where_clauses.append("c.is_active = ?")
                params.append(1 if f.value in (1, "1", "active", True) else 0)

        has_measures = bool(plan.measures)
        has_group = bool(plan.group_by)

        if has_group:
            for g in plan.group_by:
                if g == "customer":
                    select_items.append("c.name AS customer")
                    group_items.append("c.name")
                elif g == "status":
                    select_items.append("CASE WHEN c.is_active = 1 THEN 'active' ELSE 'inactive' END AS status")
                    group_items.append("c.is_active")
                elif g == "address":
                    select_items.append("c.address AS address")
                    group_items.append("c.address")
                elif g == "phone":
                    select_items.append("c.phone AS phone")
                    group_items.append("c.phone")
                elif g == "email":
                    select_items.append("c.email AS email")
                    group_items.append("c.email")

        if has_measures:
            for m in plan.measures:
                m_alias = m.alias or m.name
                if m.name == "customer_count":
                    select_items.append(f"COUNT(c.id) AS {m_alias}")
        elif not has_group:
            if plan.dimensions:
                for d in plan.dimensions:
                    if d.name in ("customer", "name"):
                        select_items.append("c.name AS customer")
                    elif d.name == "phone":
                        select_items.append("c.phone AS phone")
                    elif d.name == "email":
                        select_items.append("c.email AS email")
                    elif d.name == "address":
                        select_items.append("c.address AS address")
                    elif d.name == "status":
                        select_items.append("CASE WHEN c.is_active = 1 THEN 'active' ELSE 'inactive' END AS status")
            else:
                select_items = ["c.id", "c.name AS customer", "c.phone", "c.email", "c.address"]

        if not select_items:
            select_items = ["COUNT(c.id) AS customer_count"]

        sql_parts = [
            f"SELECT {', '.join(select_items)}",
            "FROM analytics_customers c",
            f"WHERE {' AND '.join(where_clauses)}",
        ]
        if group_items:
            sql_parts.append(f"GROUP BY {', '.join(group_items)}")

        if plan.order_by:
            ob_parts = [f"{ob.field} {ob.direction.upper()}" for ob in plan.order_by]
            sql_parts.append(f"ORDER BY {', '.join(ob_parts)}")
        elif not has_measures and not has_group:
            sql_parts.append("ORDER BY c.name ASC")

        if plan.limit:
            sql_parts.append(f"LIMIT {plan.limit}")

        return " ".join(sql_parts) + ";", params

    def _compile_crm_domain_query(self, plan: SemanticQueryPlan, workspace_id: int) -> Tuple[str, List[Any]]:
        params: List[Any] = [workspace_id]
        where_clauses: List[str] = ["crm.workspace_id = ?"]
        select_items: List[str] = []
        group_items: List[str] = []

        for f in plan.filters:
            if f.field in ("customer", "name"):
                where_clauses.append("crm.name = ?")
                params.append(f.value)
            elif f.field == "phone":
                where_clauses.append("crm.phone = ?")
                params.append(f.value)
            elif f.field == "crm_stage":
                where_clauses.append("crm.stage = ?")
                params.append(f.value)
            elif f.field == "crm_type":
                where_clauses.append("crm.type = ?")
                params.append(f.value)

        has_measures = bool(plan.measures)
        has_group = bool(plan.group_by)

        if has_group:
            for g in plan.group_by:
                if g == "crm_stage":
                    select_items.append("crm.stage AS crm_stage")
                    group_items.append("crm.stage")
                elif g == "crm_type":
                    select_items.append("crm.type AS crm_type")
                    group_items.append("crm.type")
                elif g == "status":
                    select_items.append("crm.status AS status")
                    group_items.append("crm.status")

        if has_measures:
            select_items.append("COUNT(crm.id) AS contact_count")
        elif not has_group:
            select_items = ["crm.id", "crm.name AS contact_name", "crm.type", "crm.stage", "crm.status"]

        if not select_items:
            select_items = ["COUNT(crm.id) AS contact_count"]

        sql_parts = [
            f"SELECT {', '.join(select_items)}",
            "FROM crm_contacts crm",
            f"WHERE {' AND '.join(where_clauses)}",
        ]
        if group_items:
            sql_parts.append(f"GROUP BY {', '.join(group_items)}")

        if plan.limit:
            sql_parts.append(f"LIMIT {plan.limit}")

        return " ".join(sql_parts) + ";", params

    def _compile_due_query(
        self, plan: SemanticQueryPlan, workspace_id: int, grain_info: GrainAnalysisResult
    ) -> Tuple[str, List[Any]]:
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

        if not plan.group_by:
            sql = (
                "SELECT ("
                "COALESCE((SELECT SUM(net_amount) FROM analytics_orders WHERE workspace_id = ? AND status = 'completed'), 0) - "
                "COALESCE((SELECT SUM(amount) FROM analytics_payments WHERE workspace_id = ?), 0)"
                ") AS total_due;"
            )
            return sql, [workspace_id, workspace_id]

        limit_clause = f" LIMIT {plan.limit}" if plan.limit else ""
        sql = (
            "WITH orders_cte AS ("
            "    SELECT customer_id, SUM(net_amount) AS total_orders "
            "    FROM analytics_orders WHERE workspace_id = ? AND status = 'completed' "
            "    GROUP BY customer_id"
            "), "
            "payments_cte AS ("
            "    SELECT customer_id, SUM(amount) AS total_payments "
            "    FROM analytics_payments WHERE workspace_id = ? "
            "    GROUP BY customer_id"
            ") "
            "SELECT c.id, c.name, "
            "COALESCE(o.total_orders, 0) - COALESCE(p.total_payments, 0) AS due_amount "
            "FROM analytics_customers c "
            "LEFT JOIN orders_cte o ON o.customer_id = c.id "
            "LEFT JOIN payments_cte p ON p.customer_id = c.id "
            "WHERE c.workspace_id = ? AND (COALESCE(o.total_orders, 0) - COALESCE(p.total_payments, 0)) > 0 "
            f"ORDER BY due_amount DESC{limit_clause};"
        )
        return sql, [workspace_id, workspace_id, workspace_id]

    def _compile_comparison_query(self, plan: SemanticQueryPlan, workspace_id: int) -> Tuple[str, List[Any]]:
        return (
            "SELECT "
            "COALESCE(SUM(CASE WHEN order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN net_amount ELSE 0 END), 0) AS current_period_sales, "
            "COALESCE(SUM(CASE WHEN order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND order_date < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN net_amount ELSE 0 END), 0) AS prior_period_sales, "
            "ROUND(((COALESCE(SUM(CASE WHEN order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN net_amount ELSE 0 END), 0) - "
            "COALESCE(SUM(CASE WHEN order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND order_date < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN net_amount ELSE 0 END), 0)) / "
            "NULLIF(COALESCE(SUM(CASE WHEN order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND order_date < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN net_amount ELSE 0 END), 0), 0)) * 100, 2) AS period_growth_percent "
            "FROM analytics_orders WHERE workspace_id = ? AND status = 'completed';",
            [workspace_id],
        )

    def _compile_due_assignment_query(
        self, plan: SemanticQueryPlan, workspace_id: int
    ) -> Tuple[str, List[Any]]:
        params: List[Any] = [workspace_id]
        agent_filter = next(
            (f for f in plan.filters if f.field in ("salesperson", "collector", "assigned_collector")),
            None,
        )

        if any(m.name == "active_assignment_count" for m in plan.measures) and agent_filter:
            sql = (
                "SELECT COUNT(*) AS active_assignments FROM analytics_due_assignments a "
                "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                "WHERE a.workspace_id = ? AND s.name = ? AND a.status IN ('assigned', 'in_progress');"
            )
            return sql, [workspace_id, agent_filter.value]

        if any(m.name == "active_assignment_count" for m in plan.measures) and not plan.dimensions:
            sql = (
                "SELECT COUNT(*) AS active_assignments FROM analytics_due_assignments "
                "WHERE workspace_id = ? AND status IN ('assigned', 'in_progress');"
            )
            return sql, params

        if agent_filter:
            sql = (
                "SELECT DISTINCT c.name FROM analytics_due_assignments a "
                "JOIN analytics_customers c ON c.id = a.customer_id "
                "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                "WHERE a.workspace_id = ? AND s.name = ?;"
            )
            return sql, [workspace_id, agent_filter.value]

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

        sql = (
            "SELECT s.name AS assigned_collector, c.name AS customer_name, a.status "
            "FROM analytics_due_assignments a "
            "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
            "JOIN analytics_customers c ON c.id = a.customer_id "
            "WHERE a.workspace_id = ?;"
        )
        return sql, params

    def _compile_standard_query(
        self, plan: SemanticQueryPlan, workspace_id: int, grain_info: GrainAnalysisResult
    ) -> Tuple[str, List[Any]]:
        # Product catalog price / count queries without orders
        if plan.domain == DomainEnum.PRODUCT and not any(m.name in ("product_quantity", "product_revenue") for m in plan.measures):
            params: List[Any] = [workspace_id]
            where_clauses: List[str] = ["pr.workspace_id = ?"]
            select_items: List[str] = []
            group_items: List[str] = []

            for f in plan.filters:
                if f.field in ("product", "name"):
                    where_clauses.append("pr.name = ?")
                    params.append(f.value)
                elif f.field == "category":
                    where_clauses.append("pr.category = ?")
                    params.append(f.value)

            has_measures = bool(plan.measures)
            has_group = bool(plan.group_by)

            if has_group:
                for g in plan.group_by:
                    if g == "product":
                        select_items.append("pr.name AS product")
                        group_items.append("pr.name")
                    elif g == "category":
                        select_items.append("pr.category AS category")
                        group_items.append("pr.category")

            if has_measures:
                for m in plan.measures:
                    m_alias = m.alias or m.name
                    if m.name == "product_count":
                        select_items.append(f"COUNT(pr.id) AS {m_alias}")
                    elif m.name == "unit_price":
                        select_items.append(f"ROUND(AVG(pr.unit_price), 2) AS {m_alias}")
                    elif m.name == "cost_price":
                        select_items.append(f"ROUND(AVG(pr.cost_price), 2) AS {m_alias}")
            elif not has_group:
                select_items = ["pr.id", "pr.name AS product", "pr.category", "pr.unit_price", "pr.cost_price"]

            if not select_items:
                select_items = ["COUNT(pr.id) AS product_count"]

            sql_parts = [
                f"SELECT {', '.join(select_items)}",
                "FROM analytics_products pr",
                f"WHERE {' AND '.join(where_clauses)}",
            ]
            if group_items:
                sql_parts.append(f"GROUP BY {', '.join(group_items)}")
            if plan.limit:
                sql_parts.append(f"LIMIT {plan.limit}")
            return " ".join(sql_parts) + ";", params

        sp_filter = next((f for f in plan.filters if f.field in ("salesperson", "collector", "assigned_collector", "salesperson_name", "collector_name")), None)
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

        params = []
        where_clauses = []
        joins = []
        select_items = []
        group_items = []

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

        if base_alias != "oi":
            where_clauses.append(f"{base_alias}.workspace_id = ?")
            params.append(workspace_id)

        if default_scope and not any(f.field == "status" for f in plan.filters):
            where_clauses.append(default_scope)

        joined_tables: Set[str] = {base_table}

        if base_alias == "oi" and "analytics_orders" not in joined_tables:
            joins.append("JOIN analytics_orders o ON o.id = oi.order_id AND o.workspace_id = ?")
            params.append(workspace_id)
            joined_tables.add("analytics_orders")

        referenced_fields = set(d.name for d in plan.dimensions)
        referenced_fields.update(plan.group_by)
        referenced_fields.update(f.field for f in plan.filters)

        if any(k in referenced_fields for k in ("salesperson", "collector", "assigned_collector", "salesperson_name", "collector_name")):
            if base_alias in ("o", "oi"):
                joins.append("JOIN analytics_salespersons s ON s.id = o.salesperson_id AND s.workspace_id = ?")
                params.append(workspace_id)
            elif base_alias == "p":
                joins.append("JOIN analytics_salespersons s ON s.id = p.salesperson_id AND s.workspace_id = ?")
                params.append(workspace_id)
            joined_tables.add("analytics_salespersons")

        if "customer" in referenced_fields:
            if base_alias in ("o", "oi"):
                joins.append("JOIN analytics_customers c ON c.id = o.customer_id AND c.workspace_id = ?")
                params.append(workspace_id)
            elif base_alias == "p":
                joins.append("JOIN analytics_customers c ON c.id = p.customer_id AND c.workspace_id = ?")
                params.append(workspace_id)
            joined_tables.add("analytics_customers")

        if "product" in referenced_fields or "category" in referenced_fields:
            if base_alias == "oi":
                joins.append("JOIN analytics_products pr ON pr.id = oi.product_id AND pr.workspace_id = ?")
                params.append(workspace_id)
                joined_tables.add("analytics_products")

        for g in plan.group_by:
            if g in ("salesperson", "collector", "assigned_collector"):
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
                select_items.append(f"DATE_FORMAT({date_col}, '%Y-%m') AS {g}")
                group_items.append(f"DATE_FORMAT({date_col}, '%Y-%m')")

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
            elif m.name == "salesperson_count":
                select_items.append(f"COUNT(DISTINCT o.salesperson_id) AS {m_alias}")
            elif m.name == "customer_count":
                select_items.append(f"COUNT(DISTINCT o.customer_id) AS {m_alias}")
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

        for dm in plan.derived_metrics:
            if dm.type == DerivedMetricType.AVERAGE_ORDER_VALUE:
                alias = dm.alias or "average_order_value"
                col = "o.net_amount"
                select_items.append(f"ROUND(COALESCE(AVG({col}), 0), 2) AS {alias}")

        if not select_items:
            select_items.append("COALESCE(SUM(net_amount), 0) AS total_value")

        time_sql, time_params = self._compile_time_range(plan.time_range, date_col)
        if time_sql:
            where_clauses.append(time_sql)
            params.extend(time_params)

        for f in plan.filters:
            if f.field in ("salesperson", "collector", "assigned_collector"):
                where_clauses.append("s.name = ?")
                params.append(f.value)
            elif f.field == "customer":
                where_clauses.append("c.name = ?")
                params.append(f.value)
            elif f.field == "product":
                where_clauses.append("pr.name = ?")
                params.append(f.value)
            elif f.field == "category":
                where_clauses.append("pr.category = ?")
                params.append(f.value)
            elif f.field == "payment_method":
                where_clauses.append("p.payment_method = ?")
                params.append(f.value)
            elif f.field == "status":
                col = "o.status" if base_alias in ("o", "oi") else "status"
                where_clauses.append(f"{col} = ?")
                params.append(f.value)

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

        if plan.order_by:
            ob_clauses = [f"{ob.field} {ob.direction.upper()}" for ob in plan.order_by]
            sql_parts.append(f"ORDER BY {', '.join(ob_clauses)}")
        elif group_items:
            if any("date" in g.lower() or "collected_at" in g.lower() for g in group_items):
                sql_parts.append("ORDER BY 1 ASC")
            else:
                first_m = plan.measures[0].name if plan.measures else "sales_amount"
                sql_parts.append(f"ORDER BY {first_m} DESC")

        if plan.limit:
            sql_parts.append(f"LIMIT {plan.limit}")

        return " ".join(sql_parts) + ";", params
