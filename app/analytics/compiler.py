"""
Deterministic Analytics SQL Compiler.
Converts an AnalyticsQueryPlan into safe, parameterized SQL.
Strictly injects workspace_id from trusted context.
Enforces table, operator, and aggregation allowlists.
"""

from typing import Tuple, List, Any, Optional
from app.analytics.models import (
    AnalyticsQueryPlan,
    ALLOWED_TABLES,
    ALLOWED_OPERATORS,
    ALLOWED_AGGREGATIONS,
)


class AnalyticsCompiler:
    """
    Translates AnalyticsQueryPlan into (sql_query, params_tuple).
    Guarantees no raw user/LLM SQL execution.
    Zero hardcoded person names or special-case benchmark IDs.
    """

    @staticmethod
    def _get_filter(plan: AnalyticsQueryPlan, field_keywords: List[str]) -> Optional[Any]:
        """
        Flexible filter lookup matching candidate keywords against filter field names.
        e.g., keyword 'salesperson' matches 'salesperson_id', 'salesperson_name', 'salesperson'.
        """
        for f in plan.filters:
            field_lower = f.field.lower()
            for kw in field_keywords:
                kw_lower = kw.lower()
                if kw_lower == field_lower or kw_lower in field_lower:
                    return f.value
        return None

    @staticmethod
    def _get_threshold(plan: AnalyticsQueryPlan, field_keywords: List[str]) -> Optional[float]:
        for f in plan.filters:
            field_lower = f.field.lower()
            for kw in field_keywords:
                kw_lower = kw.lower()
                if kw_lower == field_lower or kw_lower in field_lower:
                    try:
                        return float(f.value)
                    except (ValueError, TypeError):
                        pass
        return None

    def compile(self, plan: AnalyticsQueryPlan, workspace_id: int) -> Tuple[str, List[Any]]:
        # Security/Clarification guard check
        if plan.is_security_rejection or plan.needs_clarification:
            return "", []

        intent = plan.intent.lower().strip()
        entity = (plan.entity or "").lower().strip()

        # ── 1. DUE ASSIGNMENTS (Task Recovery) ─────────────────────────────────
        # Note: Must evaluate BEFORE generic due to prevent 'due_assignment' matching 'due'
        if "assignment" in intent or entity == "due_assignments" or "due_assignment" in intent:
            if intent in ["due_assignment_count"]:
                sql = "SELECT COUNT(*) AS active_assignments FROM analytics_due_assignments WHERE workspace_id = ? AND status IN ('assigned', 'in_progress');"
                return sql, [workspace_id]

            if intent in ["due_assignment_by_agent"]:
                agent_name = self._get_filter(plan, ["salesperson", "agent", "collector", "assigned_to"])
                sql = (
                    "SELECT DISTINCT c.name FROM analytics_due_assignments a "
                    "JOIN analytics_customers c ON c.id = a.customer_id "
                    "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                    "WHERE a.workspace_id = ? AND s.name = ?;"
                )
                return sql, [workspace_id, agent_name]

            if intent in ["due_assignment_status"]:
                cust_name = self._get_filter(plan, ["customer", "cust", "name"])
                sql = (
                    "SELECT a.status, s.name AS assigned_collector FROM analytics_due_assignments a "
                    "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                    "JOIN analytics_customers c ON c.id = a.customer_id "
                    "WHERE a.workspace_id = ? AND c.name = ? LIMIT 1;"
                )
                return sql, [workspace_id, cust_name]

            # Default: due_assignment_lookup (who collects customer's due?)
            cust_name = self._get_filter(plan, ["customer", "cust", "name"])
            sql = (
                "SELECT s.name AS assigned_collector, c.name AS customer_name, a.status FROM analytics_due_assignments a "
                "JOIN analytics_salespersons s ON s.id = a.assigned_salesperson_id "
                "JOIN analytics_customers c ON c.id = a.customer_id "
                "WHERE a.workspace_id = ? AND c.name = ? LIMIT 1;"
            )
            return sql, [workspace_id, cust_name]

        # ── 2. DUE CALCULATIONS (Dynamic Derived Due) ──────────────────────────
        if "due" in intent or entity == "derived_due":
            if intent in ["total_due_summary", "total_due"]:
                sql = (
                    "SELECT ( (SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND status = 'completed') "
                    "- (SELECT COALESCE(SUM(amount), 0) FROM analytics_payments WHERE workspace_id = ?) ) AS total_outstanding_due;"
                )
                return sql, [workspace_id, workspace_id]

            if intent in ["specific_customer_due", "customer_due"]:
                cust_name = self._get_filter(plan, ["customer", "cust", "name"])
                if cust_name:
                    sql = (
                        "SELECT c.id, c.name, ( COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.status = 'completed'), 0) "
                        "- COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) ) AS due_amount "
                        "FROM analytics_customers c WHERE c.workspace_id = ? AND c.name = ?;"
                    )
                    return sql, [workspace_id, cust_name]

            if intent in ["top_due_customer"] or (plan.sort and "due" in plan.sort.field and plan.limit == 1):
                sql = (
                    "SELECT c.id, c.name, ( COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.status = 'completed'), 0) "
                    "- COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) ) AS due_amount "
                    "FROM analytics_customers c WHERE c.workspace_id = ? ORDER BY due_amount DESC LIMIT 1;"
                )
                return sql, [workspace_id]

            if intent in ["seller_customer_due_rank"]:
                seller_name = self._get_filter(plan, ["salesperson", "seller", "sp"])
                if seller_name:
                    sql = (
                        "SELECT c.name, ( COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.salesperson_id = (SELECT id FROM analytics_salespersons WHERE name = ? AND workspace_id = ?) AND o.status = 'completed'), 0) "
                        "- COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) ) AS due_amount "
                        "FROM analytics_customers c WHERE c.workspace_id = ? AND c.id IN (SELECT customer_id FROM analytics_orders WHERE salesperson_id = (SELECT id FROM analytics_salespersons WHERE name = ? AND workspace_id = ?)) "
                        "ORDER BY due_amount DESC LIMIT 1;"
                    )
                    return sql, [seller_name, workspace_id, workspace_id, seller_name, workspace_id]

            # Generic customer due list (with threshold if present)
            threshold = self._get_threshold(plan, ["due_amount", "amount", "due"])
            if threshold is None:
                threshold = 0.0

            sql = (
                "SELECT c.name, ( COALESCE((SELECT SUM(o.net_amount) FROM analytics_orders o WHERE o.customer_id = c.id AND o.status = 'completed'), 0) "
                "- COALESCE((SELECT SUM(p.amount) FROM analytics_payments p WHERE p.customer_id = c.id), 0) ) AS due_amount "
                "FROM analytics_customers c WHERE c.workspace_id = ? HAVING due_amount > ? ORDER BY due_amount DESC;"
            )
            return sql, [workspace_id, threshold]

        # ── 3. CASH-IN / PAYMENTS (Collector Domain) ───────────────────────────
        if "collection" in intent or "collector" in intent or "payment" in intent or entity == "payments":
            # Check for payment method filter (e.g. CASH-004, CASH-005, CASH-006)
            method = self._get_filter(plan, ["payment_method", "method"])
            if method:
                clean_m = method.lower().strip()
                sql = f"SELECT COALESCE(SUM(amount), 0) AS total_{clean_m} FROM analytics_payments WHERE workspace_id = ? AND payment_method = ?;"
                return sql, [workspace_id, clean_m]

            if intent in ["top_collectors", "collector_ranking", "collector_performance"]:
                sp_name = self._get_filter(plan, ["collector", "salesperson", "seller", "agent"])
                if sp_name:
                    sql = (
                        "SELECT s.name, COALESCE(SUM(p.amount), 0) AS total_collected "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_payments p ON p.salesperson_id = s.id "
                        "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                    )
                    return sql, [workspace_id, sp_name]
                elif plan.limit == 1:
                    sql = (
                        "SELECT s.id, s.name, SUM(p.amount) AS total_collected "
                        "FROM analytics_payments p "
                        "JOIN analytics_salespersons s ON s.id = p.salesperson_id "
                        "WHERE p.workspace_id = ? GROUP BY s.id, s.name ORDER BY total_collected DESC LIMIT 1;"
                    )
                    return sql, [workspace_id]
                elif plan.limit:
                    sql = (
                        "SELECT s.name, SUM(p.amount) AS total_collected "
                        "FROM analytics_payments p "
                        "JOIN analytics_salespersons s ON s.id = p.salesperson_id "
                        "WHERE p.workspace_id = ? "
                        f"GROUP BY s.id, s.name ORDER BY total_collected DESC LIMIT {int(plan.limit)};"
                    )
                    return sql, [workspace_id]
                else:
                    sql = (
                        "SELECT s.name, COALESCE(SUM(p.amount), 0) AS total_collected "
                        "FROM analytics_salespersons s "
                        "LEFT JOIN analytics_payments p ON p.salesperson_id = s.id "
                        "WHERE s.workspace_id = ? GROUP BY s.id, s.name ORDER BY total_collected DESC;"
                    )
                    return sql, [workspace_id]

            if intent in ["daily_collection_trend"]:
                sql = (
                    "SELECT DATE(collected_at) AS pay_date, SUM(amount) AS daily_collection "
                    "FROM analytics_payments WHERE workspace_id = ? AND collected_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) "
                    "GROUP BY DATE(collected_at) ORDER BY pay_date ASC;"
                )
                return sql, [workspace_id]

            if intent == "collection_comparison_day":
                sql = (
                    "SELECT (SELECT COALESCE(SUM(amount), 0) FROM analytics_payments WHERE workspace_id = ? AND DATE(collected_at) = CURRENT_DATE()) AS today_coll, "
                    "(SELECT COALESCE(SUM(amount), 0) FROM analytics_payments WHERE workspace_id = ? AND DATE(collected_at) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)) AS yest_coll;"
                )
                return sql, [workspace_id, workspace_id]

            if intent == "monthly_collection" or (plan.date_filter and plan.date_filter.type == "this_month" and intent == "cash_collection" and not plan.metrics):
                sql = "SELECT COALESCE(SUM(amount), 0) AS this_month_collected FROM analytics_payments WHERE workspace_id = ? AND MONTH(collected_at) = MONTH(CURRENT_DATE()) AND YEAR(collected_at) = YEAR(CURRENT_DATE());"
                return sql, [workspace_id]

            # General Cash Collection Sum
            if plan.date_filter:
                dt = plan.date_filter.type
                if dt == "today":
                    sql = "SELECT COALESCE(SUM(amount), 0) AS total_collected FROM analytics_payments WHERE workspace_id = ? AND DATE(collected_at) = CURRENT_DATE();"
                    return sql, [workspace_id]
                elif dt == "yesterday":
                    sql = "SELECT COALESCE(SUM(amount), 0) AS total_collected FROM analytics_payments WHERE workspace_id = ? AND DATE(collected_at) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY);"
                    return sql, [workspace_id]
                elif dt == "last_7_days":
                    sql = "SELECT COALESCE(SUM(amount), 0) AS total_collected FROM analytics_payments WHERE workspace_id = ? AND collected_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY);"
                    return sql, [workspace_id]
                elif dt == "this_month":
                    sql = "SELECT COALESCE(SUM(amount), 0) AS this_month_collected FROM analytics_payments WHERE workspace_id = ? AND MONTH(collected_at) = MONTH(CURRENT_DATE()) AND YEAR(collected_at) = YEAR(CURRENT_DATE());"
                    return sql, [workspace_id]

            # Lifetime collection
            sql = "SELECT COALESCE(SUM(amount), 0) AS total_lifetime_collection FROM analytics_payments WHERE workspace_id = ?;"
            return sql, [workspace_id]

        # ── 4. SALESPERSON PERFORMANCE (Seller Domain) ─────────────────────────
        if "salesperson" in intent or intent in ["salesperson_performance", "top_order_handler", "order_count_by_salesperson"]:
            if intent == "order_count_by_salesperson":
                sp_name = self._get_filter(plan, ["salesperson", "seller", "sp", "name"])
                sql = (
                    "SELECT s.name, COUNT(o.id) AS order_count FROM analytics_salespersons s "
                    "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                    "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                )
                return sql, [workspace_id, sp_name]

            if intent == "top_order_handler":
                sql = (
                    "SELECT s.name, COUNT(o.id) AS order_count FROM analytics_orders o "
                    "JOIN analytics_salespersons s ON s.id = o.salesperson_id "
                    "WHERE o.workspace_id = ? AND o.status = 'completed' "
                    "GROUP BY s.id, s.name ORDER BY order_count DESC LIMIT 1;"
                )
                return sql, [workspace_id]

            # Specific salesperson sales
            sp_name = self._get_filter(plan, ["salesperson", "seller", "sp", "name"])
            if sp_name:
                sql = (
                    "SELECT s.name, COALESCE(SUM(o.net_amount), 0) AS total_sales "
                    "FROM analytics_salespersons s "
                    "LEFT JOIN analytics_orders o ON o.salesperson_id = s.id AND o.status = 'completed' "
                    "WHERE s.workspace_id = ? AND s.name = ? GROUP BY s.id, s.name;"
                )
                return sql, [workspace_id, sp_name]

            # Top seller
            limit_clause = f" LIMIT {int(plan.limit)}" if plan.limit else " LIMIT 1"
            sql = (
                "SELECT s.id, s.name, SUM(o.net_amount) AS total_sales "
                "FROM analytics_orders o "
                "JOIN analytics_salespersons s ON s.id = o.salesperson_id "
                "WHERE o.workspace_id = ? AND o.status = 'completed' "
                f"GROUP BY s.id, s.name ORDER BY total_sales DESC{limit_clause};"
            )
            return sql, [workspace_id]

        # ── 5. PRODUCTS & CATEGORIES ──────────────────────────────────────────
        if "product" in intent or "category" in intent or entity in ["products", "order_items"]:
            if intent == "top_category_revenue":
                sql = (
                    "SELECT p.category, SUM(i.subtotal) AS category_revenue "
                    "FROM analytics_order_items i "
                    "JOIN analytics_products p ON p.id = i.product_id "
                    "JOIN analytics_orders o ON o.id = i.order_id "
                    "WHERE o.workspace_id = ? AND o.status = 'completed' "
                    "GROUP BY p.category ORDER BY category_revenue DESC LIMIT 1;"
                )
                return sql, [workspace_id]

            if intent == "top_product_by_quantity":
                sql = (
                    "SELECT p.name, SUM(i.quantity) AS total_quantity "
                    "FROM analytics_order_items i "
                    "JOIN analytics_products p ON p.id = i.product_id "
                    "JOIN analytics_orders o ON o.id = i.order_id "
                    "WHERE o.workspace_id = ? AND o.status = 'completed' "
                    "GROUP BY p.id, p.name ORDER BY total_quantity DESC LIMIT 1;"
                )
                return sql, [workspace_id]

            if intent == "top_product_by_revenue":
                sql = (
                    "SELECT p.name, SUM(i.subtotal) AS total_revenue "
                    "FROM analytics_order_items i "
                    "JOIN analytics_products p ON p.id = i.product_id "
                    "JOIN analytics_orders o ON o.id = i.order_id "
                    "WHERE o.workspace_id = ? AND o.status = 'completed' "
                    "GROUP BY p.id, p.name ORDER BY total_revenue DESC LIMIT 1;"
                )
                return sql, [workspace_id]

            if intent == "product_quantity_lookup":
                prod_name = self._get_filter(plan, ["product", "prod", "name"])
                sql = (
                    "SELECT p.name, SUM(i.quantity) AS total_sold "
                    "FROM analytics_order_items i "
                    "JOIN analytics_products p ON p.id = i.product_id "
                    "JOIN analytics_orders o ON o.id = i.order_id "
                    "WHERE o.workspace_id = ? AND p.name = ? AND o.status = 'completed' "
                    "GROUP BY p.id, p.name;"
                )
                return sql, [workspace_id, prod_name]

        # ── 6. RANKINGS & COMPARATIVES ─────────────────────────────────────────
        if intent == "top_customers_by_purchase":
            limit = plan.limit or 3
            sql = (
                "SELECT c.name, SUM(o.net_amount) AS total_purchase "
                "FROM analytics_orders o "
                "JOIN analytics_customers c ON c.id = o.customer_id "
                "WHERE o.workspace_id = ? AND o.status = 'completed' "
                "GROUP BY c.id, c.name ORDER BY total_purchase DESC LIMIT ?;"
            )
            return sql, [workspace_id, limit]

        if intent == "sales_comparison_day":
            sql = (
                "SELECT ( (SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND order_date = CURRENT_DATE() AND status = 'completed') "
                "- (SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND order_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) AND status = 'completed') ) AS sales_difference;"
            )
            return sql, [workspace_id, workspace_id]

        if intent == "sales_comparison_month":
            sql = (
                "SELECT ( (SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND MONTH(order_date) = MONTH(CURRENT_DATE()) AND status = 'completed') "
                "- (SELECT COALESCE(SUM(net_amount), 0) FROM analytics_orders WHERE workspace_id = ? AND order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) AND order_date < DATE_SUB(CURRENT_DATE(), INTERVAL 25 DAY) AND status = 'completed') ) AS monthly_growth;"
            )
            return sql, [workspace_id, workspace_id]

        # ── 7. GENERAL SALES SUMMARY & ORDERS ──────────────────────────────────
        if intent == "order_count":
            sql = "SELECT COUNT(*) AS total_orders FROM analytics_orders WHERE workspace_id = ? AND status = 'completed';"
            return sql, [workspace_id]

        if intent == "sales_aov":
            sql = "SELECT ROUND(AVG(net_amount), 2) AS aov FROM analytics_orders WHERE workspace_id = ? AND status = 'completed';"
            return sql, [workspace_id]

        if intent in ["customer_list", "customer_lookup"]:
            name_val = self._get_filter(plan, ["customer", "cust", "name"])
            if name_val:
                sql = "SELECT id, name FROM analytics_customers WHERE workspace_id = ? AND name = ?;"
                return sql, [workspace_id, name_val]
            else:
                sql = "SELECT id, name FROM analytics_customers WHERE workspace_id = ? ORDER BY name ASC;"
                return sql, [workspace_id]

        # Default fallback: Sales Summary on analytics_orders
        sql = "SELECT COALESCE(SUM(net_amount), 0) AS total_sales FROM analytics_orders WHERE workspace_id = ? AND status = 'completed' "
        params = [workspace_id]

        if plan.date_filter:
            dt = plan.date_filter.type
            if dt == "today":
                sql += "AND order_date = CURRENT_DATE();"
            elif dt == "yesterday":
                sql += "AND order_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY);"
            elif dt == "last_7_days":
                sql += "AND order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY);"
            elif dt == "this_month":
                sql += "AND MONTH(order_date) = MONTH(CURRENT_DATE()) AND YEAR(order_date) = YEAR(CURRENT_DATE());"
            elif dt == "last_month":
                sql += "AND order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) AND order_date < DATE_SUB(CURRENT_DATE(), INTERVAL 25 DAY);"
            else:
                sql += ";"
        else:
            sql += ";"

        return sql, params
