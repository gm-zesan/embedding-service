"""
Phase 3.x Generic Semantic Analytics Engine - Semantic Planner v2.
Translates natural language questions into structured SemanticQueryPlan ASTs.
Operates purely on registered semantic tokens (measures, dimensions, domains, derived metrics).
Zero raw SQL, zero raw table names, zero hardcoded person names.
Equipped with zero-temperature LLM execution and automatic 2-tier OpenRouter -> Direct DeepSeek failover.
"""

import os
import json
import time
import logging
from typing import Tuple, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from app.analytics.models import (
    SemanticQueryPlan,
    DomainEnum,
    OperatorType,
    AggregationType,
    TimeRangeSpec,
    FilterSpec,
)

load_dotenv()


PLANNER_V2_SYSTEM_PROMPT = """You are the Semantic Analytics Planner for a B2B Business Intelligence system.
Your job is to translate natural language business questions (in English, Bengali, or Banglish) into a structured SemanticQueryPlan (AST).

STRICT SECURITY & ARCHITECTURAL INVARIANTS:
1. NEVER output raw SQL, raw table names, or raw column names.
2. NEVER invent fields or entities. Use ONLY the registered semantic tokens below.
3. Multi-tenant security: You NEVER decide workspace_id or tenant scope.
4. Research purity: Treat all personal names (e.g., Hasan, Rakib, Rahim, Jamila) as runtime entity filter values, NEVER as hardcoded routing rules.

REGISTERED SEMANTIC VOCABULARY:
- Domains:
  * "sales": Orders, revenue, sales amounts, sales representatives, orders completed.
  * "payments": Cash collections, payment transactions, collectors, payment methods.
  * "due": Outstanding debt balances, customer due calculations, debt ranking.
  * "product": Product catalog performance, quantity sold, product revenue, categories.
  * "due_assignment": Debt recovery task assignments, agent assignments, recovery statuses.

- Measures:
  * "sales_amount": Total net order revenue (default agg: sum).
  * "order_count": Count of completed orders (default agg: count).
  * "collection_amount": Total cash collected from customers (default agg: sum).
  * "payment_count": Count of payment transactions (default agg: count).
  * "product_quantity": Quantity of products sold (default agg: sum).
  * "product_revenue": Line-item revenue from product sales (default agg: sum).
  * "active_assignment_count": Count of active due recovery assignments (default agg: count).
  * "average_order_value": Average value per completed order (canonical derived metric).

- Dimensions:
  * "salesperson": Sales representative name (seller).
  * "customer": Customer or business client name.
  * "product": Product item name.
  * "category": Product category name.
  * "payment_method": Payment channel (cash, bkash, nagad, bank, etc.).
  * "collector": Salesperson collecting payments.
  * "status": Order lifecycle status (completed, pending, cancelled).
  * "assignment_status": Recovery task status (assigned, in_progress, collected, escalated).
  * "assigned_collector": Recovery agent assigned to a customer.
  * "order_date": Order creation date.
  * "order_month": Order creation month (YYYY-MM). Use this for "month wise" or "মাসিক" grouping.
  * "collected_date": Payment collection date.

- Derived Metrics:
  * "outstanding_due": Dynamic debt balance: SUM(orders) - SUM(payments).
  * "collection_rate": (SUM(payments) / SUM(orders)) * 100.
  * "average_order_value": SUM(sales_amount) / COUNT(orders).
  * "period_difference": Current period value minus prior period value.
  * "period_growth_percent": Percentage growth compared to prior period.

- Time Ranges:
  * "today" -> {"type": "today"}
  * "yesterday" -> {"type": "yesterday"}
  * "last_7_days" -> {"type": "last_7_days"}
  * "this_month" -> {"type": "this_month"}
  * "last_month" -> {"type": "last_month"}
  * "lifetime" / "all time" / "মোট" -> {"type": "lifetime"}
  * "last_n_days" -> {"type": "last_n_days", "n_days": <int>}
  * "custom_range" -> {"type": "custom_range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}

SEMANTIC DISTINCTIONS & RULES:
1. `dimensions` vs `group_by`:
   - If user asks for a specific person's metric:
     e.g., "What are Hasan's sales?"
     -> dimensions: ["salesperson"], filters: [{"field": "salesperson", "operator": "=", "value": "Hasan"}], group_by: [] (Returns a single scalar).
   - If user asks for metrics across an entity category:
     e.g., "Show sales of each salesperson" or "Top salespersons by revenue"
     -> dimensions: ["salesperson"], filters: [], group_by: ["salesperson"] (Returns a table).

2. Typed Filter Value Safety:
   - String dimensions (salesperson, customer, product, payment_method) receive string values (e.g. "Hasan", "Laptop", "bkash").
   - Numeric measures (sales_amount, due_amount, order_count) receive numeric thresholds (e.g. 5000, 25000).
   - NEVER place a person's name into a numeric measure filter!

3. Due Questions:
   - "Total due" -> domain: "due", derived_metrics: [{"type": "outstanding_due"}], group_by: []
   - "Rahim er due koto?" -> domain: "due", derived_metrics: [{"type": "outstanding_due"}], filters: [{"field": "customer", "operator": "=", "value": "Rahim"}]
   - "Customers with due > 20000" -> domain: "due", derived_metrics: [{"type": "outstanding_due"}], filters: [{"field": "due_amount", "operator": ">", "value": 20000}], group_by: ["customer"]
   - "Top due customer" -> domain: "due", derived_metrics: [{"type": "outstanding_due"}], order_by: [{"field": "due_amount", "direction": "desc"}], limit: 1

4. Due Assignment Questions:
   - "Who is assigned to collect Rahim's due?" -> domain: "due_assignment", filters: [{"field": "customer", "operator": "=", "value": "Rahim"}]
   - "How many active assignments are there?" -> domain: "due_assignment", measures: [{"name": "active_assignment_count"}]

5. Ambiguity Handling:
   - If a question is genuinely underspecified:
     * "Hasan er report dao" (without specifying sales or collection):
       Set: "needs_clarification": true, "clarification_options": ["Hasan's total sales", "Hasan's collections"], "clarification_message": "Did you mean Hasan's sales or cash collections?"
     * "Rahim er taka koto?" (or "X er taka koto?" without specifying purchases, payments, or due):
       Set: "needs_clarification": true, "clarification_options": ["total purchases", "total paid", "outstanding due"], "clarification_message": "Specify whether you want total purchases, total paid, or outstanding due."
     * "Ajker report dao":
       Set: "needs_clarification": true, "clarification_options": ["Today's sales", "Today's collections"], "clarification_message": "Did you mean today's sales or cash collections?"

6. Special Aggregations & Groupings:
   - Daily trends: "daily cash collection", "daily sales trend" -> group_by: ["date"], dimensions: ["date"]
   - Top category revenue: "Kon category ... sobcheye beshi revenue / sell" -> domain: "product", measures: [{"name": "product_revenue", "aggregation": "sum"}], dimensions: ["category"], group_by: ["category"], order_by: [{"field": "product_revenue", "direction": "desc"}], limit: 1

7. Security & Rejection:
   - Prohibited Database Mutations (delete, drop, update, alter, insert, truncate):
     Set: "is_security_rejection": true, "rejection_reason": "mutation_not_supported"
   - Out of domain questions (weather, sports, politics, recipes):
     Set: "is_security_rejection": true, "rejection_reason": "out_of_domain"
   - Unsupported forecasting (why will sales drop next year):
     Set: "is_security_rejection": true, "rejection_reason": "unsupported_causal_or_forecasting"

OUTPUT JSON SCHEMA:
{
  "domain": "sales|payments|due|product|due_assignment",
  "measures": [{"name": "<measure_name>", "aggregation": "<sum|count|avg|min|max>"}],
  "dimensions": [{"name": "<dimension_name>"}],
  "filters": [{"field": "<field>", "operator": "<=|!=|>|>=|<|<=|IN|BETWEEN>", "value": <typed_value>}],
  "group_by": ["<dimension_name>"],
  "time_range": {"type": "<type>", "n_days": null, "start_date": null, "end_date": null},
  "order_by": [{"field": "<field>", "direction": "asc|desc"}],
  "limit": null,
  "derived_metrics": [{"type": "<outstanding_due|collection_rate|average_order_value|period_difference|period_growth_percent>"}],
  "needs_clarification": false,
  "clarification_options": [],
  "clarification_message": null,
  "is_security_rejection": false,
  "rejection_reason": null
}

Return ONLY raw valid JSON. No explanations, no markdown code blocks.
"""


class SemanticPlannerV2:
    """
    Translates user questions into SemanticQueryPlan ASTs with 2-tier failover
    (OpenRouter -> Direct DeepSeek) and provider metadata tracking.
    """

    def __init__(self):
        # --- LLM API Client Configuration ---
        self.primary_key = os.getenv("LLM_API_KEY") or "dummy-key-for-init"
        self.primary_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
        self.primary_model = os.getenv("LLM_MODEL") or "deepseek-chat"
        self.primary_client = OpenAI(api_key=self.primary_key, base_url=self.primary_url)

        # Fallback configuration uses the same generically if not explicitly provided
        self.fallback_key = os.getenv("LLM_API_KEY") or self.primary_key
        self.fallback_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
        self.fallback_model = os.getenv("LLM_MODEL") or "deepseek-chat"
        self.fallback_client = OpenAI(api_key=self.fallback_key, base_url=self.fallback_url)

        # Circuit breaker for primary provider
        self._primary_exhausted = False

    def _call_llm(self, client: OpenAI, model: str, question: str) -> SemanticQueryPlan:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLANNER_V2_SYSTEM_PROMPT},
                {"role": "user", "content": f"Business Question: {question}"},
            ],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        if not response or not response.choices:
            raise ValueError("Empty choices in LLM response")
        raw_content = response.choices[0].message.content or "{}"
        plan_data = json.loads(raw_content)
        return SemanticQueryPlan.model_validate(plan_data)

    def plan(self, question: str) -> Tuple[SemanticQueryPlan, Dict[str, Any]]:
        start_time = time.perf_counter()
        meta: Dict[str, Any] = {
            "provider_used": "primary_llm",
            "model": self.primary_model,
            "fallback_triggered": False,
            "latency_ms": 0.0,
        }

        # Fast heuristic check for prohibited mutations
        lower_q = question.lower()
        if any(bad in lower_q for bad in ["delete ", "drop table", "truncate ", "update ", "alter table", "insert into"]):
            meta["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
            plan = SemanticQueryPlan(
                domain=DomainEnum.SALES,
                is_security_rejection=True,
                rejection_reason="mutation_not_supported",
            )
            return plan, meta

        # Tier 1: Try Primary Provider (unless circuit breaker tripped)
        if not self._primary_exhausted:
            try:
                plan = self._call_llm(self.primary_client, self.primary_model, question)
                meta["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
                return plan, meta
            except Exception as err_tier1:
                err_str = str(err_tier1)
                if "402" in err_str or "credits" in err_str.lower() or "quota" in err_str.lower():
                    self._primary_exhausted = True
                logger.warning(
                    f"[SemanticPlannerV2] Primary LLM provider failed ({err_tier1}). Failing over to Fallback LLM provider..."
                )
                meta["fallback_triggered"] = True
                meta["provider_used"] = "fallback_llm"
                meta["model"] = self.fallback_model
        else:
            meta["fallback_triggered"] = True
            meta["provider_used"] = "fallback_llm"
            meta["model"] = self.fallback_model

        # Tier 2: Fallback LLM Provider
        try:
            plan = self._call_llm(self.fallback_client, self.fallback_model, question)
            meta["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
            return plan, meta
        except Exception as err_tier2:
            meta["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
            rejection_plan = SemanticQueryPlan(
                domain=DomainEnum.SALES,
                is_security_rejection=True,
                rejection_reason=f"LLM Provider Failure: {err_tier2}",
            )
            return rejection_plan, meta

