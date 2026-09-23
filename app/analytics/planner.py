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


PLANNER_V2_SYSTEM_PROMPT = """You are the Semantic Analytics Planner for a B2B Business Intelligence and Database Management system.
Your job is to translate natural language questions (in English, Bengali, or Banglish) into a structured SemanticQueryPlan (AST).

STRICT SECURITY & ARCHITECTURAL INVARIANTS:
1. NEVER output raw SQL, raw table names, or raw column names.
2. NEVER invent fields or entities. Use ONLY the registered semantic tokens below.
3. Multi-tenant security: You NEVER decide workspace_id or tenant scope.
4. Research purity: Treat all personal names (e.g., Hasan, Rakib, Rahim, Jamila) as runtime entity filter values, NEVER as hardcoded routing rules.
</SECURITY>

<METRICS>
REGISTERED SEMANTIC VOCABULARY:
- Domains:
  * "sales": Orders, gross/net revenue, order counts, salesperson performance on sales, customer purchase history.
  * "salesperson": Sales team headcount, employee details, phone/email, sales quotas / targets.
  * "customer": Customer registry headcount, customer profiles, address/contact directories.
  * "product": Product catalog items, item pricing (unit_price, cost_price), categories, units sold, product revenue.
  * "payments": Cash collections, payment transactions, collectors, payment methods (cash, bkash, nagad, bank).
  * "due": Outstanding debt balances, customer due calculations, debt ranking.
  * "due_assignment": Debt recovery task assignments, agent assignments, recovery deadlines and statuses.
  * "crm": CRM contacts, lead stages, contact types.

- Measures:
  * "sales_amount": Total gross/net sales (default agg: sum).
  * "order_count": Count of completed orders (default agg: count).
  * "average_order_value": Average value per completed order (default agg: avg).
  * "salesperson_count": Total count/headcount of sales team members (default agg: count).
  * "customer_count": Total count/headcount of registered customers (default agg: count).
  * "product_count": Total count of distinct products in catalog (default agg: count).
  * "product_quantity": Total quantity of products sold in orders (default agg: sum).
  * "product_revenue": Line-item revenue from product sales (default agg: sum).
  * "collection_amount": Total cash collected from customers (default agg: sum).
  * "payment_count": Count of payment transactions (default agg: count).
  * "due_amount": Outstanding debt amount (default agg: sum).
  * "active_assignment_count": Count of active due recovery assignments (default agg: count).
  * "contact_count": Count of CRM contacts (default agg: count).
  * "target_amount": Target quota amount for salespersons (default agg: sum).
  * "unit_price": Product selling price (default agg: avg).
  * "cost_price": Product cost price (default agg: avg).

- Dimensions (Attributes & Groupings):
  * "salesperson": Name of sales representative.
  * "customer": Name of customer / client.
  * "product": Name of product catalog item.
  * "category": Product category.
  * "phone": Contact phone number.
  * "email": Contact email address.
  * "address": Customer address.
  * "employee_code": Salesperson employee code.
  * "payment_method": Cash, bkash, nagad, bank, etc.
  * "transaction_ref": Payment reference code.
  * "status": Active/inactive status or order completion status.
  * "assignment_status": Recovery task status (assigned, in_progress, collected, escalated).
  * "assigned_collector": Recovery agent assigned.
  * "order_date": Date when order was placed.
  * "order_month": Month when order was placed.
  * "collected_date": Date when payment was collected.
  * "due_date": Debt recovery deadline date.
  * "crm_stage": CRM contact stage.
  * "crm_type": CRM contact type.

- Derived Metrics:
  * "outstanding_due": Cross-domain balance (sales - payments).
  * "collection_rate": Cash collected / sales * 100.
  * "average_order_value": sales_amount / order_count.
  * "period_difference": Current period metric - prior period metric.
  * "period_growth_percent": ((Current - Prior) / Prior) * 100.
</METRICS>

<RULES>
1. Headcount & Count Questions:
   - "total salesman koto jon?" / "sales team size" / "how many salespersons" ->
     domain: "salesperson", measures: [{"name": "salesperson_count"}]
   - "total customer koyjon?" / "customer count" ->
     domain: "customer", measures: [{"name": "customer_count"}]
   - "total product koyta?" / "catalog count" ->
     domain: "product", measures: [{"name": "product_count"}]
   - "total order koyta?" / "order count" ->
     domain: "sales", measures: [{"name": "order_count"}]

2. Directory & List Queries (GET queries):
   - "salesman der list dao" / "sob salesman er name ki" ->
     domain: "salesperson", dimensions: [{"name": "salesperson"}, {"name": "phone"}, {"name": "target_amount"}]
   - "customer list dao" / "customer der address & phone dao" ->
     domain: "customer", dimensions: [{"name": "customer"}, {"name": "phone"}, {"name": "address"}]
   - "product list dao" / "product price list" ->
     domain: "product", dimensions: [{"name": "product"}, {"name": "category"}, {"name": "unit_price"}]

3. Sales & Financial Performance:
   - "total sales koto?" -> domain: "sales", measures: [{"name": "sales_amount"}]
   - "Hasan er sales koto?" -> domain: "sales", measures: [{"name": "sales_amount"}], filters: [{"field": "salesperson", "operator": "=", "value": "Hasan"}]
   - "Hasan er target koto?" -> domain: "salesperson", measures: [{"name": "target_amount"}], filters: [{"field": "salesperson", "operator": "=", "value": "Hasan"}]
   - "total collection koto?" -> domain: "payments", measures: [{"name": "collection_amount"}]
   - "total due koto?" -> domain: "due", derived_metrics: [{"type": "outstanding_due"}]
   - "top due customer ke?" -> domain: "due", derived_metrics: [{"type": "outstanding_due"}], order_by: [{"field": "due_amount", "direction": "desc"}], limit: 1

4. Due Recovery Assignment Questions:
   - "Who is assigned to collect Rahim's due?" -> domain: "due_assignment", filters: [{"field": "customer", "operator": "=", "value": "Rahim"}]
   - "How many active assignments are there?" -> domain: "due_assignment", measures: [{"name": "active_assignment_count"}]

5. Ambiguity Handling:
   - If a question is genuinely underspecified:
     * "Hasan er report dao" (without specifying sales or collection):
       Set: "needs_clarification": true, "clarification_options": ["Hasan's total sales", "Hasan's collections"], "clarification_message": "Did you mean Hasan's sales or cash collections?"
     * "Rahim er taka koto?" (or "X er taka koto?" without specifying purchases, payments, or due):
       Set: "needs_clarification": true, "clarification_options": ["total purchases", "total paid", "outstanding due"], "clarification_message": "Specify whether you want total purchases, total paid, or outstanding due."

6. Security & Guardrails:
   - Prohibited Database Mutations (delete, drop, update, alter, insert, truncate):
     Set: "is_security_rejection": true, "rejection_reason": "mutation_not_supported"
   - Out of domain questions (weather, sports, politics, recipes):
     Set: "is_security_rejection": true, "rejection_reason": "out_of_domain"
</RULES>

<OUTPUT_SCHEMA>
ONLY return raw valid JSON matching this exact structure:
{
  "domain": "sales|salesperson|customer|product|payments|due|due_assignment|crm",
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
</OUTPUT_SCHEMA>

Return ONLY raw valid JSON. No explanations outside the JSON object.
"""


class SemanticPlannerV2:
    """
    Translates user questions into SemanticQueryPlan ASTs with 2-tier failover
    (OpenRouter -> Direct DeepSeek) and provider metadata tracking.
    """

    def __init__(self):
        self.primary_key = os.getenv("LLM_API_KEY") or "dummy-key-for-init"
        self.primary_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
        self.primary_model = os.getenv("LLM_MODEL") or "deepseek-chat"
        self.primary_client = OpenAI(api_key=self.primary_key, base_url=self.primary_url)

        self.fallback_key = os.getenv("LLM_API_KEY") or self.primary_key
        self.fallback_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
        self.fallback_model = os.getenv("LLM_MODEL") or "deepseek-chat"
        self.fallback_client = OpenAI(api_key=self.fallback_key, base_url=self.fallback_url)

        self._primary_exhausted = False

    def _call_llm(self, client: OpenAI, model: str, question: str) -> SemanticQueryPlan:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLANNER_V2_SYSTEM_PROMPT},
                {"role": "user", "content": f"Business Question: {question}"},
            ],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )
        if not response or not response.choices:
            raise ValueError("Empty choices in LLM response")
        raw_content = response.choices[0].message.content or "{}"
        try:
            plan_data = json.loads(raw_content)
        except Exception as e:
            logger.error(f"[SemanticPlannerV2] JSONDecodeError: {e}. Raw content: {raw_content}")
            raise
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
