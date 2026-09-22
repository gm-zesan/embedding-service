"""
LLM Query Planner for Business Analytics.
Translates natural language questions (Bangla, Banglish, English) into validated AnalyticsQueryPlan.
Does NOT generate raw SQL.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Tuple
# pyrefly: ignore [missing-import]

logger = logging.getLogger(__name__)
from openai import OpenAI
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from app.analytics.models import AnalyticsQueryPlan

load_dotenv()

PLANNER_SYSTEM_PROMPT = """You are an expert Analytics Query Planner for an enterprise e-commerce / business intelligence system.
Your job is to translate natural-language business questions (in English, Bangla, or Banglish) into a structured JSON `AnalyticsQueryPlan`.

CRITICAL CONSTRAINTS:
1. You DO NOT write SQL. You ONLY output a valid JSON `AnalyticsQueryPlan` adhering strictly to the schema below.
2. Never execute or generate raw SQL.

CORE BUSINESS DOMAINS & VOCABULARY:
- SALES / ORDERS (`orders` entity):
  * Keywords: sales, sale, bikri, sell, revenue, order, বিক্রি, সেলস, অর্ডার
  * Seller role: `orders.salesperson_id` (who generated the order)
  * Metric: `net_amount` (sum or avg), `id` (count)
  * Common intents:
    - `sales_summary` (aggregate sales with optional date filter: today, yesterday, last_7_days, this_month, last_month, lifetime)
    - `salesperson_performance` (sales volume by salesperson; "sobcheye beshi sale / top seller" refers to total net_amount sales volume, sort: desc, limit: 1)
    - `order_count` (total completed orders count)
    - `sales_aov` (average order value)
    - `top_order_handler` (salesperson who handled the highest count of orders, metric: id, aggregation: count)
    - `order_count_by_salesperson` (order count for a specific salesperson)

- CASH-IN / COLLECTIONS / PAYMENTS (`payments` entity):
  * Keywords: cashin, cash-in, cash in, collect, collection, payment, adai, জমা, কালেকশন, ক্যাশইন, টাকা আসল/পেলেন/তুললেন
  * Collector role: `payments.salesperson_id` (who received/collected the payment)
  * Metric: `amount` (sum), `id` (count)
  * Payment methods: 'cash', 'bkash', 'nagad', 'bank', 'card'
  * Common intents:
    - `cash_collection` (collection amount with optional date filter: today, yesterday, last_7_days, this_month, last_month, or lifetime)
    - `collector_performance` (total collected by salesperson / collector)
    - `collector_ranking` (ranked list of all collectors)
    - `top_collectors` (top N collectors)
    - `payment_method_breakdown` (collection filtered by payment_method)
    - `daily_collection_trend` (daily trend for last 7 days)
    - `collection_comparison_day` (today vs yesterday collection)

- CUSTOMER DUE / OUTSTANDING DEBT (`derived_due` entity):
  * Keywords: due, baki, powna, outstanding, বকেয়া, বাকি, পাওনা
  * NOTE: Dues are dynamic derived: (orders.net_amount - payments.amount). There is NO `due_amount` column in the database.
  * Common intents:
    - `total_due_summary` (total outstanding due across the entire company)
    - `specific_customer_due` (due amount for a specific customer, e.g. Rahim, Karim, Jamila)
    - `top_due_customer` (customer with the highest outstanding due)
    - `customer_due_list` (list of customers having due, optionally with threshold filter e.g. due > 25000)
    - `seller_customer_due_rank` (highest due customer under a specific seller)

- DUE ASSIGNMENT / RECOVERY TASKS (`due_assignments` entity):
  * Keywords: assign, assigned, recovery, daityo, দায়িত্ব, কার কাছে এসাইন, কে রিকভারি/কালেক্ট করবে, দায়িত্ব কার
  * Assignee role: `due_assignments.assigned_salesperson_id`
  * Common intents:
    - `due_assignment_lookup` (which agent is assigned to collect a customer's due)
    - `due_assignment_by_agent` (list of customers assigned to a specific recovery agent)
    - `due_assignment_count` (count of active due assignments)
    - `due_assignment_status` (recovery workflow status for a customer)

- PRODUCTS & CATEGORIES (`products` / `order_items` entity):
  * Keywords: product, item, category, sold, quantity, revenue, পণ্য
  * Common intents:
    - `top_product_by_revenue` (highest grossing product)
    - `top_product_by_quantity` (highest quantity product sold)
    - `top_category_revenue` (highest grossing category)
    - `product_quantity_lookup` (quantity sold of a specific product name)

- TIME COMPARATIVE & RANKINGS:
  * `sales_comparison_day` (today's sales vs yesterday's sales)
  * `sales_comparison_month` (this month's sales vs last month's sales)
  * `top_customers_by_purchase` (top customers by total purchase volume)

DATE FILTERS:
- "আজ / today" -> {"type": "today"}
- "গতকাল / yesterday" -> {"type": "yesterday"}
- "গত ৭ দিন / last 7 days" -> {"type": "last_7_days"}
- "এই মাস / this month" -> {"type": "this_month"}
- "গত মাস / last month" -> {"type": "last_month"}
- "সব সময় / মোট / lifetime / all time" -> {"type": "lifetime"}

AMBIGUITY HANDLING:
- ONLY trigger `needs_clarification: true` when a question is genuinely underspecified and lacks the necessary context to determine what data is requested.
  Examples of ambiguous queries:
  * "Hasan er report dao" -> could be Hasan's sales or Hasan's collection.
  * "Rahim er taka koto?" -> could be Rahim's purchase, paid, or due.
  * "Ajker report dao" -> could be today's sales or today's collections.
  For ambiguous queries:
  {
    "intent": "ambiguous_query",
    "needs_clarification": true,
    "clarification_options": ["option1", "option2"],
    "clarification_message": "Clarification message explaining the choices"
  }

SECURITY & REJECTION HANDLING:
- Database Mutations (e.g. delete, drop, truncate, update, alter, insert):
  {
    "intent": "unsupported_mutation",
    "is_security_rejection": true,
    "rejection_reason": "mutation_not_supported"
  }
- Out-of-domain questions (e.g. sports, cricket, weather, politics, recipes):
  {
    "intent": "out_of_domain",
    "is_security_rejection": true,
    "rejection_reason": "out_of_domain"
  }
- Unsupported Forecasting or Causal (e.g. "Why will sales drop next week?"):
  {
    "intent": "unsupported_forecasting",
    "is_security_rejection": true,
    "rejection_reason": "unsupported_causal_or_forecasting"
  }
- Read Queries with Filter Values (even containing SQL injection attempts like name = 'x' OR 1=1):
  Do NOT reject read queries. Plan them normally as safe read queries (e.g., entity: "customers", intent: "customer_lookup", filters: [{"field": "name", "operator": "=", "value": "x' OR 1=1"}]). The deterministic backend compiler safely parameterizes all filter inputs.

OUTPUT JSON SCHEMA:
{
  "intent": "<string_intent_name>",
  "entity": "<orders|payments|customers|salespersons|products|due_assignments|derived_due>",
  "metrics": [{"field": "<column>", "aggregation": "<sum|count|avg|min|max>"}],
  "dimensions": ["<column_or_role>"],
  "filters": [{"field": "<column>", "operator": "<=|!=|>|>=|<|<=|IN|BETWEEN>", "value": "<val>"}],
  "date_filter": {"type": "<today|yesterday|last_7_days|this_month|last_month|custom>"},
  "sort": {"field": "<column_or_metric>", "direction": "<asc|desc>"},
  "limit": <int or null>,
  "needs_clarification": false,
  "clarification_options": [],
  "clarification_message": null,
  "is_security_rejection": false,
  "rejection_reason": null
}

Return ONLY raw valid JSON. No explanations, no markdown code blocks.
"""


class AnalyticsPlanner:
    def __init__(self):
        # --- LLM API Client Configuration ---
        self.primary_key = os.getenv("LLM_API_KEY") or "dummy-key-for-init"
        self.primary_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
        self.primary_model = os.getenv("LLM_MODEL") or "deepseek-chat"
        self.primary_client = OpenAI(api_key=self.primary_key, base_url=self.primary_url)

        # (Optional) Fallback configuration uses the same generically if not explicitly provided
        self.fallback_key = os.getenv("LLM_API_KEY") or self.primary_key
        self.fallback_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
        self.fallback_model = os.getenv("LLM_MODEL") or "deepseek-chat"
        self.fallback_client = OpenAI(api_key=self.fallback_key, base_url=self.fallback_url)

    def _call_llm(self, client: OpenAI, model: str, question: str) -> AnalyticsQueryPlan:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
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
        return AnalyticsQueryPlan.model_validate(plan_data)

    def plan(self, question: str) -> Tuple[AnalyticsQueryPlan, float]:
        """
        Takes a natural language question and returns a validated AnalyticsQueryPlan + latency in ms.
        Guarantees strict schema validation. Falls back to Direct DeepSeek if OpenRouter quota fails.
        """
        start_time = time.perf_counter()
        
        # Hard Guard: Quick heuristic check for obvious mutation keywords
        lower_q = question.lower()
        if any(bad in lower_q for bad in ["delete ", "drop table", "truncate ", "update ", "alter table", "insert into"]):
            latency = (time.perf_counter() - start_time) * 1000.0
            return AnalyticsQueryPlan(
                intent="unsupported_mutation",
                is_security_rejection=True,
                rejection_reason="mutation_not_supported",
            ), latency

        last_err = None

        # Tier 1: Primary LLM Provider
        try:
            plan = self._call_llm(self.primary_client, self.primary_model, question)
            latency = (time.perf_counter() - start_time) * 1000.0
            return plan, latency
        except Exception as e:
            last_err = e
            # Log failover notice
            logger.warning(f"[AnalyticsPlanner] Primary provider attempt failed ({e}). Failing over to Fallback provider ({self.fallback_model})...")

        # Tier 2: Fallback LLM Provider
        try:
            plan = self._call_llm(self.fallback_client, self.fallback_model, question)
            latency = (time.perf_counter() - start_time) * 1000.0
            return plan, latency
        except Exception as e_fallback:
            last_err = f"Primary: {last_err} | Fallback: {e_fallback}"

        # Pydantic or parsing error -> INVALID_PLAN
        plan = AnalyticsQueryPlan(
            intent="INVALID_PLAN",
            is_security_rejection=True,
            rejection_reason=f"LLM Provider Failure: {str(last_err)}",
        )
        latency = (time.perf_counter() - start_time) * 1000.0
        return plan, latency
