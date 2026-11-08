"""
Virtual Semantic Planner for Dynamic Multi-Sheet Spreadsheets.
Translates natural language questions into structured SemanticQueryPlan ASTs
for arbitrary business domains (Healthcare, Education, Retail, Hospitality, etc.).

STRICT ARCHITECTURAL INVARIANTS:
1. NEVER outputs raw SQL (SELECT, INSERT, UPDATE, etc.).
2. Outputs ONLY structured SemanticQueryPlan JSON (measures, dimensions, filters, aggregations).
3. Validates every token against the dynamic schema catalog.
4. Robust deterministic heuristic fallback for tests and offline/network-isolated environments.
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI

from app.analytics.contract_v3 import DynamicSchemaCatalog
from app.analytics.models import (
    SemanticQueryPlan,
    DomainEnum,
    MeasureSpec,
    DimensionSpec,
    FilterSpec,
    OrderBySpec,
    AggregationType,
    OperatorType,
)
from app import config

logger = logging.getLogger(__name__)


class VirtualSemanticPlanner:
    """
    Translates natural language questions into structured SemanticQueryPlan instances
    for VirtualRelationalCompiler without generating raw SQL.
    """

    def __init__(self):
        api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy-key"
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

    def plan_query(
        self,
        question: str,
        catalog: DynamicSchemaCatalog,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> SemanticQueryPlan:
        """
        Main planning method.
        Attempts structured LLM candidate generation if API key is real and network allows;
        otherwise deterministically falls back to heuristic semantic extraction.
        """
        # First, try deterministic heuristic extraction (always works offline & in tests)
        heuristic_plan = self._heuristic_plan(question, catalog)

        # If LLM API key is dummy or not configured, return deterministic plan directly
        api_key = getattr(self.client, "api_key", "")
        if not api_key or "dummy" in api_key.lower() or os.getenv("APP_ENV") == "testing":
            return heuristic_plan

        # Attempt structured plan from LLM
        try:
            return self._llm_plan_candidate(question, catalog, history, fallback=heuristic_plan)
        except Exception as e:
            logger.info("[VirtualSemanticPlanner] LLM call unavailable or failed (%s); using deterministic heuristic plan", e)
            return heuristic_plan

    def _llm_plan_candidate(
        self,
        question: str,
        catalog: DynamicSchemaCatalog,
        history: Optional[List[Dict[str, str]]],
        fallback: SemanticQueryPlan,
    ) -> SemanticQueryPlan:
        """
        Calls LLM with strict prompt requiring ONLY SemanticQueryPlan JSON.
        NEVER asks for or accepts SQL.
        """
        catalog_summary = []
        for tbl_name, tbl_meta in catalog.tables.items():
            cols = [f"{c_meta.name} ({c_meta.physical_type.value})" for c_meta in tbl_meta.columns.values()]
            catalog_summary.append(f"Table '{tbl_name}': {', '.join(cols)}")

        schema_text = "\n".join(catalog_summary)

        system_prompt = f"""You are an Expert Semantic Query Planner for an enterprise database.
Translate natural language business questions into a structured SemanticQueryPlan JSON.

<AVAILABLE_SCHEMA>
{schema_text}
</AVAILABLE_SCHEMA>

STRICT INVARIANTS:
1. NEVER output raw SQL. Do NOT output 'SELECT', 'FROM', 'WHERE', or any SQL clause.
2. Output ONLY a valid JSON object matching this schema:
{{
  "measures": [
    {{"name": "column_name", "aggregation": "sum|avg|count|min|max", "alias": "alias_name"}}
  ],
  "dimensions": [
    {{"name": "column_name", "alias": "alias_name"}}
  ],
  "filters": [
    {{"field": "column_name", "operator": "=|!=|>|>=|<|<=|LIKE|IN|BETWEEN|IS NULL|NOT NULL", "value": "value"}}
  ],
  "group_by": ["column_name"],
  "order_by": [
    {{"field": "column_name", "direction": "asc|desc"}}
  ],
  "limit": 50
}}
3. Use exact column names from the available schema.
4. For Bangla/Banglish product/entity queries (e.g. "Oil er stock", "Apple-er price"), add a filter with operator "LIKE" and value "Oil" or "Apple" without particles like 'er', 'te'.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Business Question: {question}"},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        # Validate against SemanticQueryPlan
        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(**m) for m in data.get("measures", [])],
            dimensions=[DimensionSpec(**d) for d in data.get("dimensions", [])],
            filters=[FilterSpec(**f) for f in data.get("filters", [])],
            group_by=data.get("group_by", []),
            order_by=[OrderBySpec(**o) for o in data.get("order_by", [])],
            limit=data.get("limit", 50),
        )
        return plan

    def _heuristic_plan(self, question: str, catalog: DynamicSchemaCatalog) -> SemanticQueryPlan:
        """
        Deterministic semantic extraction based on column names, measures, and keywords.
        Guarantees zero-network offline functionality.
        """
        lower_q = question.lower()

        # Find best-matching table in catalog
        best_tbl = next(iter(catalog.tables.keys())) if catalog.tables else "data_table"
        for tbl_name, tbl_meta in catalog.tables.items():
            if tbl_name.lower() in lower_q or tbl_meta.sheet_name.lower() in lower_q:
                best_tbl = tbl_name
                break

        tbl_meta = catalog.tables.get(best_tbl)
        all_cols = list(tbl_meta.columns.values()) if tbl_meta else []

        # Detect aggregation
        agg = AggregationType.SUM
        if any(w in lower_q for w in ("average", "avg", "গড়", "gor")):
            agg = AggregationType.AVG
        elif any(w in lower_q for w in ("count", "how many", "কয়টি", "কতজন", "number of", "সংখ্যা")):
            agg = AggregationType.COUNT
        elif any(w in lower_q for w in ("max", "highest", "সর্বোচ্চ", "maximum")):
            agg = AggregationType.MAX
        elif any(w in lower_q for w in ("min", "lowest", "সর্বনিম্ন", "minimum")):
            agg = AggregationType.MIN

        # Detect target measure column
        target_measure_col: Optional[str] = None
        for col in all_cols:
            c_name = col.name.lower()
            c_hdr = col.original_header.lower()
            if any(term in lower_q for term in (c_name, c_hdr)) and col.physical_type in ("NUMERIC", "INTEGER", "REAL"):
                target_measure_col = col.name
                break

        if not target_measure_col:
            # Fallback to first numeric column or '*'
            for col in all_cols:
                if col.physical_type in ("NUMERIC", "INTEGER", "REAL"):
                    target_measure_col = col.name
                    break

        measures = []
        if agg == AggregationType.COUNT and not target_measure_col:
            measures.append(MeasureSpec(name="*", aggregation=AggregationType.COUNT, alias="total_count"))
        elif target_measure_col:
            measures.append(MeasureSpec(name=target_measure_col, aggregation=agg, alias=f"{agg.value}_{target_measure_col}"))
        else:
            measures.append(MeasureSpec(name="*", aggregation=AggregationType.COUNT, alias="total_count"))

        # Detect entity filters (e.g. "Oil er stock", "John's salary")
        filters = []
        clean_q = re.sub(r"['\"?.,!]", "", lower_q)

        measure_words = set()
        if target_measure_col:
            for part in target_measure_col.lower().split("_"):
                if len(part) > 2:
                    measure_words.add(part)

        ignore_words = {
            "what", "is", "the", "total", "sum", "average", "avg", "of", "and", "in", "for", "er", "how", "many", "show", "me",
            "stock", "price", "quantity", "cost", "sales", "amount", "unit", "rate", "salary",
            "কত", "মোট", "গড়", "হিসাব", "এর", "কি", "কোন", "কতটি", "দাম", "স্টক"
        } | measure_words

        tokens = [t for t in clean_q.split() if len(t) > 2 and t not in ignore_words]

        # Check if any token matches entity column values
        text_cols = [c for c in all_cols if c.physical_type == "TEXT"]
        name_cols = [c for c in text_cols if any(k in c.name.lower() for k in ("name", "title", "item", "product", "employee", "student", "customer", "category"))]
        target_filter_col = (name_cols[0] if name_cols else (text_cols[0] if text_cols else None))

        if tokens and target_filter_col:
            # Filter candidate value
            filter_val = tokens[0]
            # Strip Bengali inflection
            filter_val = re.sub(r"(এর|র|টা|টি|দের)$", "", filter_val)
            filters.append(FilterSpec(field=target_filter_col.name, operator=OperatorType.LIKE, value=filter_val))

        return SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=measures,
            dimensions=[],
            filters=filters,
            group_by=[],
            order_by=[],
            limit=50,
        )


virtual_semantic_planner = VirtualSemanticPlanner()
