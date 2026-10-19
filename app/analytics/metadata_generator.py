"""
Upload-Time Searchable Metadata Generator for Generic Spreadsheet Analytics.
Generates deterministic profiles, domain summaries, column dictionaries,
and multi-lingual (English/Bengali) searchable keywords.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.analytics.data_source import DataSource, Dataset
from app.analytics.semantic_schema import DatasetProfile, ColumnProfile

logger = logging.getLogger("metadata_generator")

# Common bilingual (English <-> Bengali/Banglish) business dictionary
BILINGUAL_KEYWORD_MAP: Dict[str, List[str]] = {
    "salary": ["বেতন", "পারিশ্রমিক", "ভাতা", "টাকা", "মাসিক বেতন", "pay", "wage", "income", "compensation"],
    "wage": ["মজুরি", "পারিশ্রমিক", "বেতন", "pay"],
    "bonus": ["বোনাস", "ইনসেনটিভ", "উৎসব ভাতা"],
    "employee": ["কর্মী", "কর্মচারী", "কর্মকর্তা", "স্টাফ", "worker", "officer", "staff", "personnel"],
    "staff": ["কর্মী", "কর্মচারী", "স্টাফ"],
    "department": ["বিভাগ", "ডিপার্টমেন্ট", "শাখা"],
    "designation": ["পদবি", "পদবী", "ডেজিগনেশন", "পদ", "rank", "title"],
    "sales": ["বিক্রি", "সেলস", "মোট বিক্রি", "revenue", "order"],
    "order": ["অর্ডার", "ফরমায়েশ", "ক্রয়াদেশ"],
    "revenue": ["রাজস্ব", "আয়", "উপার্জন", "turnover"],
    "amount": ["পরিমাণ", "টাকা", "মূল্য", "টাকার পরিমাণ"],
    "price": ["দাম", "মূল্য", "দর", "রেট", "rate", "cost"],
    "cost": ["খরচ", "ব্যয়", "লাগত", "expense"],
    "expense": ["খরচ", "ব্যয়", "খরচাপাতি", "spending"],
    "profit": ["লাভ", "মুনাফা", "নেট লাভ", "net profit"],
    "loss": ["ক্ষতি", "লোকসান"],
    "customer": ["কাস্টমার", "গ্রাহক", "ক্রেতা", "client"],
    "client": ["ক্লায়েন্ট", "গ্রাহক"],
    "product": ["পণ্য", "আইটেম", "মালামাল", "item", "goods"],
    "stock": ["স্টক", "মজুদ", "ইনভেন্টরি", "inventory"],
    "quantity": ["পরিমাণ", "সংখ্যা", "নগদ সংখ্যা", "qty", "count"],
    "date": ["তারিখ", "দিন", "সময়"],
    "month": ["মাস", "মাসিক"],
    "year": ["বছর", "বৎসর", "সাল"],
    "status": ["স্ট্যাটাস", "অবস্থা"],
    "attendance": ["উপস্থিতি", "হাজিরা", "presence"],
    "overtime": ["ওভারটাইম", "অতিরিক্ত সময়"],
    "tax": ["ট্যাক্স", "কর", "ভ্যাট"],
    "discount": ["ছাড়", "ডিসকাউন্ট", "কমিশন"],
}


class AnalyticsSearchableMetadata(BaseModel):
    id: str
    workspace_id: int
    source_id: str
    document_id: Optional[str] = None
    ingestion_purpose: str = "data_analytics"
    lifecycle_status: str = "active"
    metadata_version: int = 1
    original_filename: str
    dataset_summary: str
    sheet_names: List[str] = Field(default_factory=list)
    columns_summary: str
    searchable_keywords: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class MetadataGenerator:
    """
    Builds structured, searchable metadata for generic spreadsheet analytics datasets.
    Guarantees:
    - Never leaks or logs raw sensitive user data or secrets.
    - Robust deterministic generation with optional LLM enrichment.
    - Tenant isolation: All metadata is strictly bound to workspace_id.
    """

    def generate_metadata(
        self,
        source: DataSource,
        document_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> AnalyticsSearchableMetadata:
        workspace_id = source.workspace_id
        source_id = source.source_id
        meta_id = f"ws_{workspace_id}_{source_id}"

        sheet_names = [d.sheet_name or d.dataset_name for d in source.datasets]
        all_cols: List[str] = []
        col_summary_lines: List[str] = []
        extracted_keywords: set[str] = set()

        # Add filename keywords
        clean_name = os.path.splitext(source.original_filename)[0].replace("_", " ").replace("-", " ")
        extracted_keywords.add(clean_name.lower())
        for token in re.findall(r"\w+", clean_name.lower()):
            if len(token) > 2:
                extracted_keywords.add(token)

        for dset in source.datasets:
            sheet_title = dset.sheet_name or dset.dataset_name
            extracted_keywords.add(sheet_title.lower())
            
            sheet_col_descs = []
            for col in dset.columns:
                all_cols.append(col)
                c_lower = col.lower()
                extracted_keywords.add(c_lower)

                # Expand bilingual synonyms
                for term, synonyms in BILINGUAL_KEYWORD_MAP.items():
                    if term in c_lower:
                        extracted_keywords.add(term)
                        for syn in synonyms:
                            extracted_keywords.add(syn.lower())

                col_type = dset.data_types.get(col, "TEXT")
                sheet_col_descs.append(f"{col} ({col_type})")

            col_summary_lines.append(f"Sheet '{sheet_title}': " + ", ".join(sheet_col_descs))

        columns_summary = "; ".join(col_summary_lines)
        if len(columns_summary) > 3000:
            columns_summary = columns_summary[:2997] + "..."

        # Base deterministic dataset summary
        dataset_summary = (
            f"Spreadsheet dataset '{source.original_filename}' with {len(source.datasets)} sheet(s) "
            f"and {source.total_rows} total row(s). Sheets: {', '.join(sheet_names)}. "
            f"Contains columns: {', '.join(all_cols[:25])}."
        )

        # Optional LLM enrichment if available and configured
        if use_llm:
            try:
                llm_summary, extra_keywords = self._enrich_with_llm(
                    source=source,
                    columns_summary=columns_summary,
                    existing_keywords=list(extracted_keywords),
                )
                if llm_summary:
                    dataset_summary = llm_summary
                if extra_keywords:
                    for kw in extra_keywords:
                        extracted_keywords.add(kw.lower().strip())
            except Exception as e:
                logger.debug(f"LLM metadata enrichment skipped: {e}")

        clean_keywords = sorted([k for k in extracted_keywords if k and len(k) > 1])[:100]

        return AnalyticsSearchableMetadata(
            id=meta_id,
            workspace_id=workspace_id,
            source_id=source_id,
            document_id=str(document_id) if document_id else None,
            ingestion_purpose="data_analytics",
            lifecycle_status="active",
            metadata_version=1,
            original_filename=source.original_filename,
            dataset_summary=dataset_summary,
            sheet_names=sheet_names,
            columns_summary=columns_summary,
            searchable_keywords=clean_keywords,
            created_at=source.uploaded_at,
        )

    def _enrich_with_llm(
        self,
        source: DataSource,
        columns_summary: str,
        existing_keywords: List[str],
    ) -> tuple[Optional[str], List[str]]:
        """Safely enrich metadata with an LLM call using strict untrusted schema data boundaries."""
        from app.llm.client import LLMClient
        from app.llm.models import LLMRequest
        import asyncio

        client = LLMClient()
        schema_json = json.dumps({
            "filename": source.original_filename,
            "sheets_count": len(source.datasets),
            "columns": columns_summary,
        })
        # Prevent prompt injection from malicious column or sheet names
        clean_schema = schema_json.replace("</UNTRUSTED>", "[TAG_DEFUSED]")

        prompt = (
            "You are a Data Cataloguer. Given this spreadsheet schema metadata, provide:\n"
            "1. A concise 2-sentence dataset summary explaining what business questions it can answer.\n"
            "2. 10-15 bilingual (English and Bengali) searchable keywords/synonyms relevant to this data.\n\n"
            "Respond ONLY in valid JSON:\n"
            '{"summary": "...", "keywords": ["...", "..."]}\n\n'
            f"<UNTRUSTED>\n{clean_schema}\n</UNTRUSTED>"
        )

        req = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            try:
                # pyrefly: ignore[missing-import]
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass

        res = loop.run_until_complete(
            asyncio.wait_for(client.generate(req), timeout=4.0)
        )
        if not res or not res.content:
            return None, []

        text = res.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        parsed = json.loads(text.strip())

        return parsed.get("summary"), parsed.get("keywords", [])


metadata_generator = MetadataGenerator()
