# Format-Aware Multi-File Data Source & Dataset Resolver.
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from .data_source import (
    DataSource,
    Dataset,
    DataSourceFormat,
    DocumentAnalyticsContext,
    ResolutionStatus,
    SourceResolutionResult,
    DatasetCategory,
)
from .source_registry import DataSourceRegistry, source_registry

logger = logging.getLogger(__name__)


def normalize_banglish_inflections(text: str) -> str:
    """
    Normalizes Banglish hyphenated/apostrophe inflections without corrupting legitimate compound names (e.g. X-Pro, Rice-20kg).
    Transforms:
      Rice-er -> Rice er
      Rice'r / Rice’r -> Rice r
      Rice-এর -> Rice এর
      Rice-র -> Rice র
      Rice-ta / Rice-টা -> Rice ta / Rice টা
      Rice-te / Rice-তে -> Rice te / Rice তে
    """
    if not text:
        return text
    pattern = r"([a-zA-Z0-9_ঀ-৿]+)[-’'](er|r|ta|te|e|gulo|gula|এর|র|টা|টি|তে|এ|গুলো|গুলা)"
    return re.sub(pattern, r" ", text, flags=re.IGNORECASE)



class DataSourceResolver:
    def __init__(self, registry: Optional[DataSourceRegistry] = None):
        self.registry = registry or source_registry

    def resolve(
        self,
        query: str,
        workspace_id: int,
        explicit_source_id: Optional[str] = None,
        context: Optional[DocumentAnalyticsContext] = None,
    ) -> SourceResolutionResult:
        clean_query = normalize_banglish_inflections(query.strip())
        q_lower = clean_query.lower()

        all_sources = self.registry.list_sources(workspace_id)
        if not all_sources:
            return SourceResolutionResult(
                status=ResolutionStatus.NOT_FOUND,
                reason="No uploaded documents or datasets found in workspace.",
                clarification_message="⚠️ **No Data Sources Found**: এই ওয়ার্কস্পেসে কোনো আপলোড করা ফাইল বা ডেটাসেট পাওয়া যায়নি। অনুগ্রহ করে প্রথমে একটি ফাইল (XLSX, CSV, PDF) আপলোড করুন।"
            )

        # Step 0: Cross-Source Comparison Detection
        cross_source_match = self._detect_cross_source_comparison(clean_query, all_sources)
        if cross_source_match:
            source_names = [s.original_filename for s in cross_source_match]
            joined_names = ', '.join(source_names)
            msg = (
                "ℹ️ **Multi-Source Comparison Notice**\n\n"
                f"আপনি একাধিক ফাইল (**{joined_names}**) একসাথে তুলনা করার অনুরোধ করেছেন। "
                "বর্তমান আর্কিটেকচারে প্রতি টার্নে একটি নির্দিষ্ট ফাইলের অ্যানালিটিক্স প্রসেস করা হয়। "
                f"অনুগ্রহ করে যেকোনো একটি ফাইল সিলেক্ট করে প্রশ্ন করুন (যেমন: *\"{source_names[0]} এর সেলস কত?\"* অথবা *\"{source_names[1]} এর সেলস কত?\"*)।"
            )
            return SourceResolutionResult(
                status=ResolutionStatus.UNSUPPORTED_CROSS_SOURCE,
                candidates=[{"source_id": s.source_id, "filename": s.original_filename} for s in cross_source_match],
                reason=f"Cross-source comparison across separate files ({joined_names}) is currently unsupported.",
                clarification_message=msg,
                clarification_options=[f"{s.original_filename}-এর তথ্য দেখাও" for s in cross_source_match]
            )

        # Step 0b: Multi-Category / Cross-Domain Comparison Detection
        cross_dom_res = self._detect_cross_domain_comparison(clean_query, all_sources)
        if cross_dom_res:
            return cross_dom_res
        cross_cat_res = self._detect_multi_category_comparison(clean_query, all_sources)
        if cross_cat_res:
            return cross_cat_res

        # Step 1: Explicit Source ID in Request
        if explicit_source_id:
            matched_src = self.registry.get_source(workspace_id, explicit_source_id)
            if matched_src:
                selected_datasets = self._resolve_datasets_in_source(clean_query, matched_src)
                if len(selected_datasets) == 1:
                    is_compat, missing_reason = self._check_schema_compatibility(clean_query, selected_datasets[0])
                    if not is_compat:
                        return SourceResolutionResult(
                            status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                            source_id=matched_src.source_id,
                            source=matched_src,
                            dataset_ids=[selected_datasets[0].dataset_id],
                            selected_datasets=selected_datasets,
                            reason=missing_reason,
                            clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                            clarification_options=[f"{matched_src.original_filename}-এর অন্যান্য তথ্য দেখাও"],
                        )
                return SourceResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    source_id=matched_src.source_id,
                    source=matched_src,
                    dataset_ids=[d.dataset_id for d in selected_datasets],
                    selected_datasets=selected_datasets,
                    reason="Resolved via explicit request source_id.",
                )
            # P2-2: When explicit file_id/source_id is supplied and cannot be resolved in the current workspace, STOP.
            # Do NOT fall back to filename, context, dataset, semantic matching, or latest source.
            return SourceResolutionResult(
                status=ResolutionStatus.NOT_FOUND,
                reason=f"Explicitly specified source_id '{explicit_source_id}' was not found in workspace {workspace_id}.",
                clarification_message="⚠️ **Source Not Found**: নির্দিষ্ট করা ফাইল বা সোর্স আইডিটি পাওয়া যায়নি বা এই ওয়ার্কস্পেসের আওতাভুক্ত নয়।",
            )

        # Step 2: Explicit Filename / Source Mention in Query
        explicit_source = self._match_source_by_query_mention(clean_query, all_sources)
        if explicit_source:
            selected_datasets = self._resolve_datasets_in_source(clean_query, explicit_source)
            if len(selected_datasets) == 1:
                is_compat, missing_reason = self._check_schema_compatibility(clean_query, selected_datasets[0])
                if not is_compat:
                    return SourceResolutionResult(
                        status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                        source_id=explicit_source.source_id,
                        source=explicit_source,
                        dataset_ids=[selected_datasets[0].dataset_id],
                        selected_datasets=selected_datasets,
                        reason=missing_reason,
                        clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                        clarification_options=[f"{explicit_source.original_filename}-এর অন্যান্য তথ্য দেখাও"],
                    )
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=explicit_source.source_id,
                source=explicit_source,
                dataset_ids=[d.dataset_id for d in selected_datasets],
                selected_datasets=selected_datasets,
                reason=f"Resolved via explicit filename reference: {explicit_source.original_filename}",
            )

        # Step 3: Explicit Dataset / Sheet Mention in Query
        sheet_matched_source, matched_dataset = self._match_by_dataset_name(clean_query, all_sources)
        if sheet_matched_source and matched_dataset:
            is_compat, missing_reason = self._check_schema_compatibility(clean_query, matched_dataset)
            if not is_compat:
                return SourceResolutionResult(
                    status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                    source_id=sheet_matched_source.source_id,
                    source=sheet_matched_source,
                    dataset_ids=[matched_dataset.dataset_id],
                    selected_datasets=[matched_dataset],
                    reason=missing_reason,
                    clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                    clarification_options=[f"{sheet_matched_source.original_filename}-এর অন্যান্য তথ্য দেখাও"],
                )
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=sheet_matched_source.source_id,
                source=sheet_matched_source,
                dataset_ids=[matched_dataset.dataset_id],
                selected_datasets=[matched_dataset],
                reason=f"Resolved via explicit sheet/dataset mention: {matched_dataset.dataset_name}",
            )

        # Step 4: Active Conversational Source Context
        if context and (context.active_source_id or context.source_id):
            active_id = context.active_source_id or context.source_id
            active_src = self.registry.get_source(workspace_id, active_id)
            if active_src:
                is_followup = self._is_contextual_followup(clean_query, context)
                if is_followup:
                    selected_datasets = self._resolve_datasets_in_source(clean_query, active_src, default_dataset_id=context.active_dataset_id)
                    return SourceResolutionResult(
                        status=ResolutionStatus.RESOLVED,
                        source_id=active_src.source_id,
                        source=active_src,
                        dataset_ids=[d.dataset_id for d in selected_datasets],
                        selected_datasets=selected_datasets,
                        reason=f"Resolved via active conversational context: {active_src.original_filename}",
                    )

        # Step 5: Document Indicator Query & Semantic Schema / Business Category Intent Matching
        has_doc_ind = self._has_document_indicators(clean_query)
        has_generic_ref = self._has_generic_file_reference(clean_query)

        if not (has_generic_ref and len(all_sources) > 1):
            # 5a. Domain-agnostic semantic schema matching
            semantic_res = self._resolve_by_semantic_schema(clean_query, all_sources, has_doc_ind)
            if semantic_res is not None:
                return semantic_res

            # 5b. Business category heuristic fallback
            category_res = self._resolve_by_business_category_and_intent(clean_query, all_sources, has_doc_ind)
            if category_res is not None:
                return category_res

        # Step 6: Semantic Keyword & Column Match Across All Sources
        matching_sources = self._score_sources_by_schema_match(clean_query, all_sources)

        if len(matching_sources) > 1:
            candidate_list = [
                {
                    "source_id": src.source_id,
                    "filename": src.original_filename,
                    "format": src.format.value,
                    "matched_datasets": [d.dataset_name for d in dsets],
                }
                for src, dsets in matching_sources
            ]
            options = [f"{src.original_filename} থেকে উত্তর দাও" for src, _ in matching_sources]
            items = []
            for idx, (src, dsets) in enumerate(matching_sources):
                dnames = ", ".join(d.dataset_name for d in dsets)
                items.append(f"{idx+1}. `{src.original_filename}` ({dnames})")
            items_str = "\n".join(items)
            clarification_msg = (
                f"🤔 **Ambiguous Source Context**: একাধিক ফাইলে আপনার প্রশ্নের সাথে সম্পর্কিত ডেটা রয়েছে:\n"
                f"{items_str}\n\n"
                f"আপনি কোন ফাইলের ডেটা দেখতে চাচ্ছেন অনুগ্রহ করে সিলেক্ট করুন।"
            )
            return SourceResolutionResult(
                status=ResolutionStatus.AMBIGUOUS,
                candidates=candidate_list,
                reason="Multiple uploaded files match the query columns/entities. User clarification required.",
                clarification_message=clarification_msg,
                clarification_options=options,
            )

        if len(matching_sources) == 1 and has_doc_ind:
            best_src, datasets = matching_sources[0]
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=best_src.source_id,
                source=best_src,
                dataset_ids=[d.dataset_id for d in datasets],
                selected_datasets=datasets,
                reason=f"Resolved via document indicator and semantic schema match: {best_src.original_filename}",
            )

        if has_doc_ind:
            if len(all_sources) == 1:
                single_src = all_sources[0]
                selected_datasets = self._resolve_datasets_in_source(clean_query, single_src)
                if len(selected_datasets) == 1:
                    is_compat, missing_reason = self._check_schema_compatibility(clean_query, selected_datasets[0])
                    if not is_compat:
                        return SourceResolutionResult(
                            status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                            source_id=single_src.source_id,
                            source=single_src,
                            dataset_ids=[selected_datasets[0].dataset_id],
                            selected_datasets=selected_datasets,
                            reason=missing_reason,
                            clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                            clarification_options=[f"{single_src.original_filename}-এর অন্যান্য তথ্য দেখাও"],
                        )
                return SourceResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    source_id=single_src.source_id,
                    source=single_src,
                    dataset_ids=[d.dataset_id for d in selected_datasets],
                    selected_datasets=selected_datasets,
                    reason=f"Resolved to single available workspace source via document indicators: {single_src.original_filename}",
                )
            elif len(all_sources) > 1:
                candidate_list = [
                    {
                        "source_id": src.source_id,
                        "filename": src.original_filename,
                        "format": src.format.value,
                        "matched_datasets": [d.dataset_name for d in src.datasets],
                    }
                    for src in all_sources
                ]
                options = [f"{src.original_filename} থেকে দেখাও" for src in all_sources[:4]]
                items = [f"{idx+1}. `{src.original_filename}`" for idx, src in enumerate(all_sources[:4])]
                items_str = "\n".join(items)
                clarification_msg = (
                    f"🤔 **Ambiguous Source Context**: একাধিক ফাইল আপলোড করা রয়েছে:\n"
                    f"{items_str}\n\n"
                    f"আপনি কোন ফাইলের তথ্য জানতে চাচ্ছেন অনুগ্রহ করে ফাইলের নাম উল্লেখ করুন।"
                )
                return SourceResolutionResult(
                    status=ResolutionStatus.AMBIGUOUS,
                    candidates=candidate_list,
                    reason="Query references an uploaded document but multiple files exist. User clarification required.",
                    clarification_message=clarification_msg,
                    clarification_options=options,
                )

        return SourceResolutionResult(
            status=ResolutionStatus.NOT_FOUND,
            reason="Could not determine a matching data source or dataset for the query.",
            clarification_message=(
                "ℹ️ **Query Interpretation Notice**: আপনার প্রশ্নের সাথে মেলানো যায় এমন কোনো ফাইল বা শিট পাওয়া যায়নি। "
                "অনুগ্রহ করে ফাইলের নাম বা শিটের নাম উল্লেখ করে প্রশ্ন করুন (যেমন: *\"products.xlsx এ Rice এর stock কত?\"* বা *\"orders.csv এ মোট সেলস কত?\"*)"
            ),
            clarification_options=[f"{s.original_filename} ব্যবহার করো" for s in all_sources[:4]],
        )

    def _has_document_indicators(self, query: str) -> bool:
        q_lower = query.lower()
        doc_indicators = [
            "file", "files", "excel", "xlsx", "xls", "csv", "sheet", "sheets",
            "spreadsheet", "document", "ফাইল", "শিট", "শীট", "এক্সেল", "স্প্রেডশিট",
            "এই file", "এই excel", "এই ফাইল", "এই এক্সেল", "oi file", "oi excel", "ঐ ফাইল", "ওই ফাইল"
        ]
        return any(ind in q_lower for ind in doc_indicators)

    def _has_generic_file_reference(self, query: str) -> bool:
        q_lower = query.lower()
        generic_refs = [
            "এই file", "এই excel", "এই ফাইল", "এই এক্সেল", "এই স্প্রেডশিট", "এই csv",
            "this file", "the file", "this excel", "the spreadsheet", "this csv",
            "oi file", "oi excel", "ঐ ফাইল", "ওই ফাইল", "ঐ excel",
        ]
        return any(ref in q_lower for ref in generic_refs)

    def _detect_cross_source_comparison(self, query: str, sources: List[DataSource]) -> Optional[List[DataSource]]:
        q_lower = query.lower()
        comparison_keywords = ["তুলনা", "compare", "মধ্যে কোন", "which one", "difference", "দুটো", "দুইটার", "both"]
        has_comp_kw = any(kw in q_lower for kw in comparison_keywords)

        seen_filenames = set()
        matched_sources = []
        for s in sources:
            if s.original_filename.lower() in seen_filenames:
                continue
            name_stem = os.path.splitext(s.original_filename.lower())[0]
            clean_stem = re.sub(r"[^a-zA-Z0-9]", "", name_stem)
            if s.original_filename.lower() in q_lower or (len(clean_stem) >= 3 and clean_stem in q_lower):
                seen_filenames.add(s.original_filename.lower())
                matched_sources.append(s)

        if len(matched_sources) >= 2 or (has_comp_kw and len(matched_sources) >= 1 and len(seen_filenames) >= 2):
            return matched_sources if len(matched_sources) >= 2 else sources[:2]
        return None

    def _match_source_by_query_mention(self, query: str, sources: List[DataSource]) -> Optional[DataSource]:
        q_lower = query.lower()
        # 1. Exact full filename match (e.g. "sales.xlsx", "orders.csv")
        for s in sources:
            if s.original_filename.lower() in q_lower:
                return s

        # 2. File stem match ONLY when accompanied by explicit document indicators
        # Avoid matching generic metric words like "sales", "orders", "products" on their own
        doc_markers = ["file", "files", "sheet", "sheets", "ফাইল", "শিট", "শীট", "excel", "এক্সেল", "csv", "xlsx"]
        has_doc_marker = any(m in q_lower for m in doc_markers)

        if has_doc_marker:
            for s in sources:
                stem = os.path.splitext(s.original_filename.lower())[0]
                clean_stem = re.sub(r"[^a-zA-Z0-9]", "", stem)
                if len(clean_stem) >= 3:
                    patterns = [
                        rf"\b{re.escape(stem)}\s*(?:file|files|sheet|sheets|ফাইল|শিট|শীট|excel|এক্সেল|csv|xlsx|এর|er|e|te|তে|এ)",
                        rf"(?:file|files|sheet|sheets|ফাইল|শিট|শীট|excel|এক্সেল)\s*(?:of|from|e|te|এ|তে|এর|er)?\s*\b{re.escape(stem)}\b",
                    ]
                    for p in patterns:
                        if re.search(p, q_lower):
                            return s
        return None

    def _match_by_dataset_name(self, query: str, sources: List[DataSource]) -> Tuple[Optional[DataSource], Optional[Dataset]]:
        q_lower = query.lower()
        sheet_markers = ["sheet", "sheets", "tab", "tabs", "শিট", "শীট", "ট্যাব", "পাতা", "পাতায়"]
        has_sheet_marker = any(m in q_lower for m in sheet_markers)

        for s in sources:
            for d in s.datasets:
                clean_name = d.dataset_name.lower()
                clean_sheet = (d.sheet_name or "").lower()

                # 1. Match when sheet marker is present (e.g. "employees sheet", "sales tab")
                if has_sheet_marker:
                    for name in (clean_name, clean_sheet):
                        if name and len(name) >= 3 and name in q_lower:
                            return s, d

                # 2. Match for multi-sheet workbooks where specific distinctive sheet name is mentioned
                if len(s.datasets) > 1:
                    for name in (clean_name, clean_sheet):
                        if name and len(name) >= 4:
                            if re.search(rf"\b{re.escape(name)}\b", q_lower):
                                return s, d
        return None, None

    def _resolve_datasets_in_source(
        self,
        query: str,
        source: DataSource,
        default_dataset_id: Optional[str] = None,
    ) -> List[Dataset]:
        if not source.datasets:
            return []
        if len(source.datasets) == 1:
            return source.datasets

        q_lower = query.lower()

        for d in source.datasets:
            if d.dataset_name.lower() in q_lower or (d.sheet_name and d.sheet_name.lower() in q_lower):
                return [d]

        scored = []
        for d in source.datasets:
            score = 0
            for col in d.columns:
                col_words = col.replace("_", " ").lower().split()
                for w in col_words:
                    if len(w) >= 3 and w in q_lower:
                        score += 2
            for m in d.numeric_measures:
                if m.replace("_", " ").lower() in q_lower:
                    score += 3
            if score > 0:
                scored.append((score, d))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return [scored[0][1]]

        if default_dataset_id:
            for d in source.datasets:
                if d.dataset_id == default_dataset_id:
                    return [d]

        return source.datasets

    def _is_contextual_followup(self, query: str, context: DocumentAnalyticsContext) -> bool:
        q_lower = query.lower()
        followup_signals = [
            "আর ", "ar ", "and ", "er ta", "এরটা", "এটার", "otar", "ওটার",
            "আগের", "ager", "previous", "same file", "oi file", "ঐ ফাইল",
            "koto?", "কত?", "show ", "what about"
        ]
        if any(sig in q_lower for sig in followup_signals):
            return True
        if len(query.split()) <= 4:
            return True
        return False

    def _score_sources_by_schema_match(self, query: str, sources: List[DataSource]) -> List[Tuple[DataSource, List[Dataset]]]:
        q_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", q_lower))

        results = []
        for s in sources:
            src_score = 0
            matching_datasets = []
            for d in s.datasets:
                d_score = 0
                for col in d.columns:
                    col_clean = col.replace("_", " ").lower()
                    if col_clean in q_lower:
                        d_score += 3
                    for tok in col.split("_"):
                        if len(tok) >= 3 and tok.lower() in q_tokens:
                            d_score += 1
                for m in d.numeric_measures:
                    if m.replace("_", " ").lower() in q_lower:
                        d_score += 3
                if d_score > 0:
                    src_score += d_score
                    matching_datasets.append((d_score, d))

            if src_score >= 3:
                matching_datasets.sort(key=lambda x: x[0], reverse=True)
                selected_dsets = [d for _, d in matching_datasets] if matching_datasets else s.datasets
                results.append((src_score, s, selected_dsets))

        results.sort(key=lambda x: x[0], reverse=True)
        if not results:
            return []

        best_score = results[0][0]
        plausible = [(s, dsets) for score, s, dsets in results if score >= max(3, best_score * 0.7)]
        return plausible

    def _detect_multi_category_comparison(self, query: str, sources: List[DataSource]) -> Optional[SourceResolutionResult]:
        q_lower = query.lower()
        comparison_keywords = ["তুলনা", "compare", "comparison", "versus", "vs", "difference", "with", "এর সাথে", "সাথে", "দুটো", "both"]
        has_comp = any(kw in q_lower for kw in comparison_keywords)
        if not has_comp:
            return None

        intents = self._extract_intent_categories(query)
        distinct_cats = [cat for cat, score in intents if score >= 2.0]
        if len(distinct_cats) >= 2:
            cat_a, cat_b = distinct_cats[0], distinct_cats[1]
            matching_a = []
            matching_b = []
            for s in sources:
                for d in s.datasets:
                    cat = getattr(d.classification, "category", DatasetCategory.UNKNOWN)
                    if cat == cat_a:
                        matching_a.append((s, d))
                    elif cat == cat_b:
                        matching_b.append((s, d))

            if matching_a and matching_b:
                candidate_list = []
                for s, d in matching_a[:2] + matching_b[:2]:
                    candidate_list.append({
                        "source_id": s.source_id,
                        "filename": s.original_filename,
                        "dataset_id": d.dataset_id,
                        "dataset_name": d.dataset_name,
                        "category": d.classification.category.value,
                    })
                msg = (
                    f"ℹ️ **Multi-Category Cross-Dataset Notice**\n\n"
                    f"আপনার প্রশ্নটি দুটি ভিন্ন ধরনের ডেটাসেটের (**{cat_a.value.upper()}** এবং **{cat_b.value.upper()}**) তুলনা বা সমন্বয় নির্দেশ করে। "
                    f"বর্তমান আর্কিটেকচারে একবারে একটি ডেটাসেটের অ্যানালিটিক্স প্রসেস করা হয়। "
                    f"অনুগ্রহ করে যেকোনো একটি ডেটাসেট নির্বাচন করে প্রশ্ন করুন।"
                )
                return SourceResolutionResult(
                    status=ResolutionStatus.UNSUPPORTED_CROSS_SOURCE,
                    candidates=candidate_list,
                    reason=f"Cross-dataset comparison between {cat_a.value} and {cat_b.value} is currently unsupported.",
                    clarification_message=msg,
                    clarification_options=[f"{cat_a.value.title()} ডেটা দেখাও", f"{cat_b.value.title()} ডেটা দেখাও"],
                )
        return None

    def _detect_cross_domain_comparison(self, query: str, sources: List[DataSource]) -> Optional[SourceResolutionResult]:
        q_lower = query.lower()
        comparison_keywords = ["তুলনা", "compare", "vs", "versus", "difference", "between", "দুটো", "দুইটার"]
        if not any(kw in q_lower for kw in comparison_keywords):
            return None

        # Check for multiple domains or entities mentioned
        matched_items: List[Tuple[DataSource, Dataset]] = []
        seen_domains = set()
        for s in sources:
            for d in s.datasets:
                schema = getattr(d, "semantic_schema", None)
                if not schema:
                    continue
                d_domain = (schema.get("domain", "") if isinstance(schema, dict) else (schema.domain or "")).lower()
                raw_entity = schema.get("primary_entity", "") if isinstance(schema, dict) else (schema.primary_entity or "")
                d_entity = raw_entity.lower().replace("_", " ")
                if d_domain and d_domain != "general_business" and d_domain.replace("_", " ") in q_lower:
                    if d_domain not in seen_domains:
                        seen_domains.add(d_domain)
                        matched_items.append((s, d))
                elif d_entity and d_entity in q_lower:
                    if d_entity not in seen_domains:
                        seen_domains.add(d_entity)
                        matched_items.append((s, d))

        if len(matched_items) >= 2:
            s1, d1 = matched_items[0]
            s2, d2 = matched_items[1]
            return SourceResolutionResult(
                status=ResolutionStatus.UNSUPPORTED_CROSS_SOURCE,
                candidates=[
                    {"source_id": s1.source_id, "filename": s1.original_filename, "dataset_id": d1.dataset_id, "dataset_name": d1.dataset_name},
                    {"source_id": s2.source_id, "filename": s2.original_filename, "dataset_id": d2.dataset_id, "dataset_name": d2.dataset_name},
                ],
                reason=f"Cross-dataset comparison between {d1.dataset_name} and {d2.dataset_name} is currently unsupported.",
                clarification_message=f"ℹ️ **Multi-Domain Cross-Dataset Notice**\n\nআপনি একাধিক ভিন্ন বিষয়ের ডেটাসেট ({d1.dataset_name} এবং {d2.dataset_name}) একসাথে তুলনা করার অনুরোধ করেছেন। বর্তমান আর্কিটেকচারে প্রতি টার্নে একটি নির্দিষ্ট ডেটাসেট প্রসেস করা হয়।",
                clarification_options=[f"{d1.dataset_name} থেকে দেখাও", f"{d2.dataset_name} থেকে দেখাও"],
            )
        return None

    def _resolve_by_semantic_schema(
        self,
        query: str,
        sources: List[DataSource],
        has_doc_ind: bool,
    ) -> Optional[SourceResolutionResult]:
        if not has_doc_ind and self._is_plain_entity_query(query):
            return None

        q_lower = query.lower()
        candidates_with_scores = []

        for s in sources:
            for d in s.datasets:
                schema = getattr(d, "semantic_schema", None)
                if not schema:
                    continue

                d_domain = schema.get("domain", "") if isinstance(schema, dict) else (schema.domain or "")
                d_entity = schema.get("primary_entity", "") if isinstance(schema, dict) else (schema.primary_entity or "")
                d_measures = schema.get("measures", []) if isinstance(schema, dict) else schema.measures
                d_dimensions = schema.get("dimensions", []) if isinstance(schema, dict) else schema.dimensions

                score = 0.0
                matched_reasons = []

                # 1. Match domain
                if d_domain and d_domain != "general_business":
                    domain_clean = d_domain.lower().replace("_", " ")
                    if domain_clean in q_lower or any(w in q_lower for w in domain_clean.split() if len(w) >= 4):
                        score += 3.0
                        matched_reasons.append(f"domain:{d_domain}")

                # 2. Match primary entity
                if d_entity:
                    entity_clean = d_entity.lower().replace("_", " ")
                    if entity_clean in q_lower or any(w in q_lower for w in entity_clean.split() if len(w) >= 4):
                        score += 4.0
                        matched_reasons.append(f"entity:{d_entity}")

                # 3. Match measures
                for m in d_measures:
                    m_clean = m.lower().replace("_", " ")
                    m_words = m_clean.split()
                    matched_m = False
                    if re.search(rf"\b{re.escape(m_clean)}\b", q_lower) or (len(m_clean) >= 4 and m_clean in q_lower):
                        matched_m = True
                    elif any(len(mw) >= 3 and mw not in ("the", "all", "per", "for") and (re.search(rf"\b{re.escape(mw)}\b", q_lower) or (len(mw) >= 4 and mw in q_lower)) for mw in m_words):
                        matched_m = True
                    if matched_m:
                        score += 3.0
                        matched_reasons.append(f"measure:{m}")

                # 4. Match dimensions
                for dim in d_dimensions:
                    dim_clean = dim.lower().replace("_", " ")
                    dim_words = dim_clean.split()
                    matched_dim = False
                    if re.search(rf"\b{re.escape(dim_clean)}\b", q_lower) or (len(dim_clean) >= 4 and dim_clean in q_lower):
                        matched_dim = True
                    elif any(len(dw) >= 3 and dw not in ("the", "all", "per", "for") and (re.search(rf"\b{re.escape(dw)}\b", q_lower) or (len(dw) >= 4 and dw in q_lower)) for dw in dim_words):
                        matched_dim = True
                    if matched_dim:
                        score += 2.0
                        matched_reasons.append(f"dimension:{dim}")

                # 5. Columns
                for col in d.columns:
                    col_clean = col.lower().replace("_", " ")
                    col_words = col_clean.split()
                    matched_col = False
                    if re.search(rf"\b{re.escape(col_clean)}\b", q_lower) or (len(col_clean) >= 4 and col_clean in q_lower):
                        matched_col = True
                    elif any(len(cw) >= 3 and cw not in ("the", "all", "per", "for", "item", "data", "file") and (re.search(rf"\b{re.escape(cw)}\b", q_lower) or (len(cw) >= 4 and cw in q_lower)) for cw in col_words):
                        matched_col = True
                    if matched_col:
                        score += 2.0
                        matched_reasons.append(f"col:{col}")

                # 6. Distinctive sheet name or filename
                if d.sheet_name and len(d.sheet_name) >= 3 and d.sheet_name.lower() in q_lower:
                    score += 5.0
                    matched_reasons.append(f"sheet:{d.sheet_name}")
                stem = os.path.splitext(s.original_filename.lower())[0]
                if len(stem) >= 4 and stem in q_lower:
                    score += 5.0
                    matched_reasons.append(f"file:{stem}")

                # Rule: Do not select a dataset solely because its category/domain label matches.
                # Must have concrete evidence from columns, measures, dimensions, entities, or sheet/file name.
                has_concrete_match = any(
                    r.startswith(("measure:", "dimension:", "entity:", "col:", "sheet:", "file:"))
                    for r in matched_reasons
                )

                if score >= 3.0 and has_concrete_match:
                    candidates_with_scores.append((score, s, d, matched_reasons))

        if not candidates_with_scores:
            if len(sources) == 1 and len(sources[0].datasets) == 1:
                single_dataset = sources[0].datasets[0]
                is_compat, missing_reason = self._check_schema_compatibility(query, single_dataset)
                if not is_compat:
                    return SourceResolutionResult(
                        status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                        source_id=sources[0].source_id,
                        source=sources[0],
                        dataset_ids=[single_dataset.dataset_id],
                        selected_datasets=[single_dataset],
                        reason=missing_reason,
                        clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                        clarification_options=[f"{sources[0].original_filename}-এর অন্যান্য তথ্য দেখাও"],
                    )
            return None

        candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
        top_score, top_s, top_d, top_reasons = candidates_with_scores[0]

        # Check schema compatibility for top candidate
        is_compat, missing_reason = self._check_schema_compatibility(query, top_d)
        if not is_compat:
            return SourceResolutionResult(
                status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                source_id=top_s.source_id,
                source=top_s,
                dataset_ids=[top_d.dataset_id],
                selected_datasets=[top_d],
                reason=missing_reason,
                clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                clarification_options=[f"{top_s.original_filename}-এর অন্যান্য তথ্য দেখাও"],
            )

        top_schema = getattr(top_d, "semantic_schema", None)
        top_domain_str = top_schema.get("domain", "business") if isinstance(top_schema, dict) else (getattr(top_schema, "domain", "business") if top_schema else "business")
        top_entity_str = top_schema.get("primary_entity", "entity") if isinstance(top_schema, dict) else (getattr(top_schema, "primary_entity", "entity") if top_schema else "entity")

        if len(candidates_with_scores) == 1:
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=top_s.source_id,
                source=top_s,
                dataset_ids=[top_d.dataset_id],
                selected_datasets=[top_d],
                reason=f"Resolved via semantic schema ({top_domain_str} / {top_entity_str}): {', '.join(top_reasons[:3])}",
            )

        second_score = candidates_with_scores[1][0]
        if top_score - second_score >= 3.0:
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=top_s.source_id,
                source=top_s,
                dataset_ids=[top_d.dataset_id],
                selected_datasets=[top_d],
                reason=f"Resolved via decisive semantic schema match: {top_s.original_filename} [{top_d.dataset_name}]",
            )

        # Ambiguous multiple candidates
        candidate_list = []
        for _, s, d, _ in candidates_with_scores[:4]:
            c_schema = getattr(d, "semantic_schema", None)
            c_domain = c_schema.get("domain", "unknown") if isinstance(c_schema, dict) else (getattr(c_schema, "domain", "unknown") if c_schema else "unknown")
            c_entity = c_schema.get("primary_entity", "unknown") if isinstance(c_schema, dict) else (getattr(c_schema, "primary_entity", "unknown") if c_schema else "unknown")
            candidate_list.append({
                "source_id": s.source_id,
                "filename": s.original_filename,
                "dataset_id": d.dataset_id,
                "dataset_name": d.dataset_name,
                "domain": c_domain,
                "primary_entity": c_entity,
            })
        items = [f"{idx+1}. `{s.original_filename}` ({d.dataset_name})" for idx, (_, s, d, _) in enumerate(candidates_with_scores[:4])]
        items_str = "\n".join(items)
        clarification_msg = (
            f"🤔 **Ambiguous Dataset Context**: একাধিক ফাইলে আপনার প্রশ্নের সাথে সম্পর্কিত ডেটা রয়েছে:\n"
            f"{items_str}\n\n"
            f"আপনি কোন ডেটাসেটের তথ্য দেখতে চাচ্ছেন অনুগ্রহ করে সিলেক্ট করুন।"
        )
        return SourceResolutionResult(
            status=ResolutionStatus.AMBIGUOUS,
            candidates=candidate_list,
            reason="Multiple datasets match the semantic intent of the query. User clarification required.",
            clarification_message=clarification_msg,
            clarification_options=[f"{s.original_filename} ({d.dataset_name}) থেকে দেখাও" for _, s, d, _ in candidates_with_scores[:4]],
        )

    def _resolve_by_business_category_and_intent(
        self,
        query: str,
        sources: List[DataSource],
        has_doc_ind: bool,
    ) -> Optional[SourceResolutionResult]:
        if not has_doc_ind and self._is_plain_entity_query(query):
            return None

        intents = self._extract_intent_categories(query)
        if not intents:
            return None

        primary_cat, intent_score = intents[0]
        if intent_score < 2.0:
            return None

        target_cats = {primary_cat}
        if primary_cat == DatasetCategory.SALES:
            target_cats.add(DatasetCategory.ORDERS)
        elif primary_cat == DatasetCategory.ORDERS:
            target_cats.add(DatasetCategory.SALES)

        matched_candidates: List[Tuple[DataSource, Dataset]] = []
        for s in sources:
            for d in s.datasets:
                d_cat = getattr(d.classification, "category", DatasetCategory.UNKNOWN)
                if d_cat in target_cats:
                    matched_candidates.append((s, d))

        if not matched_candidates:
            return None

        # Exactly 1 candidate dataset in the workspace
        if len(matched_candidates) == 1:
            best_src, best_dataset = matched_candidates[0]
            is_compat, missing_reason = self._check_schema_compatibility(query, best_dataset)
            if not is_compat:
                return SourceResolutionResult(
                    status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                    source_id=best_src.source_id,
                    source=best_src,
                    dataset_ids=[best_dataset.dataset_id],
                    selected_datasets=[best_dataset],
                    reason=missing_reason,
                    clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                    clarification_options=[f"{best_src.original_filename}-এর অন্যান্য তথ্য দেখাও"],
                )

            if has_doc_ind or intent_score >= 2.0:
                return SourceResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    source_id=best_src.source_id,
                    source=best_src,
                    dataset_ids=[best_dataset.dataset_id],
                    selected_datasets=[best_dataset],
                    reason=f"Resolved via business category ({best_dataset.classification.category.value}): {best_src.original_filename} [{best_dataset.dataset_name}]",
                )

        # Multiple candidate datasets
        if len(matched_candidates) > 1:
            q_lower = query.lower()
            generic_words = {
                "sales", "sale", "order", "orders", "product", "products", "item", "items",
                "inventory", "stock", "customer", "customers", "payment", "payments",
                "data", "report", "sheet", "file", "excel", "csv"
            }
            scored_candidates = []
            for s, d in matched_candidates:
                score = 0.0
                if d.sheet_name and d.sheet_name.lower() in q_lower:
                    score += 5.0
                stem = os.path.splitext(s.original_filename.lower())[0]
                if stem not in generic_words and stem in q_lower:
                    score += 5.0
                for col in d.columns:
                    if col.lower() in q_lower:
                        score += 2.0
                scored_candidates.append((score, s, d))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top_score, top_s, top_d = scored_candidates[0]
            second_score = scored_candidates[1][0]

            if top_score >= 5.0 and top_score > second_score:
                is_compat, missing_reason = self._check_schema_compatibility(query, top_d)
                if not is_compat:
                    return SourceResolutionResult(
                        status=ResolutionStatus.INSUFFICIENT_SCHEMA,
                        source_id=top_s.source_id,
                        source=top_s,
                        dataset_ids=[top_d.dataset_id],
                        selected_datasets=[top_d],
                        reason=missing_reason,
                        clarification_message=f"⚠️ **Insufficient Schema**: {missing_reason}",
                    )
                return SourceResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    source_id=top_s.source_id,
                    source=top_s,
                    dataset_ids=[top_d.dataset_id],
                    selected_datasets=[top_d],
                    reason=f"Resolved via category and token match: {top_s.original_filename} [{top_d.dataset_name}]",
                )

            if has_doc_ind or intent_score >= 2.0:
                candidate_list = [
                    {
                        "source_id": s.source_id,
                        "filename": s.original_filename,
                        "dataset_id": d.dataset_id,
                        "dataset_name": d.dataset_name,
                        "category": d.classification.category.value,
                    }
                    for _, s, d in scored_candidates
                ]
                items = [f"{idx+1}. `{s.original_filename}` ({d.dataset_name})" for idx, (_, s, d) in enumerate(scored_candidates[:4])]
                items_str = "\n".join(items)
                clarification_msg = (
                    f"🤔 **Ambiguous Dataset Context**: একাধিক ফাইলে আপনার প্রশ্নের মতো **{primary_cat.value.upper()}** ডেটা রয়েছে:\n"
                    f"{items_str}\n\n"
                    f"আপনি কোন ডেটাসেটের তথ্য দেখতে চাচ্ছেন অনুগ্রহ করে সিলেক্ট করুন।"
                )
                return SourceResolutionResult(
                    status=ResolutionStatus.AMBIGUOUS,
                    candidates=candidate_list,
                    reason=f"Multiple datasets match the '{primary_cat.value}' business category. User clarification required.",
                    clarification_message=clarification_msg,
                    clarification_options=[f"{s.original_filename} ({d.dataset_name}) থেকে দেখাও" for _, s, d in scored_candidates[:4]],
                )

        return None

    def _extract_intent_categories(self, query: str) -> List[Tuple[DatasetCategory, float]]:
        q_lower = query.lower()
        results = []

        specs = [
            (DatasetCategory.PRODUCTS, [
                "product", "products", "item", "items", "catalog", "price", "prices",
                "unit price", "mrp", "cost", "highest price", "cheapest", "most expensive",
                "sku", "brand", "category", "পণ্য", "প্রোডাক্ট", "আইটেম", "দাম", "মূল্য",
                "সর্বোচ্চ দাম", "সবচেয়ে দামি", "সবচেয়ে কম দাম", "ব্র্যান্ড", "ponno", "dam", "mullo"
            ]),
            (DatasetCategory.SALES, [
                "sale", "sales", "sold", "revenue", "gross sales", "net sales", "invoice",
                "last month sales", "total sales", "highest sales", "monthly sales", "today's sales",
                "profit", "net profit", "gross profit", "লাভ", "মুনাফা",
                "বিক্রয়", "বিক্রি", "সেলস", "মোট বিক্রয়", "মোট বিক্রি", "রেভিনিউ", "গত মাসের সেলস",
                "bikkroy", "bikri", "mot sales"
            ]),
            (DatasetCategory.ORDERS, [
                "order", "orders", "ordered", "delivered", "delivery", "shipping",
                "order status", "delivery status", "tracking", "অর্ডার", "অর্ডারসমূহ", "ডেলিভারি", "শিপিং"
            ]),
            (DatasetCategory.INVENTORY, [
                "stock", "inventory", "available stock", "in stock", "quantity in stock",
                "reorder", "out of stock", "warehouse", "left in stock", "মজুদ", "স্টক",
                "কতটি বাকি", "স্টকে আছে", "গুদাম", "stok", "mojud"
            ]),
            (DatasetCategory.CUSTOMERS, [
                "customer", "customers", "client", "clients", "buyer", "who bought",
                "top customer", "phone number", "customer address", "গ্রাহক", "কাস্টমার",
                "ক্রেতা", "ফোন নম্বর", "ঠিকানা", "grahok"
            ]),
            (DatasetCategory.PAYMENTS, [
                "payment", "payments", "paid", "due", "unpaid", "transaction", "gateway",
                "bkash", "nagad", "rocket", "পেমেন্ট", "পরিশোধ", "বকেয়া", "লেনদেন", "বিকাশ", "নগদ"
            ]),
        ]

        for cat, sigs in specs:
            score = 0.0
            for s in sigs:
                if re.search(rf"\b{re.escape(s)}\b", q_lower) or s in q_lower:
                    score += 2.0
            if score >= 2.0:
                results.append((cat, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _check_schema_compatibility(self, query: str, dataset: Dataset) -> Tuple[bool, Optional[str]]:
        q_lower = query.lower()

        # Extract measures from semantic_schema (safe for both model and dict)
        schema = getattr(dataset, "semantic_schema", None)
        schema_measures = []
        if schema:
            if isinstance(schema, dict):
                schema_measures = schema.get("measures", [])
            elif hasattr(schema, "measures"):
                schema_measures = schema.measures or []

        all_known_measures = (
            [c.lower() for c in dataset.columns] +
            [m.lower() for m in schema_measures if isinstance(m, str)] +
            [m.lower() for m in dataset.numeric_measures if isinstance(m, str)]
        )

        # 1. Check known multi-lingual metric dictionary
        unsupported_metrics = {
            "profit": ["profit", "মুনাফা", "লাভ", "net_profit", "gross_profit"],
            "tax": ["tax", "vat", "ট্যাক্স", "ভ্যাট"],
            "shipping": ["shipping", "delivery_charge", "ডেলিভারি চার্জ", "শিপিং"],
            "discount": ["discount", "ছাড়", "ডিসকাউন্ট"],
            "gpa": ["gpa", "grade_point", "জিপিএ"],
            "defect_rate": ["defect_rate", "defect_count", "ত্রুটি"],
            "temperature": ["temperature", "তাপমাত্রা", "temp_celsius"],
            "dosage": ["dosage", "dose", "মাত্রা"],
            "blood_pressure": ["blood_pressure", "systolic", "diastolic", "রক্তচাপ"],
            "billable_hours": ["billable_hours", "hours_billed", "কাজের সময়"],
            "budget": ["budget", "allocated_budget", "বাজেট"],
        }
        for metric_key, keywords in unsupported_metrics.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", q_lower) or kw in q_lower for kw in keywords):
                has_metric = any(metric_key in m or any(kw in m for kw in keywords) for m in all_known_measures)
                if not has_metric:
                    return False, f"The dataset '{dataset.dataset_name}' lacks '{metric_key}' data required for this query."

        # 2. Dynamic domain-agnostic metric intent detection
        metric_candidates = []

        # Pattern A: aggregation keyword + metric noun (e.g. "average score", "highest rate", "total amount")
        m_matches = re.findall(r"\b(?:average|avg|mean|total|sum of|sum|highest|lowest|maximum|max|minimum|min)\s+([a-zA-Z0-9_]+)", q_lower)
        metric_candidates.extend(m_matches)

        # Pattern B: "which record has the highest/lowest <metric>"
        superlative_matches = re.findall(r"\bwhich\s+(?:record|row|item|one|student|patient|batch)?\s*has\s+(?:the\s+)?(?:highest|lowest|maximum|minimum|max|min|best|worst)\s+([a-zA-Z0-9_]+)", q_lower)
        metric_candidates.extend(superlative_matches)

        # Pattern C: "show the <metric> by <dimension>" or "what is the <metric> for/of"
        show_matches = re.findall(r"\b(?:show|find|get|what is)\s+(?:the\s+)?([a-zA-Z0-9_]+)\s+(?:by|for|of|across)\b", q_lower)
        metric_candidates.extend(show_matches)

        # Pattern D: Bengali / Banglish: "গড় <metric>", "সর্বোচ্চ <metric>", "মোট <metric>"
        bn_matches = re.findall(r"\b(?:গড়|সর্বোচ্চ|সর্বনিম্ন|মোট)\s+([a-zA-Z0-9_]+)", q_lower)
        metric_candidates.extend(bn_matches)

        # Non-metric words to skip
        non_metric_words = {
            "record", "records", "row", "rows", "item", "items", "data", "file", "sheet",
            "table", "status", "name", "id", "date", "year", "month", "category", "type",
            "details", "detail", "info", "information", "value", "values", "number", "numbers",
            "entry", "entries", "difference", "comparison", "all", "each", "one", "student",
            "patient", "product", "client", "consultant", "project", "batch", "sales", "order",
            "orders", "customer", "customers"
        }

        for cand in metric_candidates:
            c_clean = cand.strip().lower()
            if len(c_clean) < 3 or c_clean in non_metric_words:
                continue
            has_cand = any(c_clean == m or c_clean in m or m in c_clean for m in all_known_measures)
            if not has_cand:
                return False, f"The dataset '{dataset.dataset_name}' lacks '{c_clean}' data required for this query."

        return True, None

    def _is_plain_entity_query(self, query: str) -> bool:
        q_lower = query.strip().lower()
        if self._has_document_indicators(query):
            return False
        if any(w in q_lower for w in ("file", "sheet", "tab", "excel", "csv", "xlsx", "ফাইল", "শিট", "ডকুমেন্ট")):
            return False

        # Exclude questions that start with query/analytics question words
        analytics_prefixes = (
            "what", "which", "how", "total", "overall", "compare", "list", "show",
            "highest", "lowest", "average", "avg", "sum", "count", "net", "gross",
            "সবচেয়ে", "সর্বমোট", "মোট", "কতগুলো", "কোন", "কার", "কিসের"
        )
        first_word = q_lower.split()[0] if q_lower.split() else ""
        if first_word in analytics_prefixes:
            return False

        # Pattern: Single named entity followed by sales/orders query
        # e.g., "Hasan কত sales করেছে?", "Rahim এর সেলস কত?", "Karim sales"
        # Match where first token is a specific entity name (1 word)
        tokens = query.strip().split()
        if len(tokens) >= 2 and tokens[0][0].isupper() and tokens[0].lower() not in analytics_prefixes:
            if re.search(r"(কত|how much|total)?\s*(sales|বিক্রি|বিক্রয়|সেলস)", q_lower):
                return True

        return False


source_resolver = DataSourceResolver()
