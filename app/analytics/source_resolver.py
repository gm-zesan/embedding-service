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

        # Step 1: Explicit Source ID in Request
        if explicit_source_id:
            matched_src = self.registry.get_source(workspace_id, explicit_source_id)
            if matched_src:
                selected_datasets = self._resolve_datasets_in_source(clean_query, matched_src)
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

        # Step 5: Semantic Keyword & Column Match Across All Sources
        matching_sources = self._score_sources_by_schema_match(clean_query, all_sources)

        if len(matching_sources) == 1:
            best_src, datasets = matching_sources[0]
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=best_src.source_id,
                source=best_src,
                dataset_ids=[d.dataset_id for d in datasets],
                selected_datasets=datasets,
                reason=f"Resolved via strong semantic schema match: {best_src.original_filename}",
            )

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

        if len(all_sources) == 1:
            single_src = all_sources[0]
            selected_datasets = self._resolve_datasets_in_source(clean_query, single_src)
            return SourceResolutionResult(
                status=ResolutionStatus.RESOLVED,
                source_id=single_src.source_id,
                source=single_src,
                dataset_ids=[d.dataset_id for d in selected_datasets],
                selected_datasets=selected_datasets,
                reason=f"Resolved to the single available workspace source: {single_src.original_filename}",
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
        for s in sources:
            if s.original_filename.lower() in q_lower:
                return s

        for s in sources:
            stem = os.path.splitext(s.original_filename.lower())[0]
            clean_stem = re.sub(r"[^a-zA-Z0-9]", "", stem)
            if len(clean_stem) >= 3:
                patterns = [
                    rf"\b{re.escape(stem)}\b",
                    rf"{re.escape(clean_stem)}\s*(?:file|sheet|ফাইল|এর|e|te|the)",
                ]
                for p in patterns:
                    if re.search(p, q_lower):
                        return s
        return None

    def _match_by_dataset_name(self, query: str, sources: List[DataSource]) -> Tuple[Optional[DataSource], Optional[Dataset]]:
        q_lower = query.lower()
        for s in sources:
            for d in s.datasets:
                clean_name = d.dataset_name.lower()
                clean_sheet = (d.sheet_name or "").lower()
                if (clean_name in q_lower and len(clean_name) >= 4) or (clean_sheet in q_lower and len(clean_sheet) >= 4):
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


source_resolver = DataSourceResolver()
