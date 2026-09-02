import re
from typing import Any, Dict, List, Optional, Set
from app.linguistics.models import LanguageProfile, LanguageType, LinguisticEnrichment, ScriptType
from app.linguistics.scoped_morphology import ScopedCommerceMorphologyNormalizer


class CanonicalConceptMapper:
    """
    Phrase-to-Concept Commerce Mapper.
    Extracts canonical commerce concepts without destructive query rewriting.
    Guards strictly against single-word false positive over-normalization.
    """

    # Structured Concept Definitions with Contextual Phrase Patterns

    @classmethod
    def _map_concepts_internal(cls, query: str, profile: LanguageProfile, patterns_dict: Dict[str, Dict[str, Any]]) -> LinguisticEnrichment:
        q_lower = query.lower()
        matched_concepts: List[str] = []
        matched_cues: List[str] = []

        normalized_signal = ""
        if profile.script in (ScriptType.BENGALI, ScriptType.MIXED) or profile.language in (LanguageType.BN, LanguageType.CODE_MIXED):
            normalized_signal = ScopedCommerceMorphologyNormalizer.normalize_query_signal(query).lower()

        for concept_name, defn in patterns_dict.items():
            if any(neg in q_lower for neg in defn.get("negative_guards", [])):
                continue
            if normalized_signal and any(neg in normalized_signal for neg in defn.get("negative_guards", [])):
                continue

            for phrase in defn.get("phrases", defn.get("positive_phrases", [])):
                phrase_lower = phrase.lower()
                if phrase_lower in q_lower or (normalized_signal and phrase_lower in normalized_signal):
                    matched_concepts.append(concept_name)
                    matched_cues.append(phrase)
                    break

        target_doc_types = [patterns_dict[c].get("target_doc_type") for c in matched_concepts if patterns_dict[c].get("target_doc_type")]

        return LinguisticEnrichment(
            original_query=query,
            profile=profile,
            canonical_concepts=sorted(list(set(matched_concepts))),
            matched_cues=matched_cues,
            metadata={
                "target_doc_types": sorted(list(set(target_doc_types))),
                "normalized_signal": normalized_signal if normalized_signal else None,
            },
        )

    @classmethod
    def map_concepts(cls, query: str, profile: LanguageProfile, snapshot: Optional[Dict[str, Any]] = None, telemetry: Optional[Dict[str, Any]] = None) -> LinguisticEnrichment:
        import logging
        logger = logging.getLogger(__name__)
        
        concept_patterns = snapshot.get("concept_patterns", {}) if snapshot else {}
        authoritative_enrichment = cls._map_concepts_internal(query, profile, concept_patterns)
            
        return authoritative_enrichment

