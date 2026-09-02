from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ScriptType(str, Enum):
    BENGALI = "BENGALI"
    LATIN = "LATIN"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class LanguageType(str, Enum):
    BN = "BN"
    BANGLISH = "BANGLISH"
    EN = "EN"
    CODE_MIXED = "CODE_MIXED"


@dataclass
class LanguageProfile:
    script: ScriptType
    language: LanguageType
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinguisticEnrichment:
    original_query: str
    profile: LanguageProfile
    canonical_concepts: List[str] = field(default_factory=list)
    matched_cues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
