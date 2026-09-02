import re
from typing import Dict, List, Set
from app.linguistics.models import LanguageProfile, LanguageType, ScriptType


class LanguageProfiler:
    """
    Deterministic language and script profiler.
    Runs in < 0.1ms without external model calls.
    """

    # Common Banglish phonetic particles, pronouns, and verbs
    BANGLISH_MARKERS: Set[str] = {
        "koto", "kotodin", "kivabe", "pabo", "kora", "jabe", "ache", "hobe",
        "korbo", "kori", "parbo", "deya", "shob", "kono", "ki", "na",
        "apnader", "amar", "apnara", "taka", "ferot", "dam", "khulbo", "kena",
        "choto", "boro", "noshto", "vule", "khule", "pore", "shathe", "kothay",
        "kon", "konta", "chai", "lagbe", "niye", "dibo", "hoise", "hoise?",
        "niben", "parben", "bolun", "ashbe", "pathaben", "kinbo", "lagle",
        "bujhbo", "korle", "dile", "peye", "bistarito", "niyom", "shunsi",
        "thakbe", "milse", "mile", "mille", "dekhe", "debo"
    }

    # Common English functional words
    ENGLISH_STOPWORDS: Set[str] = {
        "the", "is", "are", "can", "how", "what", "where", "when", "which",
        "my", "your", "our", "does", "do", "will", "would", "for", "about",
        "have", "has", "i", "you", "we", "they", "if", "and", "or", "to",
        "in", "on", "at", "with", "after", "before", "from", "any", "please"
    }

    @classmethod
    def profile(cls, query: str) -> LanguageProfile:
        q = query.strip()
        if not q:
            return LanguageProfile(
                script=ScriptType.UNKNOWN,
                language=LanguageType.EN,
                confidence=1.0,
                features={"char_count": 0},
            )

        # 1. Script Character Analysis
        bengali_chars = len(re.findall(r"[\u0980-\u09FF]", q))
        latin_chars = len(re.findall(r"[a-zA-Z]", q))
        total_alpha = bengali_chars + latin_chars

        if total_alpha == 0:
            return LanguageProfile(
                script=ScriptType.UNKNOWN,
                language=LanguageType.EN,
                confidence=0.5,
                features={"total_alpha": 0},
            )

        bengali_ratio = bengali_chars / total_alpha
        latin_ratio = latin_chars / total_alpha

        # 2. Script Type Determination
        if bengali_ratio >= 0.90:
            script = ScriptType.BENGALI
        elif latin_ratio >= 0.90:
            script = ScriptType.LATIN
        else:
            script = ScriptType.MIXED

        # 3. Language Type Determination
        words = [w.lower() for w in re.findall(r"[a-zA-Z\u0980-\u09FF]+", q)]
        features = {
            "bengali_chars": bengali_chars,
            "latin_chars": latin_chars,
            "bengali_ratio": round(bengali_ratio, 3),
            "latin_ratio": round(latin_ratio, 3),
            "word_count": len(words),
        }

        if script == ScriptType.BENGALI:
            # Check if there are English words embedded
            latin_words = [w for w in words if re.match(r"^[a-zA-Z]+$", w)]
            if len(latin_words) >= 2 or (len(latin_words) == 1 and len(latin_words[0]) > 3 and latin_words[0] not in {"bkash", "nagad", "cod", "faq"}):
                language = LanguageType.CODE_MIXED
                confidence = 0.92
            else:
                language = LanguageType.BN
                confidence = 0.98
        elif script == ScriptType.LATIN:
            banglish_hits = [w for w in words if w in cls.BANGLISH_MARKERS]
            english_hits = [w for w in words if w in cls.ENGLISH_STOPWORDS]

            features["banglish_hits"] = len(banglish_hits)
            features["english_hits"] = len(english_hits)

            if len(banglish_hits) > 0:
                # If it has Banglish verbs/particles, it is Banglish (even if it has English nouns like "delivery" or "return")
                language = LanguageType.BANGLISH
                confidence = min(0.99, 0.85 + (len(banglish_hits) * 0.05))
            elif len(english_hits) >= 2 or (len(english_hits) == 1 and len(words) <= 3):
                language = LanguageType.EN
                confidence = 0.95
            else:
                # Ambiguous short Latin query: check if words look phonetically Banglish or English
                # Default to English unless common Banglish suffix or patterns appear
                language = LanguageType.EN
                confidence = 0.70
        else:
            # Script is MIXED
            language = LanguageType.CODE_MIXED
            confidence = 0.95

        return LanguageProfile(
            script=script,
            language=language,
            confidence=confidence,
            features=features,
        )
