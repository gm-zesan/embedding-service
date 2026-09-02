import re
from typing import Dict, List, Set, Tuple


def canonicalize_bengali_unicode(text: str) -> str:
    """
    Step 2C.1: Deterministic Unicode Canonicalization for Bengali script.
    - Strips invisible zero-width characters (\u200B, \u200C, \u200D, \uFEFF).
    - Normalizes composite nukta character sequences into standard single code-points:
        য + nukta (\u09AF\u09BC) -> য় (\u09DF)
        ড + nukta (\u09A1\u09BC) -> ড় (\u09DC)
        ঢ + nukta (\u09A2\u09BC) -> ঢ় (\u09DD)
    - Normalizes Antastha Ya variations (e.g. সময় vs সময়).
    """
    if not text:
        return ""

    # 1. Strip zero-width control characters
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # 2. Canonicalize composite nukta sequences into single precomposed codepoints
    cleaned = cleaned.replace("\u09AF\u09BC", "\u09DF")  # য + nukta -> য়
    cleaned = cleaned.replace("\u09A1\u09BC", "\u09DC")  # ড + nukta -> ড়
    cleaned = cleaned.replace("\u09A2\u09BC", "\u09DD")  # ঢ + nukta -> ঢ়

    # 3. Canonicalize Bengali Antastha Ya in common commerce words to standard precomposed য় (\u09DF)
    cleaned = cleaned.replace("সময়", "সময়")  # Canonical standard \u09DF

    return cleaned


class ScopedCommerceMorphologyNormalizer:
    """
    Step 2C.2: Scoped Commerce Morphology Normalizer.
    Applies inflectional normalization STRICTLY over verified commerce vocabulary.
    Does NOT attempt general Bengali stemming to avoid over-normalization.
    """

    # Verified Commerce Roots
    COMMERCE_ROOTS: Set[str] = {
        "ডেলিভারি", "রিফান্ড", "অর্ডার", "আইটেম", "পণ্য", "ফেরত",
        "পেমেন্ট", "কুরিয়ার", "পার্সেল", "ওয়ারেন্টি", "গ্যারান্টি",
        "হেল্পলাইন", "কাস্টমার", "সার্ভিস", "পরিবর্তন", "বাতিল",
        "এক্সচেঞ্জ", "চার্জ", "ফি", "ঠিকানা", "ডিসকাউন্ট", "কুপন",
        "ইনভয়েস", "ইনভয়েস"
    }

    # Verified Morphological Inflections and Suffix Rules
    # Maps inflected forms or suffixes strictly back to the canonical root
    EXPLICIT_INFLECTIONS: Dict[str, str] = {
        # ডেলিভারি
        "ডেলিভারির": "ডেলিভারি",
        "ডেলিভারিতে": "ডেলিভারি",
        "ডেলিভারিটা": "ডেলিভারি",
        "ডেলিভারিটি": "ডেলিভারি",
        "ডেলিভারিও": "ডেলিভারি",
        "ডেলিভারিতেও": "ডেলিভারি",
        "ডেলিভারী": "ডেলিভারি",
        "ডেলিভারীর": "ডেলিভারি",

        # রিফান্ড
        "রিফান্ডের": "রিফান্ড",
        "রিফান্ডে": "রিফান্ড",
        "রিফান্ডটা": "রিফান্ড",
        "রিফান্ডটি": "রিফান্ড",
        "রিফান্ডও": "রিফান্ড",

        # অর্ডার
        "অর্ডারের": "অর্ডার",
        "অর্ডারে": "অর্ডার",
        "অর্ডারটা": "অর্ডার",
        "অর্ডারটি": "অর্ডার",
        "অর্ডারও": "অর্ডার",
        "অর্ডারগুলোর": "অর্ডার",
        "অর্ডারগুলো": "অর্ডার",

        # আইটেম
        "আইটেমগুলো": "আইটেম",
        "আইটেমগুলোর": "আইটেম",
        "আইটেমগুলি": "আইটেম",
        "আইটেমের": "আইটেম",
        "আইটেমে": "আইটেম",
        "আইটেমটা": "আইটেম",
        "আইটেমটি": "আইটেম",

        # পণ্য
        "পণ্যের": "পণ্য",
        "পণ্যে": "পণ্য",
        "পণ্যটি": "পণ্য",
        "পণ্যটা": "পণ্য",
        "পণ্যগুলো": "পণ্য",
        "পণ্যগুলোর": "পণ্য",

        # ফেরত
        "ফেরতযোগ্য": "ফেরত",
        "ফেরতযোগ্যতা": "ফেরত",
        "ফেরতের": "ফেরত",
        "ফেরতে": "ফেরত",
        "ফেরতটা": "ফেরত",

        # পেমেন্ট (including retail typo পেমেনট)
        "পেমেন্টের": "পেমেন্ট",
        "পেমেন্টে": "পেমেন্ট",
        "পেমেন্টটা": "পেমেন্ট",
        "পেমেন্টও": "পেমেন্ট",
        "পেমেনট": "পেমেন্ট",
        "পেমেনটের": "পেমেন্ট",

        # রিটার্ন (including retail typo রিটারন)
        "রিটার্নের": "রিটার্ন",
        "রিটার্নে": "রিটার্ন",
        "রিটার্নটা": "রিটার্ন",
        "রিটারন": "রিটার্ন",
        "রিটারনের": "রিটার্ন",

        # কুরিয়ার / কুরিয়ার
        "কুরিয়ারের": "কুরিয়ার",
        "কুরিয়ারে": "কুরিয়ার",
        "কুরিয়ারটা": "কুরিয়ার",
        "কুরিয়ার": "কুরিয়ার",
        "কুরিয়ারের": "কুরিয়ার",

        # পার্সেল
        "পার্সেলের": "পার্সেল",
        "পার্সেলে": "পার্সেল",
        "পার্সেলটা": "পার্সেল",
        "পার্সেলটি": "পার্সেল",

        # ওয়ারেন্টি / ওয়ারেন্টি
        "ওয়ারেন্টির": "ওয়ারেন্টি",
        "ওয়ারেন্টিতে": "ওয়ারেন্টি",
        "ওয়ারেন্টিরও": "ওয়ারেন্টি",
        "ওয়ারেন্টি": "ওয়ারেন্টি",
        "ওয়ারেন্টির": "ওয়ারেন্টি",

        # হেল্পলাইন
        "হেল্পলাইনের": "হেল্পলাইন",
        "হেল্পলাইনে": "হেল্পলাইন",
        "হেল্পলাইনটা": "হেল্পলাইন",

        # সার্ভিস
        "সার্ভিসিং": "সার্ভিস",
        "সার্ভিসের": "সার্ভিস",
        "সার্ভিসে": "সার্ভিস",

        # পরিবর্তন
        "পরিবর্তিত": "পরিবর্তন",
        "পরিবর্তনের": "পরিবর্তন",
        "পরিবর্তনে": "পরিবর্তন",

        # কাস্টমার
        "কাস্টমারের": "কাস্টমার",
        "কাস্টমারে": "কাস্টমার",
        "কাস্টমারকে": "কাস্টমার",
    }

    # Suffixes that attach to Bengali nouns
    COMMON_SUFFIXES: Tuple[str, ...] = (
        "গুলোর", "গুলো", "গুলির", "গুলি",
        "ের", "র", "তে", "ে", "টা", "টি", "ও"
    )

    @classmethod
    def normalize_token(cls, token: str) -> str:
        """Normalizes a single token if it represents an inflected commerce root."""
        # 1. Check explicit inflections first (O(1) lookup)
        if token in cls.EXPLICIT_INFLECTIONS:
            return cls.EXPLICIT_INFLECTIONS[token]

        # 2. Check suffix stripping strictly bounded by COMMERCE_ROOTS
        for sfx in cls.COMMON_SUFFIXES:
            if token.endswith(sfx) and len(token) > len(sfx) + 2:
                candidate_stem = token[:-len(sfx)]
                if candidate_stem in cls.COMMERCE_ROOTS:
                    return candidate_stem

        return token

    @classmethod
    def normalize_query_signal(cls, query: str) -> str:
        """
        Produces a normalized signal string for canonical concept extraction.
        DOES NOT MUTATE original_query.
        """
        # Step 1: Unicode canonicalization
        canonical_text = canonicalize_bengali_unicode(query)

        # Step 2: Token-by-token scoped morphology normalization
        tokens = re.split(r"(\s+|[^\w\s\u0980-\u09FF])", canonical_text)
        normalized_tokens = []

        for tok in tokens:
            if re.match(r"^[\u0980-\u09FF]+$", tok):
                normalized_tokens.append(cls.normalize_token(tok))
            else:
                normalized_tokens.append(tok)

        return "".join(normalized_tokens)
