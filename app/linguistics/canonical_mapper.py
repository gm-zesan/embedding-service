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
    CONCEPT_PATTERNS: Dict[str, Dict[str, Any]] = {
        "DELIVERY_TIMELINE": {
            "target_doc_type": "delivery_policy",
            "phrases": [
                "koto din lagbe", "koto din lage", "koto somoy lagbe", "koto somoy lage",
                "kotodine pabo", "kotodine ashbe", "kobe pabo", "kobe ashbe", "kobe deliver",
                "delivery timeline", "delivery timeframe", "delivery time", "delivery schedule",
                "কত দিন লাগবে", "কত দিন লাগে", "কত দিনে পাব", "কত দিনে আসবে", "কবে পাব",
                "কবে আসবে", "ডেলিভারি সময়", "ডেলিভারি হতে কত দিন", "ডেলিভারি পেতে কত দিন",
                "কত দিন সময় লাগবে", "কত দিন সময় লাগবে"
            ],
            "negative_guards": ["cancel", "বাতিল", "refund", "রিফান্ড"],
        },
        "DELIVERY_TRACKING": {
            "target_doc_type": "delivery_policy",
            "phrases": [
                "track korbo", "tracking code", "track kora", "tracking kivabe",
                "parcel kothay", "parcel koi", "parcel track", "order tracking",
                "shipment track", "kuriar tracking", "courier tracking",
                "online e tracking", "tracking korbo",
                "পার্সেল ট্র্যাক", "ট্র্যাকিং কোড", "পার্সেল কোথায়", "পার্সেল কই",
                "কুরিয়ার ট্র্যাকিং", "ট্র্যাক করব", "ট্র্যাক করবো", "ট্র্যাক কীভাবে"
            ],
            "negative_guards": [],
        },
        "DELIVERY_CHARGES": {
            "target_doc_type": "delivery_policy",
            "phrases": [
                "delivery charge koto", "delivery fee koto", "delivery cost koto",
                "shipping charge koto", "shipping cost", "delivery charge", "delivery fee",
                "same day delivery fee", "same day delivery charge", "delivery fee?",
                "charge koto?", "charge koto",
                "ডেলিভারি চার্জ কত", "ডেলিভারি খরচ কত", "ডেলিভারি চার্জ", "ডেলিভারি খরচ",
                "শিপিং চার্জ", "চার্জ কত", "ডেলিভারির ফি কত", "সেইম ডে ডেলিভারির ফি",
                "সেইম ডে ডেলিভারি ফি", "ডেলিভারি ফি কত"
            ],
            "negative_guards": ["ferot", "return", "ফেরত", "রিটার্ন"],
        },
        "RETURN_POLICY": {
            "target_doc_type": "return_policy",
            "phrases": [
                "return kora jabe", "return kora jay", "return korbo kivabe", "return policy ki",
                "return rules", "product return", "item return", "ferot deya jabe",
                "ferot dite hobe", "ferot dite", "shob item ki return", "ferot dite ki delivery fee",
                "ferot nebo kivabe", "ferot dibo", "return policy", "return er niyom",
                "apnader official rules e koto din",
                "রিটার্ন করার নিয়ম", "রিটার্ন পলিসি", "পণ্য ফেরত", "ফেরত দেওয়ার নিয়ম",
                "রিটারন পলিসি", "কোন আইটেমগুলো নন-রিটার্নেবল", "নন-রিটার্নেবল", "ফেরত দিতে পারব",
                "পণ্য রিটার্ন করার", "রিটার্ন করার অফিসিয়াল নিয়ম", "আইটেম নন-রিটার্নেবল"
            ],
            "negative_guards": ["money", "taka", "টাকা", "ক্যাশ", "cash"],
        },
        "REFUND_POLICY": {
            "target_doc_type": "refund_policy",
            "phrases": [
                "taka ferot pabo", "taka refund", "refund pabo kivabe", "refund kivabe pabo", "refund policy",
                "refund process", "cash refund", "money back", "refund er somoy",
                "refund kobe pabo", "refund pete koto din", "refund kivabe",
                "টাকা ফেরত পাব", "রিফান্ড পাব", "রিফান্ড পলিসি", "ক্যাশ রিফান্ড",
                "রিফান্ডের সময়", "টাকা ফেরত", "রিফান্ড হতে কত দিন",
                "হাতে নগদ টাকা", "নগদ টাকা পাওয়া যায়", "নগদ টাকা পাওয়া", "রিফান্ড টাকা"
            ],
            "negative_guards": [],
        },
        "EXCHANGE_POLICY": {
            "target_doc_type": "exchange_policy",
            "phrases": [
                "size na mille", "size change", "exchange kora jabe", "exchange policy",
                "size swap", "choto hoise", "boro hoise", "color change", "item exchange",
                "change kora jabe", "size boro hole", "size choto hole",
                "সাইজ বদলানো", "সাইজ না মিললে", "এক্সচেঞ্জ পলিসি", "এক্সচেঞ্জ করা যাবে",
                "বদলানো যাবে", "অন্য কালার নিতে চাই", "সাইজ ছোট হয়েছে", "সাইজ বড় হয়েছে"
            ],
            "negative_guards": [],
        },
        "PAYMENT_METHOD": {
            "target_doc_type": "payment_policy",
            "phrases": [
                "payment kora jabe", "bkash payment", "nagad payment", "card payment",
                "cod available", "cash on delivery ache", "bKash diye payment",
                "nagad diye payment", "card diye payment", "kivabe payment korbo",
                "payment methods", "payment er niyom",
                "পেমেন্ট মাধ্যম", "বিকাশে পেমেন্ট", "নগদে পেমেন্ট", "ক্যাশ অন ডেলিভারি",
                "কার্ডে পেমেন্ট", "পেমেন্ট নেওয়া হয়", "পেমেন্ট করার নিয়ম", "পেমেনট নেওয়া",
                "বিকাশ পেমেন্ট"
            ],
            # Guard against complaints: "payment problem", "payment issue"
            "negative_guards": ["problem", "issue", "failed", "atkese", "somossha", "ভুল"],
        },
        "WARRANTY_POLICY": {
            "target_doc_type": "warranty_policy",
            "phrases": [
                "warranty ache kina", "guarantee koto din", "warranty koto din",
                "selai khule gele", "selai chute", "defect claim", "warranty policy",
                "service warranty", "product warranty", "button vanga", "botam vanga",
                "button venge gese", "claim korte ki invoice lagbe", "invoice lagbe?",
                "ওয়ারেন্টি কত দিন", "গ্যারান্টি কত দিন", "সেলাই ছুটে যাওয়া", "বোতাম ভাঙা",
                "ওয়ারেন্টি পলিসি", "সার্ভিসিং ফ্রিতে পাব", "ডিসপ্লে নষ্ট",
                "সার্ভিস চার্জ দিতে হবে", "সার্ভিস চার্জ", "সার্ভিসিং ফ্রিতে"
            ],
            "negative_guards": [],
        },
        "CANCELLATION_POLICY": {
            "target_doc_type": "cancellation_policy",
            "phrases": [
                "order cancel korbo", "cancel kora jabe", "cancel kora jay", "cancel policy", "order batil",
                "cancel korbo kivabe", "order cancellation", "cancel before shipping", "dispatch hole ki cancel",
                "অর্ডার বাতিল", "বাতিল করার নিয়ম", "ক্যানসেল পলিসি", "ক্যানসেল করা যাবে",
                "অর্ডার ক্যান্সেল"
            ],
            "negative_guards": [],
        },
        "PRIVACY_POLICY": {
            "target_doc_type": "privacy_policy",
            "phrases": [
                "data secured", "data security", "third party k deya", "third party",
                "secured thakbe apnader kache", "phone number ki third party",
                "তথ্য সুরক্ষা", "থার্ড পার্টি", "ডাটা সুরক্ষা"
            ],
            "negative_guards": [],
        },
        "SOCIAL_MEDIA_POLICY": {
            "target_doc_type": "social_media_policy",
            "phrases": [
                "messenger e ki bkash pin", "bkash pin ba otp", "otp chaowa hoy",
                "মেসেঞ্জারে কি পিন", "পিন বা পাসওয়ার্ড"
            ],
            "negative_guards": [],
        },
        "CONTACT_SUPPORT": {
            "target_doc_type": "contact",
            "phrases": [
                "official whatsapp", "whatsapp number", "official helpline",
                "customer care number", "কাস্টমার কেয়ার হেল্পলাইন", "হোয়াটসঅ্যাপ নম্বর",
                "কাস্টমার কেয়ার হেল্পলাইন", "কার সাথে যোগাযোগ করব", "কাস্টমার কেয়ার", "কাস্টমার কেয়ার"
            ],
            "negative_guards": [],
        },
        "TERMS_POLICY": {
            "target_doc_type": "terms",
            "phrases": [
                "price ki notice chara", "price change hote pare", "dam change",
                "terms and conditions", "শর্তাবলী",
                "পরিবর্তিত হতে পারে", "মূল্য কি যেকোনো সময়", "মূল্য কি যেকোনো সময়"
            ],
            "negative_guards": [],
        },
    }

    @classmethod
    def map_concepts(cls, query: str, profile: LanguageProfile) -> LinguisticEnrichment:
        q_lower = query.lower()
        matched_concepts: List[str] = []
        matched_cues: List[str] = []

        # Step 2C.3: Generate auxiliary normalized signal for Bengali script
        # Original user query remains 100% immutable!
        normalized_signal = ""
        if profile.script in (ScriptType.BENGALI, ScriptType.MIXED) or profile.language in (LanguageType.BN, LanguageType.CODE_MIXED):
            normalized_signal = ScopedCommerceMorphologyNormalizer.normalize_query_signal(query).lower()

        for concept_name, defn in cls.CONCEPT_PATTERNS.items():
            # Check negative guards first (e.g. payment problem != payment method)
            if any(neg in q_lower for neg in defn["negative_guards"]):
                continue
            if normalized_signal and any(neg in normalized_signal for neg in defn["negative_guards"]):
                continue

            # Check positive phrase matches against both raw query and normalized signal
            for phrase in defn["phrases"]:
                phrase_lower = phrase.lower()
                if phrase_lower in q_lower or (normalized_signal and phrase_lower in normalized_signal):
                    matched_concepts.append(concept_name)
                    matched_cues.append(phrase)
                    break

        return LinguisticEnrichment(
            original_query=query,
            profile=profile,
            canonical_concepts=list(set(matched_concepts)),
            matched_cues=matched_cues,
            metadata={
                "target_doc_types": [cls.CONCEPT_PATTERNS[c]["target_doc_type"] for c in matched_concepts],
                "normalized_signal": normalized_signal if normalized_signal else None,
            },
        )
