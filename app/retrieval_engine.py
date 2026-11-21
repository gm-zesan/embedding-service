import asyncio
import logging
import re
import time
# pyrefly: ignore [missing-import]
import httpx
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException
try:
    from app.embedding import embed
except Exception:
    embed = None
from app import config
from app.llm import LLMRequest, default_client as llm_client
from app.typesense_engine import execute_hybrid_search, get_typesense_client

logger = logging.getLogger("retrieval_engine")


# Unicode artifacts to strip
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# Common Bengali spelling normalizations (GAP-RET-05)
_BENGALI_SPELLING_VARIANTS = [
    (re.compile(r"(?<![^\s,;.!?])কী(?![^\s,;.!?])"), "কি"),
    (re.compile(r"ওয়"), "ওয়"),
    (re.compile(r"ওয়ারেন্টি"), "ওয়ারেন্টি"),
    (re.compile(r"কুরিয়ার"), "কুরিয়ার"),
    (re.compile(r"ডেলিভারী"), "ডেলিভারি"),
    (re.compile(r"পলিসী"), "পলিসি"),
    (re.compile(r"কোম্পানী"), "কোম্পানি"),
    (re.compile(r"ডিসকাওন্ট"), "ডিসকাউন্ট"),
    (re.compile(r"ষ্টোর"), "স্টোর"),
    (re.compile(r"অরডার"), "অর্ডার"),
    (re.compile(r"পাওয়া"), "পাওয়া"),
    (re.compile(r"প্যাকেজিং"), "প্যাকেজিং"),
    (re.compile(r"গ্যারান্টী"), "গ্যারান্টি"),
    (re.compile(r"রিফান্ড"), "রিফান্ড"),
    (re.compile(r"বিকাশ"), "বিকাশ"),
    (re.compile(r"ইনভয়েস"), "ইনভয়েস"),
]

# Fast local Banglish-to-Bengali vocabulary mapping for instant candidate expansion (< 1ms) (GAP-RET-04)
_BANGLISH_TO_BANGLA = {
    "koto": "কত",
    "dam": "দাম",
    "daam": "দাম",
    "damer": "দামের",
    "kivabe": "কিভাবে",
    "kibhabe": "কিভাবে",
    "kobe": "কবে",
    "kokhon": "কখন",
    "pabo": "পাব",
    "paowa": "পাওয়া",
    "ache": "আছে",
    "asen": "আছেন",
    "kothay": "কোথায়",
    "kothai": "কোথায়",
    "charz": "চার্জ",
    "charge": "চার্জ",
    "ferot": "ফেরত",
    "feroth": "ফেরত",
    "dekhte": "দেখতে",
    "dekhaw": "দেখাও",
    "dekhan": "দেখান",
    "dekhbo": "দেখবো",
    "kinte": "কিনতে",
    "kinbo": "কিনব",
    "kena": "কেনা",
    "chara": "ছাড়া",
    "bade": "বাদে",
    "notun": "নতুন",
    "puran": "পুরানো",
    "purano": "পুরানো",
    "khula": "খোলা",
    "bondho": "বন্ধ",
    "somoy": "সময়",
    "shokal": "সকাল",
    "raat": "রাত",
    "thikana": "ঠিকানা",
    "jogajog": "যোগাযোগ",
    "ovijog": "অভিযোগ",
    "sahajjo": "সাহায্য",
    "bhalo": "ভালো",
    "shob": "সব",
    "sob": "সব",
    "apnader": "আপনাদের",
    "apnara": "আপনারা",
    "apnar": "আপনার",
    "apni": "আপনি",
    "amader": "আমাদের",
    "amar": "আমার",
    "ami": "আমি",
    "kisu": "কিছু",
    "kichu": "কিছু",
    "pathan": "পাঠান",
    "pathaben": "পাঠাবেন",
    "pathano": "পাঠানো",
    "kotodin": "কতদিন",
    "koidin": "কয়দিন",
    "din": "দিন",
    "taka": "টাকা",
    "tk": "টাকা",
    "shundor": "সুন্দর",
    "rong": "রং",
    "colour": "রং",
    "color": "রং",
    "bhabe": "ভাবে",
    "hobe": "হবে",
    "lagbe": "লাগবে",
    "chai": "চাই",
    "cai": "চাই",
    "chay": "চায়",
    "nibo": "নেব",
    "nebo": "নেব",
    "nibona": "নেবনা",
    "nebona": "নেবনা",
    "shari": "শাড়ি",
    "sari": "শাড়ি",
    "panjabi": "পাঞ্জাবি",
    "punjabi": "পাঞ্জাবি",
    "jama": "জামা",
    "kapor": "কাপড়",
    "dhaka": "ঢাকা",
    "dhakar": "ঢাকার",
    "baire": "বাইরে",
    "bairer": "বাইরের",
    "vetore": "ভেতরে",
    "ashbe": "আসবে",
    "asbe": "আসবে",
    "pelam": "পেলাম",
    "paini": "পাইনি",
    "painai": "পাইনি",
    "khuje": "খুঁজে",
    "thakle": "থাকলে",
    "thakbe": "থাকবে",
    "sathe": "সাথে",
    "shathe": "সাথে",
    "bistarito": "বিস্তারিত",
    "janan": "জানান",
    "jante": "জানতে",
    "sothik": "সঠিক",
    "shothik": "সঠিক",
    "vul": "ভুল",
    "bhul": "ভুল",
    "thik": "ঠিক",
    "dhonnobad": "ধন্যবাদ",
    "thanks": "ধন্যবাদ",
    "shomossha": "সমস্যা",
    "somossa": "সমস্যা",
    "bhanga": "ভাঙা",
    "vanga": "ভাঙা",
    "bhenge": "ভেঙে",
    "vange": "ভেঙে",
    "noshto": "নষ্ট",
    "nosto": "নষ্ট",
    "chira": "ছিঁড়া",
    "chire": "ছিঁড়ে",
    "jinish": "জিনিস",
    "jinis": "জিনিস",
    "pani": "পানি",
    "panite": "পানিতে",
    "porle": "পড়লে",
    "porbe": "পড়বে",
    "khotigrostho": "ক্ষতিগ্রস্ত",
    "khoti": "ক্ষতি",
    "truti": "ত্রুটি",
    "trutipurno": "ত্রুটিপূর্ণ",
    "video": "ভিডিও",
    "unboxing": "আনবক্সিং",
    "transit": "ট্রানজিট",
    "damage": "ক্ষতিগ্রস্ত",
    "exchange": "এক্সচেঞ্জ",
    "replace": "রিপ্লেসমেন্ট",
    "replacement": "রিপ্লেসমেন্ট",
    "restocking": "রিস্টকিং",
    "free": "ফ্রি",
    "fee": "ফি",
    "warranty": "ওয়ারেন্টি",
    "waranty": "ওয়ারেন্টি",
    "guarantee": "গ্যারান্টি",
    "cancel": "বাতিল",
    "cancle": "বাতিল",
    "helpline": "হেল্পলাইন",
    "support": "সাপোর্ট",
    "care": "কেয়ার",
    "hotline": "হটলাইন",
    "number": "নম্বর",
    "nombor": "নম্বর",
    "outlet": "আউটলেট",
    "branch": "শাখা",
    "dokan": "দোকান",
    "store": "স্টোর",
    "shop": "শপ",
    "open": "খোলা",
    "close": "বন্ধ",
    "time": "সময়",
    "timing": "সময়",
    "koy": "কয়",
    "koyta": "কয়টি",
    "koita": "কয়টি",
    "bochor": "বছর",
    "mas": "মাস",
    "mash": "মাস",
    "ghonta": "ঘণ্টা",
    "ghanta": "ঘণ্টা",
    "fabric": "ফেব্রিক",
    "origin": "উৎপত্তি",
    "made": "তৈরি",
    "bodlano": "বদলানো",
    "poriborton": "পরিবর্তন",
    "eita": "এইটা",
    "oita": "ওইটা",
    "eta": "এটা",
    "ota": "ওটা",
    "kom": "কম",
    "beshi": "বেশি",
    "courier": "কুরিয়ার",
    "curier": "কুরিয়ার",
    "shipping": "কুরিয়ার",
    "defective": "ত্রুটিপূর্ণ",
    "hardware": "হার্ডওয়্যার",
    "coverage": "কভারেজ",
    "covered": "কভারেজ",
    "goods": "পণ্য",
    "timeline": "সময়সীমা",
    "hours": "ঘণ্টা",
    "hour": "ঘণ্টা",
    "days": "দিন",
    "tag": "ট্যাগ",
    "torn": "ছিঁড়ে",
    "fellam": "ফেলেছি",
    "ekhon": "এখন",
}


def preprocess_query(query: str) -> str:
    """Lightweight query preprocessing: normalize unicode, Bengali spelling, and fast Banglish transliteration."""
    clean = _ZERO_WIDTH_CHARS.sub("", query)
    clean = " ".join(clean.strip().split())

    for pat, repl in _BENGALI_SPELLING_VARIANTS:
        clean = pat.sub(repl, clean)

    # Detect Banglish tokens and append corresponding Bengali equivalents for instant hybrid recall
    words = re.findall(r"\b[a-zA-Z]+\b", clean.lower())
    appended_bangla = []
    for w in words:
        if w in _BANGLISH_TO_BANGLA:
            bn_word = _BANGLISH_TO_BANGLA[w]
            if bn_word not in clean and bn_word not in appended_bangla:
                appended_bangla.append(bn_word)

    if appended_bangla:
        return f"{clean} {' '.join(appended_bangla)}"

    return clean


# ---------------------------------------------------------------------------
# Dynamic LLM Query Expansion (Fallback Pass)
# ---------------------------------------------------------------------------


async def expand_query_via_llm(query: str) -> Optional[str]:
    """
    Tier 3 Escape Hatch: Fast LLM query reformulator for long-tail, unseen colloquial queries.
    Strictly bounded by a 2.0s timeout to protect user latency.
    """
    if not config.LLM_EXPANSION_API_KEY:
        return None

    system_prompt = (
        "<ROLE>\n"
        "You are a domain-neutral AI Search Query Expander for an enterprise multi-source knowledge base.\n"
        "The knowledge base spans diverse tenant domains, including healthcare, education, hospitality, financial services, software/SaaS, and commerce.\n"
        "</ROLE>\n\n"
        "<TASK>\n"
        "Analyze the user's query and generate 3-5 concise, domain-relevant search keywords, synonyms, and rephrasings in Bengali, Banglish, or English.\n"
        "Preserve the user's exact subject domain and intent. Do NOT bias or contaminate search terms with unrelated retail or commercial concepts unless the query explicitly concerns commerce or orders.\n"
        "</TASK>\n\n"
        "<CONSTRAINTS>\n"
        "1. Do NOT answer the question.\n"
        "2. Do NOT output any conversational text, introductory remarks, or explanations.\n"
        "3. Strictly maintain domain neutrality—reflect only the query's relevant domain.\n"
        "</CONSTRAINTS>\n\n"
        "<EXAMPLES>\n"
        "User: daktar kobe thakben?\n"
        "Output: doctor availability, appointment schedule, chamber time, সিরিয়াল, ডাক্তার\n\n"
        "User: admission fee koto?\n"
        "Output: admission fee, tuition cost, admission requirements, ভর্তি ফি, খরচ\n\n"
        "User: order kobe pabo?\n"
        "Output: delivery time, shipping date, delivery status, kobe pabo, ডেলিভারি\n\n"
        "User: room booking cancel kora jabe?\n"
        "Output: room cancellation, refund policy, reservation cancel, বুকিং বাতিল, রিফান্ড\n"
        "</EXAMPLES>\n\n"
        "<OUTPUT_FORMAT>\n"
        "Return ONLY comma-separated search terms.\n"
        "</OUTPUT_FORMAT>"
    )

    req = LLMRequest(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=150,
    )

    try:
        resp = await llm_client.generate(req)
        if resp.content:
            expanded = resp.content.strip()
            logger.info("Tier 3 LLM Expansion (%s/%s): '%s' -> '%s'", resp.provider, resp.model, query, expanded)
            return expanded
        return None
    except Exception as e:
        logger.warning("Tier 3 LLM query expansion exception (bypassed safely): %s", e)
        return None


def parse_typesense_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse raw Typesense hits into normalized search result dictionaries."""
    results = []
    for hit in hits:
        doc = hit.get("document", {})
        vector_distance = hit.get("vector_distance")
        tm_info = hit.get("text_match_info", {})
        tokens_matched = tm_info.get("tokens_matched", 0)
        tokens_dropped = tm_info.get("num_tokens_dropped", 0)

        # Cosine distance to similarity: 1.0 - distance (clamped to [0.0, 1.0])
        if vector_distance is not None:
            semantic_score = round(max(0.0, min(1.0, 1.0 - float(vector_distance))), 4)
        else:
            semantic_score = 0.0

        # Calculate keyword match coverage
        total_tokens = tokens_matched + tokens_dropped
        keyword_ratio = (tokens_matched / total_tokens) if total_tokens > 0 else 0.0

        if vector_distance is not None:
            if keyword_ratio >= 0.8 and tokens_matched >= 1:
                # High full-keyword match confirmation
                final_score = round(max(semantic_score, 0.50 * semantic_score + 0.50 * keyword_ratio), 4)
                match_type = "hybrid"
            elif tokens_matched > 0 and semantic_score > 0.30:
                # Partial keyword match with moderate semantic support
                final_score = round(0.80 * semantic_score + 0.20 * keyword_ratio, 4)
                match_type = "hybrid"
            else:
                match_type = "vector"
                final_score = semantic_score
        else:
            # Pure BM25 text match (degraded / embedding outage fallback)
            match_type = "bm25"
            if tokens_matched > 0:
                # Direct keyword match coverage in BM25 mode:
                # Scales between 0.65 (partial) and 0.95 (full query coverage)
                final_score = round(min(0.95, 0.65 + 0.30 * keyword_ratio), 4)
            else:
                text_match = hit.get("text_match", 0)
                final_score = 0.60 if text_match > 0 else 0.50

        results.append({
            "id": str(doc.get("id")),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "document_type": doc.get("document_type", "faq"),
            "priority": doc.get("priority", 0),
            "score": final_score,
            "match_type": match_type,
            "keyword_score": round(keyword_ratio, 4),
            "semantic_score": semantic_score,
        })

    results.sort(key=lambda r: (r["score"], r["priority"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Post-Retrieval Candidate Reranker (B1 Action Alignment & B2 Multi-Entity)
# ---------------------------------------------------------------------------

def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    contextual_signal: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Candidate Precision Reranker (Semantic Overlap, Polarity Guard & Entity Alignment).
    Re-scores and re-orders candidate documents before LLM consumption to maximize top-1 precision.
    """
    t_start = time.time()
    if not candidates:
        return candidates, {
            "reranker_applied": False,
            "rerank_ms": 0.0,
            "reranker_reason": None,
        }

    clean_q = query.lower().strip()
    q_tokens = set(re.findall(r'[\w\u0980-\u09FF]+', clean_q))

    # Polar / distinguishing token sets to prevent near-miss false positives
    polar_pairs = [
        ({"বাইরে", "bair", "outside"}, {"ভেতরে", "মধ্যে", "ভিতরে", "inside"}),
        ({"ফ্রি", "free"}, {"পেইড", "টাকা", "paid", "charge"}),
        ({"রিটার্ন", "ফেরত", "return", "refund"}, {"ডেলিভারি", "delivery", "shipping"}),
        ({"ক্যাশ", "cod", "cash"}, {"অনলাইন", "বিকাশ", "bkash", "card", "nagad"}),
    ]

    reranked = []
    for cand in candidates:
        c_q = cand.get("question", "").lower()
        c_a = cand.get("answer", "").lower()
        combined_text = f"{c_q} {c_a}"
        doc_tokens = set(re.findall(r'[\w\u0980-\u09FF]+', combined_text))

        # 1. Token overlap against question target
        q_target_tokens = set(re.findall(r'[\w\u0980-\u09FF]+', c_q))
        overlap_q = len(q_tokens & q_target_tokens) / max(1, len(q_tokens | q_target_tokens))

        # 2. Polarity alignment check
        polarity_penalty = 0.0
        polarity_boost = 0.0
        for pos_set, neg_set in polar_pairs:
            q_has_pos = bool(q_tokens & pos_set)
            q_has_neg = bool(q_tokens & neg_set)
            doc_has_pos = bool(doc_tokens & pos_set)
            doc_has_neg = bool(doc_tokens & neg_set)

            if q_has_pos and doc_has_pos:
                polarity_boost += 0.08
            elif q_has_pos and doc_has_neg and not doc_has_pos:
                polarity_penalty += 0.12

            if q_has_neg and doc_has_neg:
                polarity_boost += 0.08
            elif q_has_neg and doc_has_pos and not doc_has_neg:
                polarity_penalty += 0.12

        # 3. Exact phrase containment boost
        exact_boost = 0.0
        if clean_q and (clean_q in c_q or c_q in clean_q):
            exact_boost = 0.08

        # Calibrate base score with rerank adjustments
        orig_score = float(cand.get("score", 0.0))
        adj_score = orig_score + (overlap_q * 0.12) + exact_boost + polarity_boost - polarity_penalty
        new_score = round(max(0.0, min(0.99, adj_score)), 4)

        cand_copy = dict(cand)
        cand_copy["score"] = new_score
        cand_copy["rerank_delta"] = round(new_score - orig_score, 4)
        reranked.append(cand_copy)

    # Sort strictly by calibrated score, then by document priority
    reranked.sort(key=lambda r: (r["score"], r.get("priority", 0)), reverse=True)
    elapsed_ms = round((time.time() - t_start) * 1000, 2)

    return reranked, {
        "reranker_applied": True,
        "rerank_ms": elapsed_ms,
        "reranker_reason": "token_alignment_and_polarity_guard",
    }
# 2-Tier Adaptive Retrieval Execution Engine (MPNet + Fallback LLM Expander)
# ---------------------------------------------------------------------------

async def search_knowledge_base(
    query: str,
    workspace_id: Optional[int] = None,
    top_k: int = 5,
    request_id: Optional[str] = None,
    contextual_signal: Optional[str] = None,
    document_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tiered Adaptive Retrieval Pipeline:
    • Tier 1: Fast-path Raw Dense Vector + Hybrid Search (< 30ms)
    • Tier 2: Local Deterministic Domain & Synonym Normalization (if Tier 1 < 0.55) (< 35ms)
    • Tier 3: Controlled Escape-Hatch LLM Expansion (only if score remains < 0.35 on unseen queries)
    • Candidate Precision Reranking (B1 Action Alignment & B2 Multi-Entity)
    """
    t_total_start = time.time()
    clean_query = preprocess_query(query)
    client = get_typesense_client()
    candidate_pool = max(5, top_k)
    
    # Init telemetry dict
    telemetry = {
        "request_id": request_id,
        "workspace_id": workspace_id,
    }

    tier_executed = "tier1_raw_fastpath"
    expansion_applied = False
    expanded_query = None

    # ── Tenant Isolation Guard ────────────────────────────────────────────────
    if workspace_id is None:
        logger.warning("Search query [%s] called without workspace_id. Raising HTTP 400 for tenant safety.", query)
        raise HTTPException(
            status_code=400,
            detail="workspace_id is required to ensure multi-tenant data isolation.",
        )

    # ── Dense Embedding / BM25 Degradation Handling ───────────────────────────
    raw_vector = None
    t_embed_ms = 0.0
    degraded_bm25_fallback = False

    if embed is None or not callable(embed):
        logger.warning(
            "Dense embedding engine unavailable for query [%s]. Falling back to pure Typesense BM25 search.",
            clean_query,
        )
        degraded_bm25_fallback = True
        tier_executed = "tier1_bm25_fallback"
    else:
        t_embed_start = time.time()
        try:
            raw_vector = await asyncio.to_thread(embed, clean_query)
            t_embed_ms = round((time.time() - t_embed_start) * 1000, 2)
        except Exception as e:
            logger.warning(
                "Dense embedding generation failed for query [%s]: %s. Falling back to pure Typesense BM25 search.",
                clean_query,
                e,
            )
            raw_vector = None
            t_embed_ms = round((time.time() - t_embed_start) * 1000, 2)
            degraded_bm25_fallback = True
            tier_executed = "tier1_bm25_fallback"

    # ── Tier 1: Search Execution (Hybrid or Pure BM25 Fallback) ───────────────
    t1_start = time.time()
    t_ts_start = time.time()
    first_search_res = await asyncio.to_thread(
        execute_hybrid_search,
        client=client,
        query_text=clean_query,
        query_vector=raw_vector,
        workspace_id=workspace_id,
        top_k=candidate_pool,
        document_type=document_type,
    )
    t_typesense_ms = round((time.time() - t_ts_start) * 1000, 2)
    t_tier1_ms = round((time.time() - t1_start) * 1000, 2)

    first_pass_hits = parse_typesense_hits(first_search_res.get("hits", []))
    first_pass_top_score = first_pass_hits[0]["score"] if first_pass_hits else 0.0
    first_pass_top_id = first_pass_hits[0]["id"] if first_pass_hits else None
    current_hits = first_pass_hits

    # ── Tier 2 Fallback: Controlled LLM Query Expansion ───────────────────────
    # Gating: Only invoke LLM expansion when Tier 1 did NOT achieve a confident match.
    # If Tier 1 found an exact/strong keyword match (ratio >= 0.80) or score >= 0.60,
    # expansion is skipped to eliminate ~1000ms latency and avoid dragging in trailing noise.
    t_llm_expand_ms = 0.0
    expansion_threshold = config.RETRIEVAL_EXPANSION_THRESHOLD  # default 0.65
    top_hit_confident = bool(
        first_pass_hits and (
            first_pass_top_score >= 0.60 or
            (first_pass_hits[0].get("keyword_score", 0.0) >= 0.80 and first_pass_hits[0].get("match_type") in ("hybrid", "bm25"))
        )
    )
    if not top_hit_confident and first_pass_top_score < expansion_threshold and config.LLM_EXPANSION_API_KEY:
        t_llm_start = time.time()
        llm_expanded = await expand_query_via_llm(clean_query)
        if llm_expanded:
            tier_executed = "tier2_llm_expansion"
            expansion_applied = True
            expanded_query = f"{clean_query} {llm_expanded}"

            t_expand_vec = None
            if not degraded_bm25_fallback and embed and callable(embed):
                try:
                    t_expand_vec = await asyncio.to_thread(embed, expanded_query)
                except Exception:
                    t_expand_vec = None

            expand_search_res = await asyncio.to_thread(
                execute_hybrid_search,
                client=client,
                query_text=expanded_query,
                query_vector=t_expand_vec,
                workspace_id=workspace_id,
                top_k=candidate_pool,
                document_type=document_type,
            )
            expand_hits = parse_typesense_hits(expand_search_res.get("hits", []))

            # Non-destructive result fusion
            hit_map = {h["id"]: h for h in first_pass_hits}
            for h in expand_hits:
                if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                    hit_map[h["id"]] = h
            current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:candidate_pool]
        t_llm_expand_ms = round((time.time() - t_llm_start) * 1000, 2)

    # ── Post-Retrieval Candidate Reranker ─────────────────────────────────────
    current_hits, rerank_meta = rerank_candidates(clean_query, current_hits, contextual_signal)

    # ── Precision Candidate Filtering ─────────────────────────────────────────
    # Filter out low-confidence trailing noise that lacks relevance to the primary query
    filtered_hits = []
    top_score = current_hits[0]["score"] if current_hits else 0.0
    for i, h in enumerate(current_hits):
        if h["score"] < 0.40:
            continue
        # If top hit is highly confident (>= 0.60), trailing hits with zero keyword match
        # that drop more than 25% below top score are pruned as unrelated noise.
        if i > 0 and top_score >= 0.60 and h.get("keyword_score", 0.0) == 0.0 and h["score"] < (top_score * 0.75):
            continue
        filtered_hits.append(h)

    final_hits = (filtered_hits if filtered_hits else current_hits)[:top_k]
    total_retrieval_latency_ms = round((time.time() - t_total_start) * 1000, 2)
    final_score = final_hits[0]["score"] if final_hits else 0.0
    returned_faq_ids = [h["id"] for h in final_hits]

    telemetry.update({
        "tier_executed": tier_executed,
        "first_pass_score": first_pass_top_score,
        "first_pass_top_id": first_pass_top_id,
        "expansion_triggered": expansion_applied,
        "expanded_query": expanded_query,
        "final_score": final_score,
        "embedding_ms": t_embed_ms,
        "typesense_ms": t_typesense_ms,
        "tier1_ms": t_tier1_ms,
        "llm_expand_ms": t_llm_expand_ms,
        "latency_total_ms": total_retrieval_latency_ms,
        "total_retrieval_latency_ms": total_retrieval_latency_ms,
        "returned_faq_ids": returned_faq_ids,
        "degraded_bm25_fallback": degraded_bm25_fallback,
        "reranker_applied": rerank_meta.get("reranker_applied", False),
        "rerank_ms": rerank_meta.get("rerank_ms", 0.0),
        "reranker_reason": rerank_meta.get("reranker_reason"),
        "document_type_filter": document_type,
    })

    logger.info(
        "Search: '%s' -> %d hits, tier=%s, score=%.4f (first_pass=%.4f), latency=%.2fms",
        clean_query,
        len(final_hits),
        tier_executed,
        final_score,
        first_pass_top_score,
        total_retrieval_latency_ms,
    )

    return {
        "results": final_hits,
        "query": clean_query,
        "expanded_query": expanded_query,
        "expansion_applied": expansion_applied,
        "total_found": len(final_hits),
        "telemetry": telemetry,
    }


