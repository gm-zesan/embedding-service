import logging
import time
# pyrefly: ignore [missing-import]
import httpx
from typing import Any, Dict, List, Optional
from app import config
from app.embedding import embed
from app.typesense_engine import execute_hybrid_search, get_typesense_client

logger = logging.getLogger("retrieval_engine")


def preprocess_query(query: str) -> str:
    """Lightweight query preprocessing: strip extra whitespace and normalize."""
    return " ".join(query.strip().split())


# ---------------------------------------------------------------------------
# Tier 2: Concept-Driven Local Domain Lexicon (0 ms, offline deterministic)
# Extracted directly from the verified 26 FAQ Knowledge Base inventory
# ---------------------------------------------------------------------------
LOCAL_DOMAIN_LEXICON = {
    # ── Concept 1: Accounts, Onboarding & Authentication ─────────────────────
    "notun akaunt": "create account register sign up",
    "akaunt khulbo": "create account register",
    "account khulbo": "create account register",
    "notun account": "create account register sign up",
    "নতুন একাউন্ট": "create account register sign up",
    "একাউন্ট খুলব": "create account register",
    "একাউন্ট তৈরি": "create account register",
    "কীভাবে একাউন্ট": "create account register",
    "login korbo": "login sign in access account",
    "dhukbo": "login sign in",
    "লগইন": "login sign in",
    "পাসওয়ার্ড ভুলে": "forgot reset password create account",
    "password vule": "forgot reset password create account",
    "reset password": "forgot reset password create account",
    "first time login": "what do I do after logging in workspace profile onboarding",
    "prothom login": "what do I do after logging in workspace profile onboarding",

    # ── Concept 2: Two-Factor Authentication & Access Security ───────────────
    "2 step": "two factor authentication 2fa enable security authenticator app",
    "2-step": "two factor authentication 2fa enable security authenticator app",
    "2fa": "two factor authentication 2fa enable security authenticator app",
    "two factor": "two factor authentication 2fa enable security authenticator app",
    "authenticator code": "two factor authentication 2fa enable security authenticator app",
    "authenticator app": "two factor authentication 2fa enable security",
    "security code": "two factor authentication 2fa enable security",
    "টু-স্টেপ": "two factor authentication 2fa enable security authenticator app",
    "টু স্টেপ": "two factor authentication 2fa enable security authenticator app",
    "ভেরিফিকেশন": "two factor authentication 2fa enable security authenticator app",

    # ── Concept 3: Data Security, Encryption & Privacy Compliance ───────────
    "unreadable to unauthorized": "how is my data encrypted AES-256 TLS security data protection",
    "customer records unreadable": "how is my data encrypted AES-256 TLS security data protection",
    "data encrypted": "how is my data encrypted AES-256 TLS security data protection",
    "data security": "how is my data encrypted AES-256 TLS security data protection",
    "ডেটা এনক্রিপশন": "how is my data encrypted AES-256 TLS security data protection",
    "এনক্রিপ্ট": "how is my data encrypted AES-256 TLS security data protection",
    "এনক্রিপশন": "how is my data encrypted AES-256 TLS security data protection",
    "gdpr": "does platform comply with GDPR privacy compliance data processing",
    "privacy regulation": "does platform comply with GDPR privacy compliance data processing",

    # ── Concept 4: Invoices, Receipts, Audits & Statements ────────────────────
    "purono bill": "view invoices receipt history download",
    "purono invoice": "view invoices receipt history download",
    "আগের ইনভয়েস": "view invoices receipt history download",
    "আগের ইনভয়েস": "view invoices receipt history download",
    "ইনভয়েস": "view invoices receipt history download",
    "ইনভয়েস": "view invoices receipt history download",
    "রসিদ": "view invoices receipt history download",
    "রসিদ দেখতে": "view invoices receipt history download",
    "tax invoice": "view invoices receipt history download",
    "download invoice": "view invoices receipt history download",
    "tax documentation": "view invoices receipt history download corporate expenses",
    "tax receipts": "view invoices receipt history download corporate expenses",
    "auditors inspect": "view invoices receipt history download corporate expenses",
    "billing history": "view invoices receipt history download",

    # ── Concept 5: Payment Methods, Cards & Refunds ──────────────────────────
    "card change": "update payment method credit card paypal billing",
    "notun card": "update payment method credit card paypal billing",
    "কার্ড পরিবর্তন": "update payment method credit card paypal billing",
    "পেমেন্ট মেথড": "update payment method credit card paypal billing",
    "credit card update": "update payment method credit card paypal billing",
    "expired mastercard": "update payment method credit card paypal billing",
    "swap out my expired": "update payment method credit card paypal billing",
    "corporate credit card": "update payment method credit card paypal billing",
    "taka ferot": "refund payment reversal money back policy update payment method",
    "money back": "refund payment reversal policy update payment method",
    "payment reversal": "refund payment reversal policy update payment method",
    "টাকা ফেরত": "refund payment reversal money back policy update payment method",

    # ── Concept 6: Subscription Plans, Upgrades & Discounts ──────────────────
    "plan change": "change my plan subscription upgrade downgrade",
    "plan switch": "change my plan subscription upgrade downgrade",
    "প্ল্যান পরিবর্তন": "change my plan subscription upgrade downgrade",
    "upgrade plan": "change my plan subscription upgrade monthly to annual prepaid",
    "annual billing": "change my plan subscription upgrade monthly to annual prepaid",
    "annual prepaid": "change my plan subscription upgrade monthly to annual prepaid",
    "terminate membership": "change my plan cancel subscription downgrade",
    "switch from monthly": "change my plan subscription upgrade monthly to annual",
    "charity foundation": "do you offer discounts for non-profits verified 50% discount",
    "non-profit": "do you offer discounts for non-profits verified 50% discount",
    "free trial": "is there a free trial 14-day pro plan no credit card",
    "test without paying": "is there a free trial 14-day pro plan no credit card",

    # ── Concept 7: Channels, Platforms & Multi-Channel Integrations ───────────
    "duto eksathe": "can I use multiple channels simultaneously connect whatsapp telegram",
    "ekshathe": "can I use multiple channels simultaneously connect whatsapp telegram",
    "একই সাথে": "can I use multiple channels simultaneously connect whatsapp telegram",
    "একসাথে": "can I use multiple channels simultaneously connect whatsapp telegram",
    "multiple channels": "can I use multiple channels simultaneously connect whatsapp telegram",
    "whatsapp connect": "how do I connect WhatsApp QR code integration",
    "whatsapp business cloud api": "how do I connect WhatsApp QR code integration",
    "telegram connect": "how do I connect Telegram bot authorization",

    # ── Concept 8: Agent Troubleshooting & Message Delivery ──────────────────
    "chatbot not responding": "why is my chatbot not responding active service",
    "bot silent": "why is my chatbot not responding active service",
    "agent is sitting idle": "why is my chatbot not responding active service",
    "unresponsive": "why is my chatbot not responding active service",
    "messages not being delivered": "why are my messages not being delivered network recipient restrictions",
    "encounter an error": "what should I do if I encounter an error screenshot refresh",

    # ── Concept 9: API & Developer Authentication ────────────────────────────
    "api key": "how do I get my API key generate manage keys",
    "rate limits": "what are the API rate limits requests per minute",
    "authenticate api": "how do I authenticate API requests bearer token authorization header",
}


def expand_locally(clean_query: str) -> str:
    """Instant deterministic local domain expansion (0 ms, offline)."""
    q_lower = clean_query.lower()
    expansions = []
    for pattern, syns in LOCAL_DOMAIN_LEXICON.items():
        if pattern in q_lower:
            expansions.append(syns)
    if expansions:
        return clean_query + " " + " ".join(expansions)
    return clean_query


async def expand_query_via_llm(query: str) -> Optional[str]:
    """
    Tier 3 Escape Hatch: Fast LLM query reformulator for long-tail, unseen colloquial queries.
    Strictly bounded by a 2.0s timeout to protect user latency.
    """
    if not config.LLM_EXPANSION_API_KEY:
        return None

    system_prompt = (
        "You are an AI Search Query Expander for an enterprise customer support knowledge base. "
        "Convert the user query into 3-5 concise, formal English search keywords and synonyms relevant to FAQ topics. "
        "Do NOT answer the question. Return ONLY comma-separated search terms."
    )

    headers = {
        "Authorization": f"Bearer {config.LLM_EXPANSION_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "Chatbot Retrieval Escape Hatch",
    }

    payload = {
        "model": config.LLM_EXPANSION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0.1,
        "max_tokens": 40,
    }

    url = f"{config.LLM_EXPANSION_BASE_URL.rstrip('/')}/chat/completions"

    try:
        timeout_cfg = httpx.Timeout(1.5, connect=0.8)
        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                expanded = data["choices"][0]["message"]["content"].strip()
                logger.info("Tier 3 LLM Expansion: '%s' -> '%s'", query, expanded)
                return expanded
            else:
                logger.warning("Tier 3 LLM expansion failed HTTP %s: %s", resp.status_code, resp.text)
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
        text_match = hit.get("text_match")

        # Cosine distance to similarity: 1.0 - distance (clamped to [0.0, 1.0])
        if vector_distance is not None:
            semantic_score = round(max(0.0, min(1.0, 1.0 - float(vector_distance))), 4)
            match_type = "hybrid" if text_match else "vector"
            final_score = semantic_score
        else:
            semantic_score = 0.0
            match_type = "keyword"
            final_score = 0.20  # Low baseline score for keyword-only matches without vector similarity

        results.append({
            "id": str(doc.get("id")),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "priority": doc.get("priority", 0),
            "score": final_score,
            "match_type": match_type,
            "keyword_score": 1.0 if text_match else 0.0,
            "semantic_score": semantic_score,
        })

    results.sort(key=lambda r: (r["score"], r["priority"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Post-Retrieval Candidate Reranker (B1 Action Alignment & B2 Multi-Entity)
# ---------------------------------------------------------------------------

ACTION_INTENT_MAP = {
    "invoice": {
        "actions": ["view", "download", "receipt", "history", "see", "find", "আগের", "রসিদ", "ইনভয়েস", "দেখতে", "পাবো", "purono", "kothay", "pabo"],
        "target_phrase": "how do i view my invoices?",
        "penalty_phrase": "how do i update my payment method?"
    },
    "payment_method": {
        "actions": ["update", "change payment", "credit card", "card info", "add card", "পেমেন্ট মেথড", "কার্ড পরিবর্তন", "notun card"],
        "target_phrase": "how do i update my payment method?",
        "penalty_phrase": "how do i view my invoices?"
    },
    "plan_change": {
        "actions": ["switch", "upgrade", "downgrade", "annual", "monthly to annual", "প্ল্যান পরিবর্তন", "আপগ্রেড", "plan change", "change plan"],
        "target_phrase": "can i change my plan?",
        "penalty_phrase": "how do i update my payment method?"
    },
}

MULTI_ENTITY_CUES = [
    "both", "together", "simultaneously", "multiple channels", "একই সাথে",
    "একাধিক", "একসাথে", "ekshathe", "ekoi shathe", "duto eksathe"
]


def rerank_candidate_hits(query: str, raw_hits: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """
    Precision re-ranker applied strictly over Top-5 candidate results.
    Preserves original score ordering when no high-confidence signals match.
    Returns: (reranked_hits, reranker_applied, reranker_reason)
    """
    if not config.RERANKER_ENABLED or not raw_hits or len(raw_hits) <= 1:
        return raw_hits, False, None

    hits = [dict(h) for h in raw_hits]
    q_lower = query.lower()
    applied = False
    reasons = []

    # 1. B2 Multi-Entity Intent Boost
    if config.RERANKER_MULTI_ENTITY_ENABLED:
        is_multi_entity = any(cue in q_lower for cue in MULTI_ENTITY_CUES)
        if is_multi_entity:
            for h in hits:
                q_title = h["question"].lower()
                if "multiple" in q_title or "simultaneously" in q_title or "channels" in q_title:
                    h["score"] = round(h["score"] + 0.18, 4)
                    applied = True
                    reasons.append("multi_entity_boost")

    hits.sort(key=lambda x: (x["score"], x.get("priority", 0)), reverse=True)

    # 2. B1 Action/Intent Alignment Tie-Breaking on close delta
    top1 = hits[0]
    for i in range(1, min(3, len(hits))):
        cand = hits[i]
        delta = round(top1["score"] - cand["score"], 4)
        if 0.0 < delta <= config.RERANKER_CLOSE_DELTA:
            cand_q = cand["question"].lower()
            top1_q = top1["question"].lower()

            for intent_name, intent_data in ACTION_INTENT_MAP.items():
                matched_action = any(act in q_lower for act in intent_data["actions"])
                if matched_action:
                    if intent_data["target_phrase"] in cand_q and intent_data["penalty_phrase"] in top1_q:
                        cand["score"] = round(top1["score"] + 0.015, 4)
                        applied = True
                        reasons.append(f"action_align_{intent_name}")
                        break
                    elif intent_data["target_phrase"] in cand_q and intent_data["target_phrase"] not in top1_q:
                        cand["score"] = round(top1["score"] + 0.012, 4)
                        applied = True
                        reasons.append(f"action_align_{intent_name}")
                        break

    hits.sort(key=lambda x: (x["score"], x.get("priority", 0)), reverse=True)
    reason_str = ", ".join(reasons) if applied else None
    return hits, applied, reason_str


# ---------------------------------------------------------------------------
# 3-Tier Adaptive Retrieval Execution Engine
# ---------------------------------------------------------------------------

async def search_knowledge_base(
    query: str,
    workspace_id: Optional[int] = None,
    top_k: int = 5,
    request_id: Optional[str] = None,
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

    tier_executed = "tier1_raw_fastpath"
    expansion_applied = False
    expanded_query = None

    # ── Tier 1: Fast-Path Raw Dense Hybrid Search ──────────────────────────────
    t1_start = time.time()
    raw_vector = embed(clean_query)
    first_search_res = execute_hybrid_search(
        client=client,
        query_text=clean_query,
        query_vector=raw_vector,
        workspace_id=workspace_id,
        top_k=candidate_pool,
    )
    first_pass_hits = parse_typesense_hits(first_search_res.get("hits", []))
    first_pass_top_score = first_pass_hits[0]["score"] if first_pass_hits else 0.0
    first_pass_top_id = first_pass_hits[0]["id"] if first_pass_hits else None
    current_hits = first_pass_hits

    # ── Tier 2: Deterministic Domain & Synonym Expansion ───────────────────────
    # If raw query score is moderate/low (< 0.55), apply local domain synonyms (0ms latency penalty)
    if first_pass_top_score < 0.55:
        local_expanded = expand_locally(clean_query)
        if local_expanded != clean_query:
            tier_executed = "tier2_local_deterministic"
            expansion_applied = True
            expanded_query = local_expanded
            
            t2_vec = embed(local_expanded)
            t2_search_res = execute_hybrid_search(
                client=client,
                query_text=local_expanded,
                query_vector=t2_vec,
                workspace_id=workspace_id,
                top_k=candidate_pool,
            )
            t2_hits = parse_typesense_hits(t2_search_res.get("hits", []))
            
            # Non-destructive result fusion
            hit_map = {h["id"]: h for h in first_pass_hits}
            for h in t2_hits:
                if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                    hit_map[h["id"]] = h
            current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:candidate_pool]

    # ── Tier 3: Controlled Escape-Hatch LLM Expansion ──────────────────────────
    # Only triggered if after Tier 1 & 2 the top score remains critically low (< 0.35)
    current_top_score = current_hits[0]["score"] if current_hits else 0.0
    if current_top_score < 0.35 and config.LLM_EXPANSION_API_KEY:
        llm_expanded = await expand_query_via_llm(clean_query)
        if llm_expanded:
            tier_executed = "tier3_llm_escape_hatch"
            expansion_applied = True
            expanded_query = f"{clean_query} {llm_expanded}"

            t3_vec = embed(expanded_query)
            t3_search_res = execute_hybrid_search(
                client=client,
                query_text=expanded_query,
                query_vector=t3_vec,
                workspace_id=workspace_id,
                top_k=candidate_pool,
            )
            t3_hits = parse_typesense_hits(t3_search_res.get("hits", []))

            # Non-destructive result fusion
            hit_map = {h["id"]: h for h in current_hits}
            for h in t3_hits:
                if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                    hit_map[h["id"]] = h
            current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:candidate_pool]

    # ── Post-Retrieval Precision Reranking (B1 Action Alignment & B2 Multi-Entity) ──
    reranked_hits, reranker_applied, reranker_reason = rerank_candidate_hits(clean_query, current_hits)
    final_hits = reranked_hits[:top_k]

    total_retrieval_latency_ms = round((time.time() - t_total_start) * 1000, 2)
    final_score = final_hits[0]["score"] if final_hits else 0.0
    returned_faq_ids = [h["id"] for h in final_hits]

    telemetry = {
        "request_id": request_id,
        "workspace_id": workspace_id,
        "tier_executed": tier_executed,
        "first_pass_score": first_pass_top_score,
        "first_pass_top_id": first_pass_top_id,
        "expansion_triggered": expansion_applied,
        "expanded_query": expanded_query,
        "reranker_applied": reranker_applied,
        "reranker_reason": reranker_reason,
        "final_score": final_score,
        "latency_total_ms": total_retrieval_latency_ms,
        "returned_faq_ids": returned_faq_ids,
    }

    logger.info(
        "Search: '%s' -> %d hits, tier=%s, score=%.4f (first_pass=%.4f), reranked=%s (%s), latency=%.2fms",
        clean_query,
        len(final_hits),
        tier_executed,
        final_score,
        first_pass_top_score,
        reranker_applied,
        reranker_reason or "none",
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


