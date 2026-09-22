import asyncio
import logging
import time
# pyrefly: ignore [missing-import]
import httpx
from typing import Any, Dict, List, Optional
from app import config
from app.embedding import embed
from app.llm import LLMRequest, default_client as llm_client
from app.typesense_engine import execute_hybrid_search, get_typesense_client

logger = logging.getLogger("retrieval_engine")


def preprocess_query(query: str) -> str:
    """Lightweight query preprocessing: strip extra whitespace and normalize."""
    return " ".join(query.strip().split())


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
        "You are an AI Search Query Expander for an enterprise multi-source customer support knowledge base. "
        "Analyze the user's query and generate 3-5 concise, formal search keywords and synonyms in both Bengali and English "
        "relevant to FAQ topics (e.g. refund, cancellation, delivery charges, payment methods, warranty). "
        "Do NOT answer the question. Return ONLY comma-separated search terms."
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
            "document_type": doc.get("document_type", "faq"),
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




# ---------------------------------------------------------------------------
# 2-Tier Adaptive Retrieval Execution Engine (MPNet + Fallback LLM Expander)
# ---------------------------------------------------------------------------

async def search_knowledge_base(
    query: str,
    workspace_id: Optional[int] = None,
    top_k: int = 5,
    request_id: Optional[str] = None,
    contextual_signal: Optional[str] = None,
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

    # ── Tier 1: Fast-Path Raw Dense Hybrid Search ──────────────────────────────
    t1_start = time.time()
    t_embed_start = time.time()
    raw_vector = await asyncio.to_thread(embed, clean_query)
    t_embed_ms = round((time.time() - t_embed_start) * 1000, 2)

    t_ts_start = time.time()
    first_search_res = await asyncio.to_thread(
        execute_hybrid_search,
        client=client,
        query_text=clean_query,
        query_vector=raw_vector,
        workspace_id=workspace_id,
        top_k=candidate_pool,
    )
    t_typesense_ms = round((time.time() - t_ts_start) * 1000, 2)
    t_tier1_ms = round((time.time() - t1_start) * 1000, 2)

    first_pass_hits = parse_typesense_hits(first_search_res.get("hits", []))
    first_pass_top_score = first_pass_hits[0]["score"] if first_pass_hits else 0.0
    first_pass_top_id = first_pass_hits[0]["id"] if first_pass_hits else None
    current_hits = first_pass_hits

    # ── Tier 2 Fallback: Controlled LLM Query Expansion ───────────────────────
    # Triggered dynamically when the raw MPNet vector score is low/unconfident (< RETRIEVAL_EXPANSION_THRESHOLD)
    t_llm_expand_ms = 0.0
    expansion_threshold = config.RETRIEVAL_EXPANSION_THRESHOLD  # default 0.55 - 0.65
    if first_pass_top_score < expansion_threshold and config.LLM_EXPANSION_API_KEY:
        t_llm_start = time.time()
        llm_expanded = await expand_query_via_llm(clean_query)
        if llm_expanded:
            tier_executed = "tier2_llm_expansion"
            expansion_applied = True
            expanded_query = f"{clean_query} {llm_expanded}"

            t_expand_vec = await asyncio.to_thread(embed, expanded_query)
            expand_search_res = await asyncio.to_thread(
                execute_hybrid_search,
                client=client,
                query_text=expanded_query,
                query_vector=t_expand_vec,
                workspace_id=workspace_id,
                top_k=candidate_pool,
            )
            expand_hits = parse_typesense_hits(expand_search_res.get("hits", []))

            # Non-destructive result fusion
            hit_map = {h["id"]: h for h in first_pass_hits}
            for h in expand_hits:
                if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                    hit_map[h["id"]] = h
            current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:candidate_pool]
        t_llm_expand_ms = round((time.time() - t_llm_start) * 1000, 2)

    final_hits = current_hits[:top_k]
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


