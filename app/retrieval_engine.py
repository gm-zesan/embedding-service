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


async def expand_query_via_llm(query: str) -> Optional[str]:
    """
    Call a fast LLM to expand/reformulate an ambiguous or transliterated query into clear retrieval keywords.
    NOTE: The LLM NEVER generates answers; it only produces concise search keywords.
    """
    if not config.LLM_EXPANSION_API_KEY:
        return None

    system_prompt = (
        "You are an AI Search Query Expander for an e-commerce / customer support knowledge base. "
        "Your ONLY task is to convert the user query (which might be in Bengali, Banglish, or informal English) "
        "into 3-5 concise, formal English search keywords and synonyms relevant to company policies, FAQ, and account/order actions. "
        "Do NOT answer the question. Do NOT output explanations. Return ONLY comma-separated search terms."
    )

    headers = {
        "Authorization": f"Bearer {config.LLM_EXPANSION_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "Chatbot Retrieval Engine",
    }

    payload = {
        "model": config.LLM_EXPANSION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0.1,
        "max_tokens": 60,
    }

    url = f"{config.LLM_EXPANSION_BASE_URL.rstrip('/')}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                expanded = data["choices"][0]["message"]["content"].strip()
                logger.info("LLM Query Expansion: '%s' -> '%s'", query, expanded)
                return expanded
            else:
                logger.warning("LLM expansion failed HTTP %s: %s", resp.status_code, resp.text)
                return None
    except Exception as e:
        logger.warning("LLM query expansion exception: %s", e)
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
            final_score = 0.50  # Baseline score for keyword-only matches without embedding

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

    # Sort results by final score descending, breaking ties with priority
    results.sort(key=lambda r: (r["score"], r["priority"]), reverse=True)
    return results


async def search_knowledge_base(
    query: str,
    workspace_id: Optional[int] = None,
    top_k: int = 5,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end adaptive retrieval pipeline with full observability telemetry:
    1. Light Preprocessing
    2. Dense Vector Generation
    3. First Typesense Hybrid Search
    4. Threshold evaluation & conditional LLM query expansion
    5. Result Fusion & Regression prevention
    6. Structured Telemetry & Latency recording
    """
    t_total_start = time.time()
    clean_query = preprocess_query(query)
    client = get_typesense_client()

    # Step 1: Dense Vector Generation & First Hybrid Search
    t1_start = time.time()
    query_vector = embed(clean_query)
    search_res = execute_hybrid_search(
        client=client,
        query_text=clean_query,
        query_vector=query_vector,
        workspace_id=workspace_id,
        top_k=top_k,
    )
    first_pass_latency_ms = round((time.time() - t1_start) * 1000, 2)

    first_pass_hits = parse_typesense_hits(search_res.get("hits", []))
    first_pass_top_score = first_pass_hits[0]["score"] if first_pass_hits else 0.0
    first_pass_top_id = first_pass_hits[0]["id"] if first_pass_hits else None

    expanded_query = None
    expansion_applied = False
    expansion_latency_ms = 0.0
    second_pass_latency_ms = 0.0
    second_pass_top_score = None
    second_pass_top_id = None
    final_hits = first_pass_hits

    # Step 2: Adaptive LLM Expansion (if below threshold and key configured)
    if first_pass_top_score < config.RETRIEVAL_EXPANSION_THRESHOLD and config.LLM_EXPANSION_API_KEY:
        t_exp_start = time.time()
        expanded_query = await expand_query_via_llm(clean_query)
        expansion_latency_ms = round((time.time() - t_exp_start) * 1000, 2)

        if expanded_query:
            expansion_applied = True
            t2_start = time.time()
            expanded_vector = embed(expanded_query)

            second_search_res = execute_hybrid_search(
                client=client,
                query_text=f"{clean_query} {expanded_query}",
                query_vector=expanded_vector,
                workspace_id=workspace_id,
                top_k=top_k,
            )
            second_pass_latency_ms = round((time.time() - t2_start) * 1000, 2)

            second_pass_hits = parse_typesense_hits(second_search_res.get("hits", []))
            if second_pass_hits:
                second_pass_top_score = second_pass_hits[0]["score"]
                second_pass_top_id = second_pass_hits[0]["id"]

            # Step 3: Result Fusion (Merge & de-duplicate preserving highest score per FAQ ID)
            hit_map: Dict[str, Dict[str, Any]] = {}
            for h in first_pass_hits:
                hit_map[h["id"]] = h

            for h in second_pass_hits:
                doc_id = h["id"]
                if doc_id not in hit_map or h["score"] > hit_map[doc_id]["score"]:
                    hit_map[doc_id] = h

            final_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:top_k]

    total_retrieval_latency_ms = round((time.time() - t_total_start) * 1000, 2)
    final_score = final_hits[0]["score"] if final_hits else 0.0
    returned_faq_ids = [h["id"] for h in final_hits]

    telemetry = {
        "request_id": request_id,
        "workspace_id": workspace_id,
        "first_pass_score": first_pass_top_score,
        "first_pass_top_id": first_pass_top_id,
        "expansion_triggered": expansion_applied,
        "expanded_query": expanded_query,
        "second_pass_score": second_pass_top_score,
        "second_pass_top_id": second_pass_top_id,
        "final_score": final_score,
        "first_pass_latency_ms": first_pass_latency_ms,
        "expansion_latency_ms": expansion_latency_ms,
        "second_pass_latency_ms": second_pass_latency_ms,
        "total_retrieval_latency_ms": total_retrieval_latency_ms,
        "returned_faq_ids": returned_faq_ids,
    }

    logger.info(
        "[Retrieval Telemetry] query='%s' score=%.3f exp=%s total_ms=%.1f returned=%s",
        clean_query[:50],
        final_score,
        "Y" if expansion_applied else "N",
        total_retrieval_latency_ms,
        returned_faq_ids[:3],
    )

    return {
        "results": final_hits,
        "query": clean_query,
        "expanded_query": expanded_query,
        "expansion_applied": expansion_applied,
        "total_found": len(final_hits),
        "telemetry": telemetry,
    }

