import logging
import time
# pyrefly: ignore [missing-import]
import httpx
from typing import Any, Dict, List, Optional
from app import config
from app.embedding import embed
from app.linguistics import CanonicalConceptMapper, LanguageProfiler, LinguisticEnrichment
from app.llm import LLMRequest, default_client as llm_client
from app.typesense_engine import execute_hybrid_search, get_typesense_client

logger = logging.getLogger("retrieval_engine")


def preprocess_query(query: str) -> str:
    """Lightweight query preprocessing: strip extra whitespace and normalize."""
    return " ".join(query.strip().split())


# ---------------------------------------------------------------------------
# Tier 2: Concept-Driven Local Domain Lexicon (0 ms, offline deterministic)
# Extracted directly from the verified 26 FAQ Knowledge Base inventory
# ---------------------------------------------------------------------------


def _compute_expansion(
    clean_query: str,
    lexicon_dict: Dict[str, str],
    normalized_signal: Optional[str] = None,
    contextual_signal: Optional[str] = None,
) -> str:
    """Core domain expansion logic over an arbitrary lexicon dictionary."""
    q_lower = clean_query.lower()
    norm_lower = normalized_signal.lower() if normalized_signal else ""
    ctx_lower = contextual_signal.lower() if contextual_signal else ""
    expansions = []
    
    # Process lexicon
    for pattern, syn_str in lexicon_dict.items():
        if pattern in q_lower or (norm_lower and pattern in norm_lower) or (ctx_lower and pattern in ctx_lower):
            expansions.append(syn_str)
            
    if expansions:
        return clean_query + " " + " ".join(expansions)
    return clean_query


def expand_locally(
    clean_query: str,
    normalized_signal: Optional[str] = None,
    contextual_signal: Optional[str] = None,
    snapshot: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> str:
    """Instant deterministic local domain expansion (0 ms, offline).
    Authoritative mode: Evaluates DB snapshot in parallel. Falls back to hardcoded config if missing."""
    
    # Convert snapshot format to flat dict
    db_lexicon_dict = {}
    if snapshot and "domain_entries" in snapshot:
        for concept_key, entries in snapshot["domain_entries"].items():
            for entry in entries:
                pattern = entry.get("pattern", "").lower()
                expansion = entry.get("expansion", "")
                if pattern and expansion:
                    if pattern in db_lexicon_dict:
                        db_lexicon_dict[pattern] += " " + expansion
                    else:
                        db_lexicon_dict[pattern] = expansion
                        
    authoritative_expansion = _compute_expansion(
        clean_query, 
        db_lexicon_dict, 
        normalized_signal, 
        contextual_signal
    )
            
    return authoritative_expansion


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

    req = LLMRequest(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=40,
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




def _rerank_internal(
    query: str,
    raw_hits: List[Dict[str, Any]],
    enrichment: Optional[LinguisticEnrichment],
    action_intent_map: Dict[str, Any],
    multi_entity_cues: List[str],
    policy_intent_map: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Core reranking logic over arbitrary mappings."""
    if not config.RERANKER_ENABLED or not raw_hits or len(raw_hits) <= 1:
        return raw_hits, False, None

    hits = [dict(h) for h in raw_hits]
    q_lower = query.lower()
    applied = False
    reasons = []

    # 1. B2 Multi-Entity Intent Boost
    if config.RERANKER_MULTI_ENTITY_ENABLED:
        is_multi_entity = any(cue in q_lower for cue in multi_entity_cues)
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

            for intent_name, intent_data in action_intent_map.items():
                matched_action = any(act in q_lower for act in intent_data.get("actions", intent_data.get("action_phrases", intent_data.get("action_keywords", []))))
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

    # 2.5 Scoped Canonical Concept Alignment (Phase 2B Additive Signal)
    if enrichment and enrichment.canonical_concepts:
        top1 = hits[0]
        # target_doc_types are already collected in metadata
        target_doc_types = enrichment.metadata.get("target_doc_types", [])
        for target_doc in target_doc_types:
            for i in range(1, min(3, len(hits))):
                cand = hits[i]
                delta = round(top1["score"] - cand["score"], 4)
                if 0.0 < delta <= 0.15:
                    cand_type = cand.get("document_type", "faq")
                    top1_type = top1.get("document_type", "faq")
                    if cand_type == target_doc and top1_type != target_doc:
                        cand["score"] = round(top1["score"] + 0.02, 4)
                        applied = True
                        reasons.append(f"concept_align_{target_doc}")
                        hits.sort(key=lambda x: (x["score"], x.get("priority", 0)), reverse=True)
                        top1 = hits[0]
                        break

    # 3. Direct Commerce Policy Intent Alignment on close delta
    top1 = hits[0]
    for i in range(1, min(3, len(hits))):
        cand = hits[i]
        delta = round(top1["score"] - cand["score"], 4)
        if 0.0 < delta <= 0.15:
            cand_type = cand.get("document_type", "faq")
            top1_type = top1.get("document_type", "faq")
            if cand_type != top1_type:
                for pol_name, pol_data in policy_intent_map.items():
                    if any(cue in q_lower for cue in pol_data.get("cues", pol_data.get("action_phrases", pol_data.get("cue_phrases", [])))):
                        if cand_type in pol_data.get("target_doc_types", []) and top1_type not in pol_data.get("target_doc_types", []):
                            cand["score"] = round(top1["score"] + 0.02, 4)
                            applied = True
                            reasons.append(f"policy_align_{pol_name}")
                            break

    hits.sort(key=lambda x: (x["score"], x.get("priority", 0)), reverse=True)
    reason_str = ", ".join(reasons) if applied else None
    return hits, applied, reason_str


def rerank_candidate_hits(
    query: str,
    raw_hits: List[Dict[str, Any]],
    enrichment: Optional[LinguisticEnrichment] = None,
    snapshot: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """
    Precision re-ranker applied strictly over Top-5 candidate results.
    Authoritative mode evaluates DB-backed snapshot. Falls back to hardcoded config.
    """

    action_mappings = snapshot.get("action_mappings", {}) if snapshot else {}
    policy_mappings = snapshot.get("policy_mappings", {}) if snapshot else {}

    db_multi_entity_cues = []
    if snapshot and "concept_patterns" in snapshot and "MULTI_ENTITY_DETECTION" in snapshot["concept_patterns"]:
        db_multi_entity_cues = snapshot["concept_patterns"]["MULTI_ENTITY_DETECTION"].get("positive_phrases", [])

    hits, applied, reason = _rerank_internal(
        query, raw_hits, enrichment, action_mappings, db_multi_entity_cues, policy_mappings
    )

    return hits, applied, reason



# ---------------------------------------------------------------------------
# 3-Tier Adaptive Retrieval Execution Engine
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
        "shadow_match_concept": None,
        "shadow_match_expansion": None,
        "shadow_match_rerank": None,
    }

    # Fetch DB snapshot for Shadow Mode
    from app.lexicon_repository import repository
    snapshot = await repository.get_or_fetch_snapshot(workspace_id)

    # Phase 2A & 2B: Language Profiling & Canonical Concept Mapping
    lang_profile = LanguageProfiler.profile(clean_query)
    enrichment = CanonicalConceptMapper.map_concepts(clean_query, lang_profile, snapshot, telemetry)

    # Phase 2E: If auxiliary contextual_signal provided, enrich canonical concepts
    # User original query remains 100% immutable!
    if contextual_signal:
        ctx_profile = LanguageProfiler.profile(contextual_signal)
        ctx_enrichment = CanonicalConceptMapper.map_concepts(contextual_signal, ctx_profile, snapshot)
        if ctx_enrichment.canonical_concepts:
            for c in ctx_enrichment.canonical_concepts:
                if c not in enrichment.canonical_concepts:
                    enrichment.canonical_concepts.append(c)
            curr_target_types = set(enrichment.metadata.get("target_doc_types", []))
            for dt in ctx_enrichment.metadata.get("target_doc_types", []):
                curr_target_types.add(dt)
            enrichment.metadata["target_doc_types"] = list(curr_target_types)

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
        norm_sig = enrichment.metadata.get("normalized_signal")
        local_expanded = expand_locally(clean_query, normalized_signal=norm_sig, contextual_signal=contextual_signal, snapshot=snapshot, telemetry=telemetry)
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
    reranked_hits, reranker_applied, reranker_reason = rerank_candidate_hits(clean_query, current_hits, enrichment=enrichment, snapshot=snapshot, telemetry=telemetry)
    final_hits = reranked_hits[:top_k]

    total_retrieval_latency_ms = round((time.time() - t_total_start) * 1000, 2)
    final_score = final_hits[0]["score"] if final_hits else 0.0
    returned_faq_ids = [h["id"] for h in final_hits]

    telemetry.update({
        "language_profile": lang_profile.language.value,
        "script_profile": lang_profile.script.value,
        "canonical_concepts": enrichment.canonical_concepts,
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
    })

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


