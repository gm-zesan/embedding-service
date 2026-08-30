import asyncio
import io
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app import config
from app.typesense_engine import get_typesense_client
from app.embedding import embed
from app.retrieval_engine import execute_hybrid_search, parse_typesense_hits, rerank_candidate_hits, preprocess_query, expand_locally

from benchmark_100_queries import BENCHMARK_100_DATASET

# ---------------------------------------------------------------------------
# Local Fast Synonym & Normalization Dictionary (0ms latency, zero remote LLM)
# ---------------------------------------------------------------------------
LOCAL_SYNONYM_DICT = {
    # Account & Registration
    "notun akaunt": "create account register sign up",
    "akaunt khulbo": "create account register",
    "account khulbo": "create account register",
    "নতুন একাউন্ট": "create account register sign up",
    "একাউন্ট খুলব": "create account register",
    "একাউন্ট তৈরি": "create account register",
    # Login & Authentication
    "login korbo": "login sign in access account",
    "dhukbo": "login sign in",
    "লগইন": "login sign in",
    "পাসওয়ার্ড ভুলে": "forgot reset password",
    "password vule": "forgot reset password",
    # Invoices & Billing
    "purono bill": "view invoices receipt history",
    "purono invoice": "view invoices receipt history",
    "আগের ইনভয়েস": "view invoices receipt history",
    "রসিদ দেখতে": "view invoices receipt history",
    # Payment method
    "card change": "update payment method credit card",
    "notun card": "update payment method credit card",
    "কার্ড পরিবর্তন": "update payment method credit card",
    "পেমেন্ট মেথড": "update payment method credit card",
    # Subscription plan
    "plan change": "change subscription plan upgrade",
    "plan switch": "change subscription plan upgrade",
    "প্ল্যান পরিবর্তন": "change subscription plan upgrade",
    # Multi-channel
    "duto eksathe": "connect multiple channels simultaneously",
    "ekshathe": "multiple channels simultaneously",
    "একই সাথে": "multiple channels simultaneously",
    "একসাথে": "multiple channels simultaneously",
}

def expand_locally(clean_query: str) -> str:
    """Instant deterministic local synonym expansion (0 ms, offline)."""
    q_lower = clean_query.lower()
    expansions = []
    for pattern, syns in LOCAL_SYNONYM_DICT.items():
        if pattern in q_lower:
            expansions.append(syns)
    if expansions:
        return clean_query + " " + " ".join(expansions)
    return clean_query


async def search_mode(query: str, mode: str, client: Any, workspace_id: int = 1, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    """
    Execute retrieval in 3 distinct modes:
    - 'remote_llm': Baseline with remote LLM query expansion
    - 'pure_dense_hybrid': No expansion at all (Typesense Hybrid + Reranker only)
    - 'local_deterministic': Instant Local Synonym Normalizer (0 ms remote LLM) + Typesense + Reranker
    """
    t_start = time.time()
    clean_query = preprocess_query(query)
    candidate_pool = max(5, top_k)

    if mode == "local_deterministic":
        search_text = expand_locally(clean_query)
        vec = embed(search_text)
        search_res = execute_hybrid_search(
            client=client,
            query_text=search_text,
            query_vector=vec,
            workspace_id=workspace_id,
            top_k=candidate_pool,
        )
        hits = parse_typesense_hits(search_res.get("hits", []))
        reranked, _, _ = rerank_candidate_hits(query, hits)
        latency_ms = round((time.time() - t_start) * 1000, 2)
        return reranked[:top_k], latency_ms

    elif mode == "pure_dense_hybrid":
        vec = embed(clean_query)
        search_res = execute_hybrid_search(
            client=client,
            query_text=clean_query,
            query_vector=vec,
            workspace_id=workspace_id,
            top_k=candidate_pool,
        )
        hits = parse_typesense_hits(search_res.get("hits", []))
        reranked, _, _ = rerank_candidate_hits(query, hits)
        latency_ms = round((time.time() - t_start) * 1000, 2)
        return reranked[:top_k], latency_ms

    else:  # 'remote_llm'
        vec = embed(clean_query)
        search_res = execute_hybrid_search(
            client=client,
            query_text=clean_query,
            query_vector=vec,
            workspace_id=workspace_id,
            top_k=candidate_pool,
        )
        first_hits = parse_typesense_hits(search_res.get("hits", []))
        top_score = first_hits[0]["score"] if first_hits else 0.0

        if top_score < config.RETRIEVAL_EXPANSION_THRESHOLD and config.LLM_EXPANSION_API_KEY:
            expanded = await expand_query_via_llm(clean_query)
            if expanded:
                exp_vec = embed(expanded)
                second_res = execute_hybrid_search(
                    client=client,
                    query_text=f"{clean_query} {expanded}",
                    query_vector=exp_vec,
                    workspace_id=workspace_id,
                    top_k=candidate_pool,
                )
                second_hits = parse_typesense_hits(second_res.get("hits", []))
                hit_map = {h["id"]: h for h in first_hits}
                for h in second_hits:
                    if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                        hit_map[h["id"]] = h
                fused = list(hit_map.values())
                fused.sort(key=lambda x: (x["score"], x.get("priority", 0)), reverse=True)
                reranked, _, _ = rerank_candidate_hits(query, fused)
                latency_ms = round((time.time() - t_start) * 1000, 2)
                return reranked[:top_k], latency_ms

        reranked, _, _ = rerank_candidate_hits(query, first_hits)
        latency_ms = round((time.time() - t_start) * 1000, 2)
        return reranked[:top_k], latency_ms


async def evaluate_dataset_in_mode(mode: str, label: str) -> Dict[str, Any]:
    print(f"\nEvaluating Mode: {label} ({mode})...")
    client = get_typesense_client()
    
    top1_correct = 0
    top3_correct = 0
    rr_scores = []
    ood_true_negatives = 0
    ood_total = 0
    in_domain_total = 0
    latencies = []

    for item in BENCHMARK_100_DATASET:
        q = item["query"]
        expected_target = item.get("expected_target")
        is_ood = item.get("is_ood", False)

        hits, lat_ms = await search_mode(q, mode, client)
        latencies.append(lat_ms)

        if is_ood:
            ood_total += 1
            top_score = hits[0]["score"] if hits else 0.0
            if top_score < 0.45:
                ood_true_negatives += 1
        else:
            in_domain_total += 1
            hit_questions = [h["question"].lower() for h in hits]
            target_lower = expected_target.lower() if expected_target else ""

            # Check Top-1
            if hit_questions and target_lower in hit_questions[0]:
                top1_correct += 1
                rr_scores.append(1.0)
            else:
                # Check reciprocal rank in top 5
                rank_found = 0
                for rank, hq in enumerate(hit_questions[:5], 1):
                    if target_lower in hq:
                        rank_found = rank
                        break
                if rank_found > 0:
                    rr_scores.append(1.0 / rank_found)
                else:
                    rr_scores.append(0.0)

            # Check Top-3
            if any(target_lower in hq for hq in hit_questions[:3]):
                top3_correct += 1

    top1_acc = (top1_correct / in_domain_total) * 100 if in_domain_total else 0
    top3_acc = (top3_correct / in_domain_total) * 100 if in_domain_total else 0
    mrr = statistics.mean(rr_scores) if rr_scores else 0
    ood_safety = (ood_true_negatives / ood_total) * 100 if ood_total else 0
    p50_lat = statistics.median(latencies)
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    avg_lat = statistics.mean(latencies)

    return {
        "label": label,
        "mode": mode,
        "in_domain_total": in_domain_total,
        "top1_acc": round(top1_acc, 2),
        "top3_acc": round(top3_acc, 2),
        "mrr": round(mrr, 4),
        "ood_safety": round(ood_safety, 2),
        "avg_lat_ms": round(avg_lat, 2),
        "p50_lat_ms": round(p50_lat, 2),
        "p95_lat_ms": round(p95_lat, 2),
    }


async def main():
    print("=================================================================================================")
    print("🔬 110-QUERY RETRIEVAL DETERMINISTIC BENCHMARK")
    print("=================================================================================================")

    from app.embedding import load_model
    load_model()

    m1 = await evaluate_dataset_in_mode("pure_dense_hybrid", "Pure Hybrid (No Expansion)")
    m2 = await evaluate_dataset_in_mode("local_deterministic", "Local Fast Normalizer + Synonym Expansion (0ms)")

    modes = [m1, m2]

    print("\n=================================================================================================")
    print("📊 110-QUERY RETRIEVAL EXPANSION A/B BENCHMARK SCORECARD")
    print("=================================================================================================")
    print(f"{'Retrieval Architecture Mode':<48} | {'Top-1':<7} | {'Top-3':<7} | {'MRR':<7} | {'OOD Safe':<8} | {'Avg Lat':<9} | {'P95 Lat':<9}")
    print("-------------------------------------------------------------------------------------------------------------------------")
    for m in modes:
        print(f"{m['label']:<48} | {m['top1_acc']:>5.1f}% | {m['top3_acc']:>5.1f}% | {m['mrr']:>7.4f} | {m['ood_safety']:>6.1f}% | {m['avg_lat_ms']:>7.1f}ms | {m['p95_lat_ms']:>7.1f}ms")
    print("=================================================================================================")


if __name__ == "__main__":
    asyncio.run(main())
