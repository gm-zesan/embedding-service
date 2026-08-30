import asyncio
import io
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app import config
from app.typesense_engine import get_typesense_client
from app.embedding import load_model, embed
from app.retrieval_engine import (
    preprocess_query,
    expand_locally,
    expand_query_via_llm,
    execute_hybrid_search,
    parse_typesense_hits,
    rerank_candidate_hits,
)

# ── Genuine Long-Tail & Transliterated Support Query Evaluation Suite ────────
# Specifically chosen long-tail queries where direct vocabulary is novel / transliterated
GENUINE_LONGTAIL_DATASET = [
    # 1. Informal Bengali / Banglish query with zero direct keyword overlap
    {"query": "je payment ta vul kore gesilo oita ki abar back ashbe?", "target": "payment method", "label": "Payment Reversal Inquiry"},
    # 2. Transliterated forgot password (where dense alone has low score 0.33)
    {"query": "password vule gechi kivabe reset korbo?", "target": "create an account", "label": "Forgot Password / Reset"},
    # 3. Terminology shift: terminate membership vs cancel subscription
    {"query": "Can I terminate my membership and halt recurring billing?", "target": "change my plan", "label": "Membership Termination"},
    # 4. Long-tail question on accounting receipt audits
    {"query": "How can our financial auditors inspect previous billing transactions?", "target": "invoices", "label": "Auditor Invoice Inspection"},
    # 5. Colloquial channel coupling
    {"query": "ekshathe duti messaging app chalu rakha jabe kina?", "target": "multiple channels", "label": "Simultaneous Multi-channel"},
    # 6. Novel wording for unassisted 2-factor setup
    {"query": "kivabe account er security barate 2-step verification lagabo?", "target": "two-factor", "label": "Security 2-Step Activation"},
    # 7. Unseen refund policy query
    {"query": "Can a user request a refund if they made an accidental payment?", "target": "payment method", "label": "Accidental Payment Refund"},
    # 8. Unresponsive agent on third-party channel
    {"query": "Bot is completely silent and not delivering outbound replies on Messenger", "target": "not responding", "label": "Bot Unresponsive on Messenger"},
]


async def execute_retrieval_in_mode(query: str, mode: str, client: Any, workspace_id: int = 1, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float, int, str]:
    """
    Execute retrieval in 3 isolated modes:
    - 'mode_a': Tier 1 Only (Raw Dense Vector + Typesense Hybrid)
    - 'mode_b': Tier 1 + Tier 2 (Raw + Local Deterministic Domain Expansion)
    - 'mode_c': Tier 1 + Tier 2 + Tier 3 (Raw + Local + Remote LLM Escape Hatch if score < 0.35)
    Returns: (final_hits, latency_ms, llm_calls_made, tier_executed)
    """
    t_start = time.time()
    clean_query = preprocess_query(query)
    candidate_pool = max(5, top_k)
    llm_calls = 0
    tier_executed = "tier1_raw"

    # ── Tier 1: Raw Dense Vector + Hybrid Search ──────────────────────────────
    raw_vec = embed(clean_query)
    s1 = execute_hybrid_search(client, clean_query, raw_vec, workspace_id, candidate_pool)
    hits1 = parse_typesense_hits(s1.get("hits", []))
    current_hits = hits1
    top1_score = hits1[0]["score"] if hits1 else 0.0

    if mode in ("mode_b", "mode_c"):
        # ── Tier 2: Local Domain Expansion ────────────────────────────────────
        if top1_score < 0.55:
            local_exp = expand_locally(clean_query)
            if local_exp != clean_query:
                tier_executed = "tier2_local"
                t2_vec = embed(local_exp)
                s2 = execute_hybrid_search(client, local_exp, t2_vec, workspace_id, candidate_pool)
                hits2 = parse_typesense_hits(s2.get("hits", []))
                # Non-destructive fusion
                hit_map = {h["id"]: h for h in current_hits}
                for h in hits2:
                    if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                        hit_map[h["id"]] = h
                current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:candidate_pool]
                top1_score = current_hits[0]["score"] if current_hits else 0.0

    if mode == "mode_c":
        # ── Tier 3: LLM Escape Hatch ──────────────────────────────────────────
        if top1_score < 0.35 and config.LLM_EXPANSION_API_KEY:
            llm_calls += 1
            tier_executed = "tier3_llm"
            llm_exp = await expand_query_via_llm(clean_query)
            if llm_exp:
                expanded_q = f"{clean_query} {llm_exp}"
                t3_vec = embed(expanded_q)
                s3 = execute_hybrid_search(client, expanded_q, t3_vec, workspace_id, candidate_pool)
                hits3 = parse_typesense_hits(s3.get("hits", []))
                # Non-destructive fusion
                hit_map = {h["id"]: h for h in current_hits}
                for h in hits3:
                    if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                        hit_map[h["id"]] = h
                current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:candidate_pool]

    # Re-ranker
    reranked, _, _ = rerank_candidate_hits(clean_query, current_hits)
    final_hits = reranked[:top_k]
    lat_ms = round((time.time() - t_start) * 1000, 2)

    return final_hits, lat_ms, llm_calls, tier_executed


async def run_mode_eval(mode_key: str, mode_label: str, client: Any) -> Dict[str, Any]:
    print(f"\n─────────────────────────────────────────────────────────────────────────────────────────────────")
    print(f"▶ Evaluating Mode: {mode_label}")
    print(f"─────────────────────────────────────────────────────────────────────────────────────────────────")

    total_q = len(GENUINE_LONGTAIL_DATASET)
    top1_count = 0
    top3_count = 0
    rr_list = []
    latencies = []
    total_llm_calls = 0

    for item in GENUINE_LONGTAIL_DATASET:
        q = item["query"]
        target = item["target"]
        label = item["label"]

        hits, lat_ms, llm_calls, tier = await execute_retrieval_in_mode(q, mode_key, client)
        latencies.append(lat_ms)
        total_llm_calls += llm_calls

        hit_questions = [h["question"].lower() for h in hits]
        t_lower = target.lower()
        is_top1 = hit_questions and t_lower in hit_questions[0]
        is_top3 = any(t_lower in hq for hq in hit_questions[:3])

        rank_found = 0
        for rank, hq in enumerate(hit_questions[:5], 1):
            if t_lower in hq:
                rank_found = rank
                break
        rr = (1.0 / rank_found) if rank_found > 0 else 0.0
        rr_list.append(rr)

        if is_top1:
            top1_count += 1
        if is_top3:
            top3_count += 1

        top_hit_name = hits[0]["question"] if hits else "None"
        top_score = hits[0]["score"] if hits else 0.0
        status_icon = "✅" if is_top1 else ("🟡" if is_top3 else "❌")
        print(f"  • {label:<32} ➜ Tier: {tier:<11} | Score: {top_score:.4f} | Latency: {lat_ms:6.1f}ms | Match: {top_hit_name} {status_icon}")

    sorted_lats = sorted(latencies)
    p50 = statistics.median(sorted_lats)
    p95 = statistics.quantiles(sorted_lats, n=20)[18] if len(sorted_lats) >= 20 else max(sorted_lats)
    p99 = sorted_lats[-1]
    avg_lat = statistics.mean(sorted_lats)
    top1_pct = (top1_count / total_q) * 100
    top3_pct = (top3_count / total_q) * 100
    mrr_val = statistics.mean(rr_list) if rr_list else 0.0

    return {
        "label": mode_label,
        "top1": top1_pct,
        "top3": top3_pct,
        "mrr": mrr_val,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "avg": avg_lat,
        "llm_calls": total_llm_calls,
    }


async def main():
    print("=================================================================================================")
    print("🔬 CONTROLLED A/B/C COMPARATIVE RETRIEVAL BENCHMARK ON GENUINE LONG-TAIL QUERIES")
    print("=================================================================================================")

    load_model()
    client = get_typesense_client()

    mA = await run_mode_eval("mode_a", "Mode A: Tier 1 Only (Pure Dense Hybrid)", client)
    mB = await run_mode_eval("mode_b", "Mode B: Tier 1 + Tier 2 (Local Domain Expansion)", client)
    mC = await run_mode_eval("mode_c", "Mode C: Tier 1 + Tier 2 + Tier 3 (LLM Escape Hatch)", client)

    modes = [mA, mB, mC]

    print("\n=================================================================================================")
    print("📊 COMPARATIVE A/B/C RETRIEVAL SCORECARD & LATENCY BREAKDOWN")
    print("=================================================================================================")
    print(f"{'Retrieval Architecture Mode':<48} | {'Top-1':<7} | {'Top-3':<7} | {'MRR':<7} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'LLM Calls':<9}")
    print("-" * 115)
    for m in modes:
        print(f"{m['label']:<48} | {m['top1']:>5.1f}% | {m['top3']:>5.1f}% | {m['mrr']:>7.4f} | {m['p50']:>7.1f}ms | {m['p95']:>7.1f}ms | {m['llm_calls']:>9d}")
    print("=================================================================================================")


if __name__ == "__main__":
    asyncio.run(main())
