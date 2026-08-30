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
    search_knowledge_base,
    preprocess_query,
    expand_locally,
    expand_query_via_llm,
    execute_hybrid_search,
    parse_typesense_hits,
    rerank_candidate_hits,
)

# ── Long-Tail Multi-Category Benchmark Dataset ─────────────────────────────
LONGTAIL_DATASET = [
    # ── Category 1: Direct / Exact Queries (Tier 1 Target) ─────────────────────
    {"query": "How do I create an account?", "target": "create an account", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},
    {"query": "How do I enable two-factor authentication?", "target": "two-factor", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},
    {"query": "How is my data encrypted?", "target": "encrypted", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},
    {"query": "Does the platform comply with GDPR?", "target": "gdpr", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},
    {"query": "How do I view my invoices?", "target": "invoices", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},
    {"query": "How do I connect Telegram?", "target": "telegram", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},
    {"query": "How do I set up my workspace?", "target": "workspace", "cat": "Cat 1: Exact High-Confidence", "is_ood": False},

    # ── Category 2: Domain Synonym / Known Paraphrases (Tier 2 Target) ─────────
    {"query": "notun akaunt kivabe khulbo?", "target": "create an account", "cat": "Cat 2: Domain Synonym", "is_ood": False},
    {"query": "card change korbo kivabe?", "target": "payment method", "cat": "Cat 2: Domain Synonym", "is_ood": False},
    {"query": "purono invoice kothay pabo?", "target": "invoices", "cat": "Cat 2: Domain Synonym", "is_ood": False},
    {"query": "duto channel eksathe connect kora jabe?", "target": "multiple channels", "cat": "Cat 2: Domain Synonym", "is_ood": False},
    {"query": "password vule gechi kivabe reset korbo?", "target": "create an account", "cat": "Cat 2: Domain Synonym", "is_ood": False},
    {"query": "2-step verification kivabe on korbo?", "target": "two-factor", "cat": "Cat 2: Domain Synonym", "is_ood": False},
    {"query": "plan upgrade korar upay ki?", "target": "change my plan", "cat": "Cat 2: Domain Synonym", "is_ood": False},

    # ── Category 3: Novel Unseen Terminology & Long-Tail (Tier 3 Target) ──────
    {"query": "Is there a way to inspect the monthly billing receipts sent to our accounting department?", "target": "invoices", "cat": "Cat 3: Unseen Long-Tail", "is_ood": False},
    {"query": "What should I do if the automated messaging agent is totally silent and unresponsive on Messenger?", "target": "not responding", "cat": "Cat 3: Unseen Long-Tail", "is_ood": False},
    {"query": "Can we link our official Meta communication channel and Telegram simultaneously?", "target": "multiple channels", "cat": "Cat 3: Unseen Long-Tail", "is_ood": False},
    {"query": "Is it possible to switch from monthly billing increments to an annual yearly settlement?", "target": "change my plan", "cat": "Cat 3: Unseen Long-Tail", "is_ood": False},
    {"query": "Where do I submit updated corporate credit card details for recurring subscription charges?", "target": "payment method", "cat": "Cat 3: Unseen Long-Tail", "is_ood": False},
    {"query": "Where can new trial users test platform features for a fortnight without upfront fees?", "target": "free trial", "cat": "Cat 3: Unseen Long-Tail", "is_ood": False},

    # ── Category 4: Completely Out-of-Domain (OOD Negative Safety) ────────────
    {"query": "What is the recipe for traditional chicken biryani with saffron?", "target": None, "cat": "Cat 4: Out-of-Domain", "is_ood": True},
    {"query": "Who won the 2022 FIFA World Cup final in Qatar?", "target": None, "cat": "Cat 4: Out-of-Domain", "is_ood": True},
    {"query": "Write a Python function to invert a binary search tree using recursion", "target": None, "cat": "Cat 4: Out-of-Domain", "is_ood": True},
    {"query": "What are the key differences between Renaissance and Baroque architecture?", "target": None, "cat": "Cat 4: Out-of-Domain", "is_ood": True},
]


async def run_evaluation():
    print("=================================================================================================")
    print("🔬 3-TIER ADAPTIVE RETRIEVAL & LONG-TAIL ESCAPE HATCH EVALUATION")
    print("=================================================================================================")

    load_model()
    
    tier_counts = {"tier1_raw_fastpath": 0, "tier2_local_deterministic": 0, "tier3_llm_escape_hatch": 0}
    category_results = {}
    all_latencies = []
    tier_latencies = {"tier1_raw_fastpath": [], "tier2_local_deterministic": [], "tier3_llm_escape_hatch": []}

    for item in LONGTAIL_DATASET:
        q = item["query"]
        target = item.get("target")
        cat = item["cat"]
        is_ood = item["is_ood"]

        if cat not in category_results:
            category_results[cat] = {
                "total": 0, "top1": 0, "top3": 0, "rr": [],
                "ood_correct": 0, "latencies": [],
                "tiers": {"tier1_raw_fastpath": 0, "tier2_local_deterministic": 0, "tier3_llm_escape_hatch": 0}
            }

        t_start = time.time()
        res = await search_knowledge_base(query=q, workspace_id=1, top_k=5)
        lat_ms = round((time.time() - t_start) * 1000, 2)

        telemetry = res["telemetry"]
        tier = telemetry["tier_executed"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        tier_latencies[tier].append(lat_ms)
        all_latencies.append(lat_ms)

        category_results[cat]["total"] += 1
        category_results[cat]["latencies"].append(lat_ms)
        category_results[cat]["tiers"][tier] += 1

        hits = res["results"]
        top_score = hits[0]["score"] if hits else 0.0
        hit_questions = [h["question"].lower() for h in hits]

        if is_ood:
            # Safe negative rejection if below answerability threshold 0.45
            if top_score < 0.45:
                category_results[cat]["ood_correct"] += 1
            print(f"  [{cat}] \"{q[:40]}...\" ➜ Tier: {tier} | Top Score: {top_score:.4f} | OOD Safe: {'✅' if top_score < 0.45 else '❌'}")
        else:
            t_lower = target.lower() if target else ""
            is_top1 = hit_questions and t_lower in hit_questions[0]
            is_top3 = any(t_lower in hq for hq in hit_questions[:3])
            
            rank_found = 0
            for rank, hq in enumerate(hit_questions[:5], 1):
                if t_lower in hq:
                    rank_found = rank
                    break
            rr = (1.0 / rank_found) if rank_found > 0 else 0.0
            category_results[cat]["rr"].append(rr)

            if is_top1:
                category_results[cat]["top1"] += 1
            if is_top3:
                category_results[cat]["top3"] += 1

            status_icon = "✅" if is_top1 else ("🟡" if is_top3 else "❌")
            print(f"  [{cat}] \"{q[:40]}...\" ➜ Tier: {tier} | Score: {top_score:.4f} | Latency: {lat_ms:5.1f}ms | Top Match: {hits[0]['question'] if hits else 'None'} {status_icon}")

    print("\n=================================================================================================")
    print("📊 1. TIER EXECUTION DISTRIBUTION & INVOCATION RATES")
    print("=================================================================================================")
    total_q = len(LONGTAIL_DATASET)
    for tier, count in tier_counts.items():
        pct = (count / total_q) * 100
        avg_t_lat = statistics.mean(tier_latencies[tier]) if tier_latencies[tier] else 0.0
        print(f"  • {tier:<30} : {count:2d} / {total_q} queries ({pct:5.1f}%) | Avg Latency: {avg_t_lat:6.1f} ms")

    print("\n=================================================================================================")
    print("📊 2. CATEGORY ACCURACY & RECALL SCORECARD")
    print("=================================================================================================")
    printf_format = "%-35s | %-6s | %-7s | %-7s | %-7s | %-9s\n"
    print(printf_format % ("Category", "Count", "Top-1", "Top-3", "MRR", "Avg Lat"))
    print("-" * 85)
    for cat, data in category_results.items():
        tot = data["total"]
        if "Out-of-Domain" in cat:
            ood_rate = (data["ood_correct"] / tot) * 100
            print(f"{cat:<35} | {tot:6d} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {statistics.mean(data['latencies']):>7.1f}ms (Safe Rejection: {ood_rate:.1f}%)")
        else:
            top1_pct = (data["top1"] / tot) * 100
            top3_pct = (data["top3"] / tot) * 100
            mrr_val = statistics.mean(data["rr"]) if data["rr"] else 0.0
            avg_l = statistics.mean(data["latencies"])
            print(printf_format % (cat, f"{tot}", f"{top1_pct:.1f}%", f"{top3_pct:.1f}%", f"{mrr_val:.4f}", f"{avg_l:.1f}ms"))

    print("\n=================================================================================================")
    print("📊 3. LATENCY PERCENTILE PROFILE (P50, P95, P99)")
    print("=================================================================================================")
    sorted_lats = sorted(all_latencies)
    p50 = statistics.median(sorted_lats)
    p95 = statistics.quantiles(sorted_lats, n=20)[18] if len(sorted_lats) >= 20 else max(sorted_lats)
    p99 = sorted_lats[-1]
    avg_total = statistics.mean(sorted_lats)

    print(f"  • P50 Latency (Median)  : {p50:6.1f} ms")
    print(f"  • P95 Latency           : {p95:6.1f} ms")
    print(f"  • P99 Latency (Tail)    : {p99:6.1f} ms")
    print(f"  • Average Latency       : {avg_total:6.1f} ms")
    print("=================================================================================================")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
