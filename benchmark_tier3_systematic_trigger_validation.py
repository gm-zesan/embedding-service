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

# ── 4-Dataset Systematic Validation Testbed ─────────────────────────────────
SYSTEMATIC_DATASET = [
    # ── Dataset A: Known / Normal Queries (Tier 1 Target) ─────────────────────
    {"query": "How do I create an account?", "target": "create an account", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "How do I view my invoices?", "target": "invoices", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "How do I enable two-factor authentication?", "target": "two-factor", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "How is my data encrypted?", "target": "encrypted", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "Does the platform comply with GDPR?", "target": "gdpr", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "How do I connect WhatsApp?", "target": "whatsapp", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "How do I connect Telegram?", "target": "telegram", "dataset": "Dataset A: Known/Normal", "is_ood": False},
    {"query": "Is there a free trial available?", "target": "free trial", "dataset": "Dataset A: Known/Normal", "is_ood": False},

    # ── Dataset B: Known-Domain Long-Tail Paraphrases (Tier 2 Target) ─────────
    {"query": "notun akaunt kivabe khulbo?", "target": "create an account", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "purono invoice kothay pabo?", "target": "invoices", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "card change korbo kivabe?", "target": "payment method", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "duto channel eksathe connect kora jabe?", "target": "multiple channels", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "2-step verification kivabe on korbo?", "target": "two-factor", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "plan upgrade korar upay ki?", "target": "change my plan", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "password vule gechi kivabe reset korbo?", "target": "create an account", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},
    {"query": "taka ferot pabo kivabe?", "target": "payment method", "dataset": "Dataset B: Known Long-Tail", "is_ood": False},

    # ── Dataset C: Truly Novel / Unseen Terminology (Tier 3 Target) ───────────
    {"query": "Can our accounting auditors inspect the official tax receipts from last quarter?", "target": "invoices", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "I need to swap out my expired Mastercard on file before the next billing cycle", "target": "payment method", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "How do we integrate our company customer helpdesk with Meta WhatsApp Business Cloud API?", "target": "whatsapp", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "Is there any safety feature requiring an authenticator code on my smartphone when logging in?", "target": "two-factor", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "Our automated agent is sitting idle and not answering any visitor questions on our webpage", "target": "not responding", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "We are an accredited charity foundation looking for reduced subscription pricing tiers", "target": "non-profit", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "How can I transition my monthly subscription tier into an annual prepaid contract?", "target": "change my plan", "dataset": "Dataset C: Truly Novel", "is_ood": False},
    {"query": "How do you ensure our customer records remain unreadable to unauthorized third parties?", "target": "encrypted", "dataset": "Dataset C: Truly Novel", "is_ood": False},

    # ── Dataset D: Out-of-Domain Queries (OOD Negative Safety) ────────────────
    {"query": "What is the recipe for traditional chicken biryani with saffron?", "target": None, "dataset": "Dataset D: Out-of-Domain", "is_ood": True},
    {"query": "Who won the 2022 FIFA World Cup final in Qatar?", "target": None, "dataset": "Dataset D: Out-of-Domain", "is_ood": True},
    {"query": "Write a Python function to invert a binary search tree using recursion", "target": None, "dataset": "Dataset D: Out-of-Domain", "is_ood": True},
    {"query": "What are the key differences between Renaissance and Baroque architecture?", "target": None, "dataset": "Dataset D: Out-of-Domain", "is_ood": True},
    {"query": "Can you recommend a 5-day travel itinerary for visiting Tokyo and Kyoto?", "target": None, "dataset": "Dataset D: Out-of-Domain", "is_ood": True},
    {"query": "How does photosynthesis convert sunlight into glucose in plant cells?", "target": None, "dataset": "Dataset D: Out-of-Domain", "is_ood": True},
]


async def run_systematic_validation():
    print("=================================================================================================")
    print("🔬 SYSTEMATIC 4-DATASET TIER 3 TRIGGER POLICY & OBSERVABILITY BENCHMARK")
    print("=================================================================================================")

    load_model()
    client = get_typesense_client()

    dataset_stats = {}
    tier_counts = {"tier1_raw_fastpath": 0, "tier2_local_deterministic": 0, "tier3_llm_escape_hatch": 0}
    tier_latencies = {"tier1_raw_fastpath": [], "tier2_local_deterministic": [], "tier3_llm_escape_hatch": []}

    t1_t2_failed_count = 0
    t3_recovered_count = 0
    t3_triggered_total = 0
    t3_false_recoveries = 0
    t3_ood_escapes = 0

    for item in SYSTEMATIC_DATASET:
        q = item["query"]
        target = item["target"]
        ds_name = item["dataset"]
        is_ood = item["is_ood"]

        if ds_name not in dataset_stats:
            dataset_stats[ds_name] = {
                "total": 0, "top1": 0, "top3": 0, "rr": [],
                "ood_safe": 0, "latencies": [],
                "tiers": {"tier1_raw_fastpath": 0, "tier2_local_deterministic": 0, "tier3_llm_escape_hatch": 0}
            }

        clean_q = preprocess_query(q)
        t_start = time.time()

        # Step 1: Tier 1 Raw
        raw_vec = embed(clean_q)
        s1 = execute_hybrid_search(client, clean_q, raw_vec, workspace_id=1, top_k=5)
        h1 = parse_typesense_hits(s1.get("hits", []))
        t1_top_score = h1[0]["score"] if h1 else 0.0
        current_hits = h1
        tier_exec = "tier1_raw_fastpath"

        # Step 2: Tier 2 Local
        t1_t2_passed = False
        if not is_ood:
            t_lower = target.lower() if target else ""
            t1_t2_passed = h1 and t_lower in h1[0]["question"].lower() and t1_top_score >= 0.45

        if t1_top_score < 0.55:
            local_q = expand_locally(clean_q)
            if local_q != clean_q:
                tier_exec = "tier2_local_deterministic"
                t2_vec = embed(local_q)
                s2 = execute_hybrid_search(client, local_q, t2_vec, workspace_id=1, top_k=5)
                h2 = parse_typesense_hits(s2.get("hits", []))
                hit_map = {h["id"]: h for h in h1}
                for h in h2:
                    if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                        hit_map[h["id"]] = h
                current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:5]
                t2_top_score = current_hits[0]["score"] if current_hits else 0.0
                if not is_ood:
                    t_lower = target.lower() if target else ""
                    t1_t2_passed = current_hits and t_lower in current_hits[0]["question"].lower() and t2_top_score >= 0.45

        # Step 3: Tier 3 Escape Hatch
        current_score = current_hits[0]["score"] if current_hits else 0.0
        t3_triggered = False
        if current_score < 0.35 and config.LLM_EXPANSION_API_KEY:
            t3_triggered = True
            t3_triggered_total += 1
            tier_exec = "tier3_llm_escape_hatch"
            llm_q = await expand_query_via_llm(clean_q)
            if llm_q:
                exp_q = f"{clean_q} {llm_q}"
                t3_vec = embed(exp_q)
                s3 = execute_hybrid_search(client, exp_q, t3_vec, workspace_id=1, top_k=5)
                h3 = parse_typesense_hits(s3.get("hits", []))
                hit_map = {h["id"]: h for h in current_hits}
                for h in h3:
                    if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                        hit_map[h["id"]] = h
                current_hits = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:5]

        # Rerank
        reranked, _, _ = rerank_candidate_hits(clean_q, current_hits)
        final_hits = reranked[:5]
        lat_ms = round((time.time() - t_start) * 1000, 2)

        tier_counts[tier_exec] += 1
        tier_latencies[tier_exec].append(lat_ms)
        dataset_stats[ds_name]["total"] += 1
        dataset_stats[ds_name]["latencies"].append(lat_ms)
        dataset_stats[ds_name]["tiers"][tier_exec] += 1

        final_top = final_hits[0] if final_hits else None
        final_score = final_top["score"] if final_top else 0.0
        final_match = final_top["question"] if final_top else "None"

        if is_ood:
            is_safe = final_score < 0.45
            if is_safe:
                dataset_stats[ds_name]["ood_safe"] += 1
            elif t3_triggered:
                t3_ood_escapes += 1
            print(f"  [{ds_name[:9]}] \"{q[:42]}...\" ➜ Tier: {tier_exec:<25} | Score: {final_score:.4f} | Lat: {lat_ms:5.1f}ms | OOD Safe: {'✅' if is_safe else '❌'}")
        else:
            t_lower = target.lower() if target else ""
            is_top1 = final_hits and t_lower in final_hits[0]["question"].lower()
            is_top3 = any(t_lower in h["question"].lower() for h in final_hits[:3])

            rank = 0
            for r_idx, h in enumerate(final_hits, 1):
                if t_lower in h["question"].lower():
                    rank = r_idx
                    break
            rr = (1.0 / rank) if rank > 0 else 0.0
            dataset_stats[ds_name]["rr"].append(rr)

            if is_top1:
                dataset_stats[ds_name]["top1"] += 1
            if is_top3:
                dataset_stats[ds_name]["top3"] += 1

            if not t1_t2_passed:
                t1_t2_failed_count += 1
                if is_top1 and t3_triggered:
                    t3_recovered_count += 1

            if t3_triggered and not is_top1:
                t3_false_recoveries += 1

            status = "✅" if is_top1 else ("🟡" if is_top3 else "❌")
            print(f"  [{ds_name[:9]}] \"{q[:42]}...\" ➜ Tier: {tier_exec:<25} | Score: {final_score:.4f} | Lat: {lat_ms:5.1f}ms | Match: {final_match} {status}")

    # ── Summary Calculations ───────────────────────────────────────────────────
    print("\n=================================================================================================")
    print("📊 1. SYSTEMATIC 4-DATASET ACCURACY & RECALL PROFILE")
    print("=================================================================================================")
    print(f"{'Dataset Category':<30} | {'Count':<6} | {'Top-1':<7} | {'Top-3':<7} | {'MRR':<7} | {'P50 Lat':<9} | {'P95 Lat':<9}")
    print("-" * 92)
    for ds_name, stats in dataset_stats.items():
        tot = stats["total"]
        lats = sorted(stats["latencies"])
        p50_l = statistics.median(lats)
        p95_l = statistics.quantiles(lats, n=20)[18] if len(lats) >= 20 else max(lats)
        if "Out-of-Domain" in ds_name:
            ood_pct = (stats["ood_safe"] / tot) * 100
            print(f"{ds_name:<30} | {tot:6d} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {p50_l:>7.1f}ms | {p95_l:>7.1f}ms (OOD Safe: {ood_pct:.1f}%)")
        else:
            t1_pct = (stats["top1"] / tot) * 100
            t3_pct = (stats["top3"] / tot) * 100
            mrr_v = statistics.mean(stats["rr"]) if stats["rr"] else 0.0
            print(f"{ds_name:<30} | {tot:6d} | {t1_pct:>5.1f}% | {t3_pct:>5.1f}% | {mrr_v:>7.4f} | {p50_l:>7.1f}ms | {p95_l:>7.1f}ms")

    print("\n=================================================================================================")
    print("📊 2. TIER 3 OBSERVABILITY METRICS & RECOVERY INTEGRITY")
    print("=================================================================================================")
    total_in_domain = sum(stats["total"] for name, stats in dataset_stats.items() if "Out-of-Domain" not in name)
    t3_trigger_rate = (tier_counts["tier3_llm_escape_hatch"] / len(SYSTEMATIC_DATASET)) * 100
    t3_recovery_rate = (t3_recovered_count / t1_t2_failed_count * 100) if t1_t2_failed_count > 0 else 0.0
    t3_false_recovery_rate = (t3_false_recoveries / t3_triggered_total * 100) if t3_triggered_total > 0 else 0.0
    t3_ood_escape_rate = (t3_ood_escapes / dataset_stats["Dataset D: Out-of-Domain"]["total"] * 100)

    print(f"  • Tier 3 Trigger Rate       : {tier_counts['tier3_llm_escape_hatch']} / {len(SYSTEMATIC_DATASET)} queries ({t3_trigger_rate:.1f}%)")
    print(f"  • Queries Failing Tier 1/2  : {t1_t2_failed_count} / {total_in_domain}")
    print(f"  • Tier 3 Recovery Rate      : {t3_recovery_rate:.1f}%")
    print(f"  • Tier 3 False Recovery Rate: {t3_false_recovery_rate:.1f}% (Target: 0.0% - No wrong document pushes)")
    print(f"  • Tier 3 OOD Escape Rate    : {t3_ood_escape_rate:.1f}% (Target: 0.0% - No hallucinations)")

    print("\n=================================================================================================")
    print("📊 3. TIER-SPECIFIC LATENCY PROFILES")
    print("=================================================================================================")
    t1_t2_lats = tier_latencies["tier1_raw_fastpath"] + tier_latencies["tier2_local_deterministic"]
    if t1_t2_lats:
        print(f"  • Tier 1 + Tier 2 (Local Path) : P50 = {statistics.median(t1_t2_lats):5.1f} ms | P95 = {max(t1_t2_lats):5.1f} ms (Target: < 50ms)")
    if tier_latencies["tier3_llm_escape_hatch"]:
        t3_lats = tier_latencies["tier3_llm_escape_hatch"]
        print(f"  • Tier 3 (LLM Escape Hatch)    : P50 = {statistics.median(t3_lats):5.1f} ms | P95 = {max(t3_lats):5.1f} ms (Target Timeout: <= 1.5s)")
    print("=================================================================================================\n")


if __name__ == "__main__":
    asyncio.run(run_systematic_validation())
