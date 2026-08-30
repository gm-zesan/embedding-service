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

# ── Stress Evaluation Suite: Challenging Unseen Long-Tail Queries ──────────
TIER3_ISOLATION_DATASET = [
    # ── In-Domain Difficult Queries (Zero direct overlap with LOCAL_DOMAIN_SYNONYMS)
    {
        "query": "Where can I find the official tax documentation for our corporate expenses?",
        "target": "invoices",
        "description": "Tax Documentation for Corporate Expenses",
        "is_ood": False
    },
    {
        "query": "I need to swap out my expired Mastercard on file before the next billing cycle",
        "target": "payment method",
        "description": "Swap Expired Mastercard on File",
        "is_ood": False
    },
    {
        "query": "How do we integrate our company customer helpdesk with Meta WhatsApp Business Cloud API?",
        "target": "whatsapp",
        "description": "WhatsApp Business Cloud API Integration",
        "is_ood": False
    },
    {
        "query": "Is there any safety feature requiring an authenticator code on my smartphone when logging in?",
        "target": "two-factor",
        "description": "Smartphone Authenticator Code Login Security",
        "is_ood": False
    },
    {
        "query": "Our automated agent is sitting idle and not answering any visitor questions on our webpage",
        "target": "not responding",
        "description": "Automated Agent Idle on Webpage",
        "is_ood": False
    },
    {
        "query": "We are an accredited charity foundation looking for reduced subscription pricing tiers",
        "target": "non-profit",
        "description": "Accredited Charity Foundation Special Pricing",
        "is_ood": False
    },
    {
        "query": "How can I transition my monthly subscription tier into an annual prepaid contract?",
        "target": "change my plan",
        "description": "Transition Monthly to Annual Prepaid Contract",
        "is_ood": False
    },
    {
        "query": "How do you ensure our customer records remain unreadable to unauthorized third parties?",
        "target": "encrypted",
        "description": "Customer Records Unreadable to Third Parties",
        "is_ood": False
    },

    # ── Out-of-Domain Safety Cases (Must NEVER create false recovery / hallucinations)
    {
        "query": "Can you provide a recipe for baking gluten-free chocolate chip cookies?",
        "target": None,
        "description": "Recipe for Gluten-Free Cookies (OOD)",
        "is_ood": True
    },
    {
        "query": "What is the capital city of Australia?",
        "target": None,
        "description": "Capital City of Australia (OOD)",
        "is_ood": True
    },
]


async def run_tier3_isolation():
    print("=================================================================================================")
    print("🔬 TIER 3 LLM EXPANSION ISOLATION & INCREMENTAL RECOVERY BENCHMARK")
    print("=================================================================================================")

    load_model()
    client = get_typesense_client()

    results = []

    for item in TIER3_ISOLATION_DATASET:
        q = item["query"]
        target = item["target"]
        desc = item["description"]
        is_ood = item["is_ood"]

        clean_q = preprocess_query(q)

        # ── Step 1: Tier 1 Only (Raw Dense Vector + Hybrid) ────────────────────
        t0 = time.time()
        v1 = embed(clean_q)
        s1 = execute_hybrid_search(client, clean_q, v1, workspace_id=1, top_k=5)
        h1 = parse_typesense_hits(s1.get("hits", []))
        t1_hits, _, _ = rerank_candidate_hits(clean_q, h1)
        t1_lat = round((time.time() - t0) * 1000, 2)
        t1_top = t1_hits[0] if t1_hits else None
        t1_score = t1_top["score"] if t1_top else 0.0
        t1_match = t1_top["question"] if t1_top else "None"

        # ── Step 2: Tier 1 + Tier 2 (Local Domain Expansion) ───────────────────
        t0 = time.time()
        local_q = expand_locally(clean_q)
        t2_applied = local_q != clean_q
        if t2_applied:
            v2 = embed(local_q)
            s2 = execute_hybrid_search(client, local_q, v2, workspace_id=1, top_k=5)
            h2 = parse_typesense_hits(s2.get("hits", []))
            hit_map = {h["id"]: h for h in h1}
            for h in h2:
                if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                    hit_map[h["id"]] = h
            fused2 = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:5]
        else:
            fused2 = h1
        t2_hits, _, _ = rerank_candidate_hits(clean_q, fused2)
        t2_lat = round((time.time() - t0) * 1000, 2)
        t2_top = t2_hits[0] if t2_hits else None
        t2_score = t2_top["score"] if t2_top else 0.0
        t2_match = t2_top["question"] if t2_top else "None"

        # ── Step 3: Tier 1 + Tier 2 + Tier 3 (LLM Expansion) ───────────────────
        t0 = time.time()
        llm_expanded_text = None
        t3_applied = False
        if config.LLM_EXPANSION_API_KEY:
            llm_expanded_text = await expand_query_via_llm(clean_q)
        
        if llm_expanded_text:
            t3_applied = True
            t3_search_q = f"{clean_q} {llm_expanded_text}"
            v3 = embed(t3_search_q)
            s3 = execute_hybrid_search(client, t3_search_q, v3, workspace_id=1, top_k=5)
            h3 = parse_typesense_hits(s3.get("hits", []))
            hit_map = {h["id"]: h for h in fused2}
            for h in h3:
                if h["id"] not in hit_map or h["score"] > hit_map[h["id"]]["score"]:
                    hit_map[h["id"]] = h
            fused3 = sorted(hit_map.values(), key=lambda x: (x["score"], x["priority"]), reverse=True)[:5]
        else:
            fused3 = fused2
        t3_hits, _, _ = rerank_candidate_hits(clean_q, fused3)
        t3_lat = round((time.time() - t0) * 1000, 2)
        t3_top = t3_hits[0] if t3_hits else None
        t3_score = t3_top["score"] if t3_top else 0.0
        t3_match = t3_top["question"] if t3_top else "None"

        # Evaluation accuracy checks
        if is_ood:
            t1_pass = t1_score < 0.45
            t2_pass = t2_score < 0.45
            t3_pass = t3_score < 0.45
        else:
            t_lower = target.lower() if target else ""
            t1_pass = t_lower in t1_match.lower()
            t2_pass = t_lower in t2_match.lower()
            t3_pass = t_lower in t3_match.lower()

        results.append({
            "desc": desc,
            "query": q,
            "is_ood": is_ood,
            "target": target,
            "t1_score": t1_score, "t1_match": t1_match, "t1_pass": t1_pass, "t1_lat": t1_lat,
            "t2_score": t2_score, "t2_match": t2_match, "t2_pass": t2_pass, "t2_lat": t2_lat, "t2_applied": t2_applied,
            "t3_score": t3_score, "t3_match": t3_match, "t3_pass": t3_pass, "t3_lat": t3_lat, "t3_applied": t3_applied,
            "llm_terms": llm_expanded_text,
        })

        print(f"\n─────────────────────────────────────────────────────────────────────────────────────────────────")
        print(f"Query: \"{q}\" ({desc})")
        print(f"  • Tier 1 (Raw)       : Score: {t1_score:.4f} | Lat: {t1_lat:5.1f}ms | {'✅' if t1_pass else '❌'} {t1_match}")
        print(f"  • Tier 2 (Local Exp) : Score: {t2_score:.4f} | Lat: {t2_lat:5.1f}ms | {'✅' if t2_pass else '❌'} {t2_match} (Applied: {t2_applied})")
        print(f"  • Tier 3 (LLM Exp)   : Score: {t3_score:.4f} | Lat: {t3_lat:5.1f}ms | {'✅' if t3_pass else '❌'} {t3_match} (Terms: {llm_expanded_text})")

    # ── Summary Metrics ────────────────────────────────────────────────────────
    in_domain = [r for r in results if not r["is_ood"]]
    ood_cases = [r for r in results if r["is_ood"]]

    t1_top1_in = sum(1 for r in in_domain if r["t1_pass"]) / len(in_domain) * 100
    t2_top1_in = sum(1 for r in in_domain if r["t2_pass"]) / len(in_domain) * 100
    t3_top1_in = sum(1 for r in in_domain if r["t3_pass"]) / len(in_domain) * 100

    ood_t1_safe = sum(1 for r in ood_cases if r["t1_pass"]) / len(ood_cases) * 100
    ood_t2_safe = sum(1 for r in ood_cases if r["t2_pass"]) / len(ood_cases) * 100
    ood_t3_safe = sum(1 for r in ood_cases if r["t3_pass"]) / len(ood_cases) * 100

    t1_avg_lat = statistics.mean([r["t1_lat"] for r in results])
    t2_avg_lat = statistics.mean([r["t2_lat"] for r in results])
    t3_avg_lat = statistics.mean([r["t3_lat"] for r in results])

    print("\n=================================================================================================")
    print("📊 TIER 3 INCREMENTAL VALUE & LATENCY PROFILE SCORECARD")
    print("=================================================================================================")
    print(f"{'Retrieval Tier Stage':<35} | {'In-Domain Top-1':<17} | {'OOD Safety':<12} | {'Avg Latency':<12}")
    print("-" * 85)
    print(f"{'Tier 1 (Raw Dense Hybrid)':<35} | {t1_top1_in:>14.1f}%   | {ood_t1_safe:>9.1f}%  | {t1_avg_lat:>9.1f} ms")
    print(f"{'Tier 1 + Tier 2 (Local Domain)':<35} | {t2_top1_in:>14.1f}%   | {ood_t2_safe:>9.1f}%  | {t2_avg_lat:>9.1f} ms")
    print(f"{'Tier 1 + Tier 2 + Tier 3 (LLM)':<35} | {t3_top1_in:>14.1f}%   | {ood_t3_safe:>9.1f}%  | {t3_avg_lat:>9.1f} ms")
    print("=================================================================================================")

    # Incremental recovery calculation
    recovered_by_t3 = [r for r in in_domain if not r["t2_pass"] and r["t3_pass"]]
    print(f"\n🔍 Incremental Recovery Analysis:")
    print(f"  • Queries where Tier 1 & 2 Failed : {len([r for r in in_domain if not r['t2_pass']])} / {len(in_domain)}")
    print(f"  • Queries Recovered by Tier 3 LLM : {len(recovered_by_t3)}")
    if recovered_by_t3:
        for r in recovered_by_t3:
            print(f"     ↳ Recovered: \"{r['query']}\" ➜ {r['t3_match']} (Terms: {r['llm_terms']})")
    print(f"  • Incremental Recall Gain from Tier 3: +{(t3_top1_in - t2_top1_in):.1f}%\n")


if __name__ == "__main__":
    asyncio.run(run_tier3_isolation())
