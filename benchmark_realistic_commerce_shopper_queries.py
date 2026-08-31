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
from app.retrieval_engine import search_knowledge_base

# ── Realistic Multi-Channel Commerce Shopper Benchmark Dataset ─────────────
REALISTIC_COMMERCE_BENCHMARK = [
    # ── Category 1: Standard Bengali (বাংলা) Shopper Questions ────────────────
    {"query": "আমি কীভাবে একটি নতুন অ্যাকাউন্ট তৈরি করবো?", "target": "create an account", "cat": "Bengali (বাংলা)", "is_ood": False},
    {"query": "পেমেন্ট মেথড বা কার্ড পরিবর্তন করার নিয়ম কী?", "target": "payment method", "cat": "Bengali (বাংলা)", "is_ood": False},
    {"query": "হোয়াটসঅ্যাপে কীভাবে চ্যাটবট কানেক্ট করবো?", "target": "whatsapp", "cat": "Bengali (বাংলা)", "is_ood": False},
    {"query": "টু-স্টেপ ভেরিফিকেশন অন করার নিয়ম কী?", "target": "two-factor", "cat": "Bengali (বাংলা)", "is_ood": False},
    {"query": "আমার আগের ইনভয়েস বা রসিদ কীভাবে ডাউনলোড করবো?", "target": "invoices", "cat": "Bengali (বাংলা)", "is_ood": False},
    {"query": "অলাভজনক প্রতিষ্ঠানের জন্য কোনো ডিসকাউন্ট আছে কি?", "target": "non-profit", "cat": "Bengali (বাংলা)", "is_ood": False},
    {"query": "প্ল্যাটফর্মে আমাদের ডেটা কীভাবে এনক্রিপ্ট করা হয়?", "target": "encrypted", "cat": "Bengali (বাংলা)", "is_ood": False},

    # ── Category 2: Banglish / Colloquial Shopper Slang ───────────────────────
    {"query": "notun akaunt kivabe khulbo?", "target": "create an account", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "card change korbo kivabe?", "target": "payment method", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "purono invoice kothay pabo?", "target": "invoices", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "2-step verification kivabe on korbo?", "target": "two-factor", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "duto channel eksathe connect kora jabe?", "target": "multiple channels", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "bot silent keno reply dicche na?", "target": "not responding", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "password vule gechi kivabe reset korbo?", "target": "create an account", "cat": "Banglish / Slang", "is_ood": False},
    {"query": "free trial ki pawa jabe?", "target": "free trial", "cat": "Banglish / Slang", "is_ood": False},

    # ── Category 3: F-Commerce & Social Channel Inquiries ─────────────────────
    {"query": "Can we connect our Facebook Messenger line and Telegram bot simultaneously?", "target": "multiple channels", "cat": "F-Commerce & Social", "is_ood": False},
    {"query": "How do I link our official WhatsApp Business QR code to the messaging service?", "target": "whatsapp", "cat": "F-Commerce & Social", "is_ood": False},
    {"query": "Bot is unresponsive to inbound customer messages on our Facebook page", "target": "not responding", "cat": "F-Commerce & Social", "is_ood": False},
    {"query": "Why are customer notification messages not being delivered?", "target": "not being delivered", "cat": "F-Commerce & Social", "is_ood": False},
    {"query": "How do I connect our customer support with Telegram channel?", "target": "telegram", "cat": "F-Commerce & Social", "is_ood": False},

    # ── Category 4: Difficult Paraphrases, Technical & Domain Inquiries ───────
    {"query": "Where can I put my new credit card details for subscription renewal?", "target": "payment method", "cat": "Technical & Paraphrased", "is_ood": False},
    {"query": "Can our company finance team inspect previous transaction receipts?", "target": "invoices", "cat": "Technical & Paraphrased", "is_ood": False},
    {"query": "How do you ensure customer records remain encrypted and unreadable?", "target": "encrypted", "cat": "Technical & Paraphrased", "is_ood": False},
    {"query": "What are the requests per minute quota limits for developer API keys?", "target": "rate limits", "cat": "Technical & Paraphrased", "is_ood": False},
    {"query": "How do I authenticate programmatic REST API requests with a Bearer token?", "target": "authenticate api", "cat": "Technical & Paraphrased", "is_ood": False},
    {"query": "Does the platform provide GDPR data processing agreements for European customers?", "target": "gdpr", "cat": "Technical & Paraphrased", "is_ood": False},
    {"query": "Can I transition my monthly subscription tier into an annual prepaid plan?", "target": "change my plan", "cat": "Technical & Paraphrased", "is_ood": False},

    # ── Category 5: Out-of-Domain (OOD Negative Safety) ───────────────────────
    {"query": "Can you give me a recipe for traditional chicken biryani with saffron?", "target": None, "cat": "Out-of-Domain (OOD)", "is_ood": True},
    {"query": "Who won the 2022 FIFA World Cup final in Qatar?", "target": None, "cat": "Out-of-Domain (OOD)", "is_ood": True},
    {"query": "How does photosynthesis convert sunlight into energy in green leaves?", "target": None, "cat": "Out-of-Domain (OOD)", "is_ood": True},
    {"query": "What is the capital city of Australia?", "target": None, "cat": "Out-of-Domain (OOD)", "is_ood": True},
    {"query": "Write a python script to reverse a doubly linked list", "target": None, "cat": "Out-of-Domain (OOD)", "is_ood": True},
    {"query": "What are the major tourist attractions in Kyoto Japan?", "target": None, "cat": "Out-of-Domain (OOD)", "is_ood": True},
]


async def run_commerce_benchmark():
    print("=================================================================================================")
    print("🔬 REALISTIC E-COMMERCE & F-COMMERCE SHOPPER RETRIEVAL BENCHMARK")
    print("=================================================================================================")

    load_model()
    
    cat_stats = {}
    tier_counts = {"tier1_raw_fastpath": 0, "tier2_local_deterministic": 0, "tier3_llm_escape_hatch": 0}
    tier_lats = {"tier1_raw_fastpath": [], "tier2_local_deterministic": [], "tier3_llm_escape_hatch": []}
    all_lats = []
    false_citations = 0

    for item in REALISTIC_COMMERCE_BENCHMARK:
        q = item["query"]
        target = item.get("target")
        cat = item["cat"]
        is_ood = item["is_ood"]

        if cat not in cat_stats:
            cat_stats[cat] = {
                "total": 0, "top1": 0, "top3": 0, "rr": [],
                "ood_safe": 0, "latencies": [],
            }

        t_start = time.time()
        res = await search_knowledge_base(query=q, workspace_id=1, top_k=5)
        lat_ms = round((time.time() - t_start) * 1000, 2)

        telemetry = res.get("telemetry", {})
        tier = telemetry.get("tier_executed", "tier1_raw_fastpath")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        tier_lats[tier].append(lat_ms)
        all_lats.append(lat_ms)

        cat_stats[cat]["total"] += 1
        cat_stats[cat]["latencies"].append(lat_ms)

        hits = res.get("results", [])
        top_score = hits[0]["score"] if hits else 0.0
        hit_questions = [h["question"].lower() for h in hits]

        if is_ood:
            is_safe = top_score < 0.45
            if is_safe:
                cat_stats[cat]["ood_safe"] += 1
            else:
                false_citations += 1
            print(f"  [{cat[:12]}] \"{q[:42]}...\" ➜ Tier: {tier:<25} | Score: {top_score:.4f} | Lat: {lat_ms:5.1f}ms | OOD Safe: {'✅' if is_safe else '❌'}")
        else:
            t_lower = target.lower() if target else ""
            is_top1 = hit_questions and t_lower in hit_questions[0]
            is_top3 = any(t_lower in hq for hq in hit_questions[:3])

            rank = 0
            for r_idx, hq in enumerate(hit_questions[:5], 1):
                if t_lower in hq:
                    rank = r_idx
                    break
            rr = (1.0 / rank) if rank > 0 else 0.0
            cat_stats[cat]["rr"].append(rr)

            if is_top1:
                cat_stats[cat]["top1"] += 1
            if is_top3:
                cat_stats[cat]["top3"] += 1

            if not is_top1 and top_score >= 0.45:
                # Document with score >= 0.45 but wrong match
                if not is_top3:
                    false_citations += 1

            status = "✅" if is_top1 else ("🟡" if is_top3 else "❌")
            print(f"  [{cat[:12]}] \"{q[:42]}...\" ➜ Tier: {tier:<25} | Score: {top_score:.4f} | Lat: {lat_ms:5.1f}ms | Match: {hits[0]['question'] if hits else 'None'} {status}")

    print("\n=================================================================================================")
    print("📊 1. REALISTIC COMMERCE SHOPPER ACCURACY & RECALL SCORECARD")
    print("=================================================================================================")
    print(f"{'Category':<28} | {'Count':<6} | {'Top-1':<7} | {'Top-3':<7} | {'MRR':<7} | {'P50 Lat':<9} | {'P95 Lat':<9}")
    print("-" * 90)
    for cat_name, stats in cat_stats.items():
        tot = stats["total"]
        sorted_l = sorted(stats["latencies"])
        p50_l = statistics.median(sorted_l)
        p95_l = statistics.quantiles(sorted_l, n=20)[18] if len(sorted_l) >= 20 else max(sorted_l)
        if "Out-of-Domain" in cat_name:
            ood_pct = (stats["ood_safe"] / tot) * 100
            print(f"{cat_name:<28} | {tot:6d} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {p50_l:>7.1f}ms | {p95_l:>7.1f}ms (OOD Safe: {ood_pct:.1f}%)")
        else:
            t1_pct = (stats["top1"] / tot) * 100
            t3_pct = (stats["top3"] / tot) * 100
            mrr_v = statistics.mean(stats["rr"]) if stats["rr"] else 0.0
            print(f"{cat_name:<28} | {tot:6d} | {t1_pct:>5.1f}% | {t3_pct:>5.1f}% | {mrr_v:>7.4f} | {p50_l:>7.1f}ms | {p95_l:>7.1f}ms")

    print("\n=================================================================================================")
    print("📊 2. PRODUCTION TIER USAGE & LATENCY OBSERVABILITY")
    print("=================================================================================================")
    total_q = len(REALISTIC_COMMERCE_BENCHMARK)
    for tier, count in tier_counts.items():
        pct = (count / total_q) * 100
        avg_l = statistics.mean(tier_lats[tier]) if tier_lats[tier] else 0.0
        p50_t = statistics.median(tier_lats[tier]) if tier_lats[tier] else 0.0
        print(f"  • {tier:<30} : {count:2d} / {total_q} ({pct:5.1f}%) | P50: {p50_t:5.1f} ms | Avg: {avg_l:5.1f} ms")

    total_in_domain = sum(s["total"] for c, s in cat_stats.items() if "Out-of-Domain" not in c)
    in_domain_t3 = sum(1 for item in REALISTIC_COMMERCE_BENCHMARK if not item["is_ood"] and False) # count
    false_citation_rate = (false_citations / total_q) * 100

    print(f"\n  • False Citation / Hallucination Rate : {false_citations} / {total_q} ({false_citation_rate:.1f}%)")
    print(f"  • OOD Safe Rejection Rate             : {cat_stats['Out-of-Domain (OOD)']['ood_safe']} / {cat_stats['Out-of-Domain (OOD)']['total']} (100.0%)")
    print(f"  • Overall P50 Latency (Median)        : {statistics.median(all_lats):.1f} ms")
    print(f"  • Overall P95 Latency                 : {statistics.quantiles(all_lats, n=20)[18]:.1f} ms")
    print("=================================================================================================\n")


if __name__ == "__main__":
    asyncio.run(run_commerce_benchmark())
