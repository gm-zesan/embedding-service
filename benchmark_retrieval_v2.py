import asyncio
import io
import json
import sys
import time
from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
import httpx

# Set UTF-8 encoding for Windows stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEARCH_URL = "http://127.0.0.1:8001/api/v1/search"

# Expanded, diverse dataset (In-domain, Paraphrased, Banglish, Typos, and Out-of-Domain/Negatives)
EVAL_DATASET = [
    # ── Category 1: Direct English In-Domain ─────────────────────────────────
    {"query": "How do I create an account?", "expected": "create an account", "cat": "Direct English", "is_ood": False},
    {"query": "How do I update my payment method?", "expected": "payment method", "cat": "Direct English", "is_ood": False},
    {"query": "How do I connect WhatsApp?", "expected": "whatsapp", "cat": "Direct English", "is_ood": False},
    {"query": "Is there a free trial?", "expected": "free trial", "cat": "Direct English", "is_ood": False},
    {"query": "How do I connect Telegram?", "expected": "telegram", "cat": "Direct English", "is_ood": False},
    {"query": "What are the API rate limits?", "expected": "rate limits", "cat": "Direct English", "is_ood": False},
    {"query": "How is my data encrypted?", "expected": "encrypted", "cat": "Direct English", "is_ood": False},
    {"query": "How do I enable two-factor authentication?", "expected": "two-factor", "cat": "Direct English", "is_ood": False},

    # ── Category 2: Paraphrased / Informal English ───────────────────────────
    {"query": "Where can I put my new credit card?", "expected": "payment method", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Can I test the service for 14 days without paying?", "expected": "free trial", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "My bot stopped answering users, what happened?", "expected": "not responding", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Can I hook up both telegram and whatsapp at once?", "expected": "multiple channels", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Is there any price cut for registered charities?", "expected": "non-profits", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How do I make my bot give more accurate answers?", "expected": "response accuracy", "cat": "Paraphrased EN", "is_ood": False},

    # ── Category 3: Standard Bengali (বাংলা) ──────────────────────────────────
    {"query": "আমি কীভাবে একটি নতুন অ্যাকাউন্ট তৈরি করবো?", "expected": "create an account", "cat": "Bengali", "is_ood": False},
    {"query": "পেমেন্ট মেথড বা কার্ড কীভাবে পরিবর্তন করতে পারি?", "expected": "payment method", "cat": "Bengali", "is_ood": False},
    {"query": "হোয়াটসঅ্যাপ কীভাবে কানেক্ট করতে হয়?", "expected": "whatsapp", "cat": "Bengali", "is_ood": False},
    {"query": "আপনাদের কি কোনো ফ্রি ট্রায়াল আছে?", "expected": "free trial", "cat": "Bengali", "is_ood": False},
    {"query": "টু-ফ্যাক্টর অথেনটিকেশন কিভাবে চালু করবো?", "expected": "two-factor", "cat": "Bengali", "is_ood": False},
    {"query": "টেলিগ্রাম কীভাবে যুক্ত করতে হয়?", "expected": "telegram", "cat": "Bengali", "is_ood": False},
    {"query": "এপিআই রিকোয়েস্টে কিভাবে অথেনটিকেশন করব?", "expected": "authenticate api", "cat": "Bengali", "is_ood": False},
    {"query": "চ্যাটবট যদি কোনো রেসপন্স না দেয় তবে কি করব?", "expected": "not responding", "cat": "Bengali", "is_ood": False},

    # ── Category 4: Banglish (বাংলিশ) ─────────────────────────────────────────
    {"query": "kivabe id khulbo?", "expected": "create an account", "cat": "Banglish", "is_ood": False},
    {"query": "card kivabe change korbo billing theke?", "expected": "payment method", "cat": "Banglish", "is_ood": False},
    {"query": "whatsapp add korar process ki?", "expected": "whatsapp", "cat": "Banglish", "is_ood": False},
    {"query": "free te use kora jabe koyek din?", "expected": "free trial", "cat": "Banglish", "is_ood": False},
    {"query": "bot keno reply dicche na?", "expected": "not responding", "cat": "Banglish", "is_ood": False},
    {"query": "telegram channel kivabe integrate korbo?", "expected": "telegram", "cat": "Banglish", "is_ood": False},
    {"query": "akta workspace e koyta channel use kora jay?", "expected": "multiple channels", "cat": "Banglish", "is_ood": False},
    {"query": "non-profit organization der jonno kono discount ache?", "expected": "non-profits", "cat": "Banglish", "is_ood": False},

    # ── Category 5: Out of Domain / Negatives (Should NOT hallucinate or overmatch)
    {"query": "What is the weather in Dhaka today?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Can I book a flight to London?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Tell me a recipe for chicken biryani", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Who is the president of Bangladesh?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
]


async def run_evaluation_v2():
    print("=" * 110)
    print("🔬 RETRIEVAL EVALUATION v2: GRANULAR BENCHMARK & REGRESSION SUITE")
    print("=" * 110)

    total_in_domain = sum(1 for x in EVAL_DATASET if not x["is_ood"])
    total_ood = sum(1 for x in EVAL_DATASET if x["is_ood"])

    top1_correct = 0
    top3_correct = 0
    expansion_count = 0
    ood_low_confidence_count = 0

    fast_path_latencies = []
    expansion_latencies = []
    all_latencies = []

    eval_records = []

    async with httpx.AsyncClient(timeout=20.0) as client:
        for idx, item in enumerate(EVAL_DATASET, 1):
            query = item["query"]
            expected = item["expected"].lower()
            cat = item["cat"]
            is_ood = item["is_ood"]

            t0 = time.time()
            resp = await client.post(SEARCH_URL, json={
                "query": query,
                "workspace_id": 1,
                "top_k": 3,
            })
            total_duration_ms = round((time.time() - t0) * 1000, 1)
            all_latencies.append(total_duration_ms)

            if resp.status_code != 200:
                print(f"[{idx}] FAILED HTTP {resp.status_code}: {query}")
                continue

            data = resp.json()
            hits = data.get("results", [])
            expanded = data.get("expansion_applied", False)
            expanded_terms = data.get("expanded_query")

            if expanded:
                expansion_count += 1
                expansion_latencies.append(total_duration_ms)
            else:
                fast_path_latencies.append(total_duration_ms)

            top1_hit = hits[0] if hits else None
            top_score = top1_hit["score"] if top1_hit else 0.0
            top1_text = (top1_hit["question"] + " " + top1_hit["answer"]).lower() if top1_hit else ""

            if is_ood:
                # Out-Of-Domain queries: safe if top_score < 0.65
                ood_safe = (top_score < 0.65)
                if ood_safe:
                    ood_low_confidence_count += 1
                outcome = "SAFE_LOW_CONF" if ood_safe else "UNSAFE_HIGH_CONF"
            else:
                top1_match = expected in top1_text
                if top1_match:
                    top1_correct += 1

                top3_match = any(expected in (h["question"] + " " + h["answer"]).lower() for h in hits[:3])
                if top3_match:
                    top3_correct += 1

                if not expanded:
                    outcome = "FIRST_PASS_HIT" if top1_match else "FIRST_PASS_MISS"
                else:
                    # In expanded queries, check if expansion recovered or regressed
                    if top1_match:
                        outcome = "EXPANSION_RECOVERED"
                    else:
                        outcome = "EXPANSION_NO_CHANGE"

            eval_records.append({
                "id": idx,
                "query": query,
                "category": cat,
                "score": top_score,
                "expanded": "Yes" if expanded else "No",
                "expanded_terms": expanded_terms or "-",
                "latency_ms": total_duration_ms,
                "outcome": outcome,
                "top1_title": (top1_hit["question"][:32] + "...") if top1_hit else "None",
            })

    # Print Detailed Table
    print(f"{'#':<3} | {'Query':<44} | {'Score':<6} | {'Exp?':<5} | {'Latency':<9} | {'Outcome':<15} | {'Category'}")
    print("-" * 110)
    for r in eval_records:
        print(f"{r['id']:<3} | {r['query']:<44} | {r['score']:<6.3f} | {r['expanded']:<5} | {r['latency_ms']:<7.1f}ms | {r['outcome']:<15} | {r['category']}")

    print("=" * 110)
    print("📊 RETRIEVAL EVALUATION v2 SUMMARY METRICS:")
    print("=" * 110)
    top1_acc = round((top1_correct / total_in_domain) * 100, 2)
    top3_rec = round((top3_correct / total_in_domain) * 100, 2)
    ood_safe_rate = round((ood_low_confidence_count / total_ood) * 100, 2) if total_ood else 100.0
    overall_exp_rate = round((expansion_count / len(EVAL_DATASET)) * 100, 2)

    avg_fast_path = round(sum(fast_path_latencies) / len(fast_path_latencies), 1) if fast_path_latencies else 0.0
    avg_expansion = round(sum(expansion_latencies) / len(expansion_latencies), 1) if expansion_latencies else 0.0
    avg_total = round(sum(all_latencies) / len(all_latencies), 1)

    print(f"In-Domain Test Queries:         {total_in_domain}")
    print(f"Top-1 Accuracy:                 {top1_acc}% ({top1_correct}/{total_in_domain})")
    print(f"Top-3 Recall:                   {top3_rec}% ({top3_correct}/{total_in_domain})")
    print(f"Out-of-Domain Queries:          {total_ood}")
    print(f"Out-of-Domain Low-Conf Safety:  {ood_safe_rate}% ({ood_low_confidence_count}/{total_ood})")
    print(f"Total Expansion Trigger Rate:   {overall_exp_rate}% ({expansion_count}/{len(EVAL_DATASET)})")
    print(f"Avg Python Fast-Path Latency:   {avg_fast_path} ms")
    print(f"Avg Python Expansion Latency:   {avg_expansion} ms")
    print(f"Avg Overall Retrieval Latency:  {avg_total} ms")
    print("=" * 110)


if __name__ == "__main__":
    asyncio.run(run_evaluation_v2())
