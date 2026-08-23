import asyncio
import time
# pyrefly: ignore [missing-import]
import httpx
from typing import Dict, List, Any

# Dataset of benchmark test cases mapped to expected FAQ ID substring / question pattern
BENCHMARK_CASES = [
    # ── Category 1: Direct English Questions ──────────────────────────────────
    {
        "query": "How do I create an account?",
        "expected_match": "create an account",
        "category": "English Direct",
    },
    {
        "query": "How do I update my payment method?",
        "expected_match": "payment method",
        "category": "English Direct",
    },
    {
        "query": "How do I connect WhatsApp?",
        "expected_match": "WhatsApp",
        "category": "English Direct",
    },
    {
        "query": "Is there a free trial available?",
        "expected_match": "free trial",
        "category": "English Direct",
    },

    # ── Category 2: Paraphrased / Informal English ───────────────────────────
    {
        "query": "Where can I put my new credit card?",
        "expected_match": "payment method",
        "category": "English Paraphrased",
    },
    {
        "query": "Can I test the service for 14 days without paying?",
        "expected_match": "free trial",
        "category": "English Paraphrased",
    },
    {
        "query": "My bot stopped answering users, what happened?",
        "expected_match": "not responding",
        "category": "English Paraphrased",
    },

    # ── Category 3: Standard Bengali (বাংলা) ──────────────────────────────────
    {
        "query": "আমি কীভাবে একটি নতুন অ্যাকাউন্ট তৈরি করবো?",
        "expected_match": "create an account",
        "category": "Bengali (বাংলা)",
    },
    {
        "query": "পেমেন্ট মেথড বা কার্ড কীভাবে পরিবর্তন করতে পারি?",
        "expected_match": "payment method",
        "category": "Bengali (বাংলা)",
    },
    {
        "query": "হোয়াটসঅ্যাপ কীভাবে কানেক্ট করতে হয়?",
        "expected_match": "WhatsApp",
        "category": "Bengali (বাংলা)",
    },
    {
        "query": "আপনাদের কি কোনো ফ্রি ট্রায়াল আছে?",
        "expected_match": "free trial",
        "category": "Bengali (বাংলা)",
    },
    {
        "query": "টু-ফ্যাক্টর অথেনটিকেশন কিভাবে চালু করবো?",
        "expected_match": "two-factor",
        "category": "Bengali (বাংলা)",
    },

    # ── Category 4: Banglish (বাংলিশ) ─────────────────────────────────────────
    {
        "query": "kivabe id khulbo?",
        "expected_match": "create an account",
        "category": "Banglish",
    },
    {
        "query": "card kivabe change korbo billing theke?",
        "expected_match": "payment method",
        "category": "Banglish",
    },
    {
        "query": "whatsapp add korar process ki?",
        "expected_match": "WhatsApp",
        "category": "Banglish",
    },
    {
        "query": "free te use kora jabe koyek din?",
        "expected_match": "free trial",
        "category": "Banglish",
    },
    {
        "query": "bot keno reply dicche na?",
        "expected_match": "not responding",
        "category": "Banglish",
    },
]

import sys
import io

# Set UTF-8 encoding for Windows stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SEARCH_URL = "http://127.0.0.1:8001/api/v1/search"


async def run_benchmark():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE RETRIEVAL QUALITY BENCHMARK")
    print("=" * 80)

    total_tests = len(BENCHMARK_CASES)
    top1_correct = 0
    top3_correct = 0
    expansion_count = 0
    latencies = []

    results_table = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx, tc in enumerate(BENCHMARK_CASES, 1):
            query = tc["query"]
            expected = tc["expected_match"].lower()
            cat = tc["category"]

            start_t = time.time()
            resp = await client.post(SEARCH_URL, json={
                "query": query,
                "workspace_id": 1,
                "top_k": 3,
            })
            duration_ms = round((time.time() - start_t) * 1000, 1)
            latencies.append(duration_ms)

            if resp.status_code != 200:
                print(f"[{idx}/{total_tests}] FAILED HTTP {resp.status_code} for query: '{query}'")
                continue

            data = resp.json()
            hits = data.get("results", [])
            expanded = data.get("expansion_applied", False)
            if expanded:
                expansion_count += 1

            top1_hit = hits[0] if hits else None
            top1_match = False
            top3_match = False

            if top1_hit and expected in (top1_hit["question"] + " " + top1_hit["answer"]).lower():
                top1_match = True
                top1_correct += 1

            for h in hits[:3]:
                if expected in (h["question"] + " " + h["answer"]).lower():
                    top3_match = True
                    break

            if top3_match:
                top3_correct += 1

            top_score = top1_hit["score"] if top1_hit else 0.0
            top_q = (top1_hit["question"][:35] + "...") if top1_hit else "NO_MATCH"

            results_table.append({
                "id": idx,
                "query": query,
                "category": cat,
                "top1_q": top_q,
                "top_score": top_score,
                "top1_ok": "PASS" if top1_match else "FAIL",
                "top3_ok": "PASS" if top3_match else "FAIL",
                "expanded": "Yes" if expanded else "No",
                "latency_ms": duration_ms,
            })

    # Print Table
    print(f"{'#':<3} | {'Query':<45} | {'Top-1':<6} | {'Score':<6} | {'Exp?':<5} | {'Time(ms)':<8} | {'Category'}")
    print("-" * 105)
    for r in results_table:
        print(f"{r['id']:<3} | {r['query']:<45} | {r['top1_ok']:<6} | {r['top_score']:<6.3f} | {r['expanded']:<5} | {r['latency_ms']:<8.1f} | {r['category']}")

    print("=" * 80)
    print("BENCHMARK METRIC SUMMARY:")
    print("=" * 80)
    top1_acc = round((top1_correct / total_tests) * 100, 2)
    top3_rec = round((top3_correct / total_tests) * 100, 2)
    exp_rate = round((expansion_count / total_tests) * 100, 2)
    avg_latency = round(sum(latencies) / len(latencies), 1)

    print(f"Total Benchmark Queries:    {total_tests}")
    print(f"Top-1 Accuracy:             {top1_acc}% ({top1_correct}/{total_tests})")
    print(f"Top-3 Recall:               {top3_rec}% ({top3_correct}/{total_tests})")
    print(f"LLM Expansion Trigger Rate: {exp_rate}% ({expansion_count}/{total_tests})")
    print(f"Average Search Latency:     {avg_latency} ms")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

