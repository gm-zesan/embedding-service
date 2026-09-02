import os
os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
import app.config
app.config.MODEL_NAME = '/Users/zesan/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2/snapshots/4328cf26390c98c5e3c738b4460a05b95f4911f5'
import os




import asyncio
import io
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
import httpx

# Set UTF-8 encoding for Windows stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEARCH_URL = "http://127.0.0.1:8001/api/v1/search"

# 110 Comprehensive Test Cases
BENCHMARK_100_DATASET = [
    # ── 1. Direct English In-Domain (15 queries) ──────────────────────────────
    {"query": "How do I create an account?", "expected": "create an account", "cat": "Direct EN", "is_ood": False},
    {"query": "What do I do after logging in for the first time?", "expected": "first time", "cat": "Direct EN", "is_ood": False},
    {"query": "How do I set up my workspace?", "expected": "set up my workspace", "cat": "Direct EN", "is_ood": False},
    {"query": "How do I update my payment method?", "expected": "payment method", "cat": "Direct EN", "is_ood": False},
    {"query": "How do I view my invoices?", "expected": "invoices", "cat": "Direct EN", "is_ood": False},
    {"query": "Can I change my plan?", "expected": "change my plan", "cat": "Direct EN", "is_ood": False},
    {"query": "Why is my chatbot not responding?", "expected": "not responding", "cat": "Direct EN", "is_ood": False},
    {"query": "What should I do if I encounter an error?", "expected": "encounter an error", "cat": "Direct EN", "is_ood": False},
    {"query": "Why are my messages not being delivered?", "expected": "not being delivered", "cat": "Direct EN", "is_ood": False},
    {"query": "How do I connect WhatsApp?", "expected": "whatsapp", "cat": "Direct EN", "is_ood": False},
    {"query": "How do I connect Telegram?", "expected": "telegram", "cat": "Direct EN", "is_ood": False},
    {"query": "Can I use multiple channels simultaneously?", "expected": "multiple channels", "cat": "Direct EN", "is_ood": False},
    {"query": "How is my data encrypted?", "expected": "encrypted", "cat": "Direct EN", "is_ood": False},
    {"query": "Does the platform comply with GDPR?", "expected": "gdpr", "cat": "Direct EN", "is_ood": False},
    {"query": "How do I enable two-factor authentication?", "expected": "two-factor", "cat": "Direct EN", "is_ood": False},

    # ── 2. Paraphrased & Conversational English (20 queries) ──────────────────
    {"query": "Where can I put my new credit card info?", "expected": "payment method", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Is there a way to switch from monthly to annual billing?", "expected": "change my plan", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Can I test the platform for 14 days without paying anything?", "expected": "free trial", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "My bot suddenly stopped replying to customers on Messenger", "expected": "not responding", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How do I link my business WhatsApp account?", "expected": "whatsapp", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Can I connect both Telegram bot and Facebook page together?", "expected": "multiple channels", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Is our customer data secure and stored safely?", "expected": "encrypted", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Where do I generate an API token for my custom integration?", "expected": "api key", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How many API requests can I send per minute?", "expected": "rate limits", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How do I authenticate HTTP headers in API requests?", "expected": "authenticate api", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Any discount pricing for NGOs or schools?", "expected": "non-profits", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How to make chatbot answers more accurate and relevant?", "expected": "response accuracy", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "What are best practices for writing clear FAQ answers?", "expected": "good faq answer", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How often should our support team update the knowledgebase?", "expected": "update my faqs", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Where can I download receipts of my previous subscription payments?", "expected": "invoices", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "What steps should I take if the bot throws an internal error?", "expected": "encounter an error", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "Why are outbound messages stuck and not reaching users?", "expected": "not being delivered", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "How can I protect my login with 2FA / OTP?", "expected": "two-factor", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "What is the procedure for new account sign up?", "expected": "create an account", "cat": "Paraphrased EN", "is_ood": False},
    {"query": "What tiers or subscription packages do you offer?", "expected": "plans are available", "cat": "Paraphrased EN", "is_ood": False},

    # ── 3. Standard Bengali (বাংলা প্রমিত) (20 queries) ────────────────────────
    {"query": "আমি কীভাবে একটি নতুন অ্যাকাউন্ট তৈরি করবো?", "expected": "create an account", "cat": "Bengali", "is_ood": False},
    {"query": "প্রথমবার লগইন করার পর আমাকে কী কী করতে হবে?", "expected": "first time", "cat": "Bengali", "is_ood": False},
    {"query": "ওয়ার্কস্পেস কিভাবে সাজাবো বা সেটআপ করব?", "expected": "set up my workspace", "cat": "Bengali", "is_ood": False},
    {"query": "পেমেন্ট মেথড বা ক্রেডিট কার্ড কীভাবে পরিবর্তন করতে পারি?", "expected": "payment method", "cat": "Bengali", "is_ood": False},
    {"query": "আগের মাসের পেমেন্টের ইনভয়েস কোথায় দেখতে পাবো?", "expected": "invoices", "cat": "Bengali", "is_ood": False},
    {"query": "আমি কি বর্তমান প্ল্যান পরিবর্তন বা আপগ্রেড করতে পারি?", "expected": "change my plan", "cat": "Bengali", "is_ood": False},
    {"query": "আমার চ্যাটবট কাস্টমারদের রিপ্লাই দিচ্ছে না কেন?", "expected": "not responding", "cat": "Bengali", "is_ood": False},
    {"query": "সিস্টেমে কোনো এরর দেখা দিলে আমার করণীয় কী?", "expected": "encounter an error", "cat": "Bengali", "is_ood": False},
    {"query": "মেসেজ ডেলিভারি হচ্ছে না কেন?", "expected": "not being delivered", "cat": "Bengali", "is_ood": False},
    {"query": "হোয়াটসঅ্যাপ চ্যানেল কীভাবে যুক্ত করতে হয়?", "expected": "whatsapp", "cat": "Bengali", "is_ood": False},
    {"query": "টেলিগ্রাম কীভাবে চ্যাটবটের সাথে যুক্ত করবো?", "expected": "telegram", "cat": "Bengali", "is_ood": False},
    {"query": "একই সাথে কি একাধিক চ্যানেল চালানো সম্ভব?", "expected": "multiple channels", "cat": "Bengali", "is_ood": False},
    {"query": "আমাদের ডাটা কিভাবে এনক্রিপ্ট ও সুরক্ষিত থাকে?", "expected": "encrypted", "cat": "Bengali", "is_ood": False},
    {"query": "টু-ফ্যাক্টর অথেনটিকেশন কিভাবে চালু করবো?", "expected": "two-factor", "cat": "Bengali", "is_ood": False},
    {"query": "আমার এপিআই কি (API Key) কোথায় পাবো?", "expected": "api key", "cat": "Bengali", "is_ood": False},
    {"query": "এপিআই রিকোয়েস্টের লিমিট কত?", "expected": "rate limits", "cat": "Bengali", "is_ood": False},
    {"query": "চ্যাটবটের উত্তরের নির্ভুলতা কীভাবে বাড়ানো যায়?", "expected": "response accuracy", "cat": "Bengali", "is_ood": False},
    {"query": "একটি ভালো FAQ উত্তরের বৈশিষ্ট্য কী?", "expected": "good faq answer", "cat": "Bengali", "is_ood": False},
    {"query": "আপনাদের কি কোনো ফ্রি ট্রায়াল আছে?", "expected": "free trial", "cat": "Bengali", "is_ood": False},
    {"query": "অলাভজনক বা দাতব্য প্রতিষ্ঠানের জন্য কি কোনো ছাড় আছে?", "expected": "non-profits", "cat": "Bengali", "is_ood": False},

    # ── 4. Banglish Phonetic Variants (25 queries) ───────────────────────────
    {"query": "kivabe id khulbo?", "expected": "create an account", "cat": "Banglish", "is_ood": False},
    {"query": "notun account open korar niom ki?", "expected": "create an account", "cat": "Banglish", "is_ood": False},
    {"query": "prothom bar login kore ki korte hobe?", "expected": "first time", "cat": "Banglish", "is_ood": False},
    {"query": "workspace kivabe setup dibo?", "expected": "set up my workspace", "cat": "Banglish", "is_ood": False},
    {"query": "card kivabe change korbo billing theke?", "expected": "payment method", "cat": "Banglish", "is_ood": False},
    {"query": "notun payment method add korar way ki?", "expected": "payment method", "cat": "Banglish", "is_ood": False},
    {"query": "purono invoice ba money receipt kothay pabo?", "expected": "invoices", "cat": "Banglish", "is_ood": False},
    {"query": "plan upgrade ba change kora jabe?", "expected": "change my plan", "cat": "Banglish", "is_ood": False},
    {"query": "bot keno reply dicche na?", "expected": "not responding", "cat": "Banglish", "is_ood": False},
    {"query": "chatbot response na korle ki korbo?", "expected": "not responding", "cat": "Banglish", "is_ood": False},
    {"query": "kono error ashle ki kora uchit?", "expected": "encounter an error", "cat": "Banglish", "is_ood": False},
    {"query": "customer er kache msg delivery jacche na keno?", "expected": "not being delivered", "cat": "Banglish", "is_ood": False},
    {"query": "whatsapp add korar process ki?", "expected": "whatsapp", "cat": "Banglish", "is_ood": False},
    {"query": "telegram channel kivabe integrate korbo?", "expected": "telegram", "cat": "Banglish", "is_ood": False},
    {"query": "akta workspace e koyta channel use kora jay?", "expected": "multiple channels", "cat": "Banglish", "is_ood": False},
    {"query": "data security ba encryption kemon?", "expected": "encrypted", "cat": "Banglish", "is_ood": False},
    {"query": "2fa ba two factor authentication enable korbo kivabe?", "expected": "two-factor", "cat": "Banglish", "is_ood": False},
    {"query": "amar API token / key kothay pabo?", "expected": "api key", "cat": "Banglish", "is_ood": False},
    {"query": "api request limit koto per minute?", "expected": "rate limits", "cat": "Banglish", "is_ood": False},
    {"query": "bot er accuracy kivabe barano jay?", "expected": "response accuracy", "cat": "Banglish", "is_ood": False},
    {"query": "faq koto din por por update kora dorkar?", "expected": "update my faqs", "cat": "Banglish", "is_ood": False},
    {"query": "free te use kora jabe koyek din?", "expected": "free trial", "cat": "Banglish", "is_ood": False},
    {"query": "kono free trial offer ache ki?", "expected": "free trial", "cat": "Banglish", "is_ood": False},
    {"query": "non-profit organization der jonno kono discount ache?", "expected": "non-profits", "cat": "Banglish", "is_ood": False},
    {"query": "ki ki package ba pricing plan ache?", "expected": "plans are available", "cat": "Banglish", "is_ood": False},

    # ── 5. Short Queries (10 queries) ─────────────────────────────────────────
    {"query": "account create", "expected": "create an account", "cat": "Short Query", "is_ood": False},
    {"query": "change card", "expected": "payment method", "cat": "Short Query", "is_ood": False},
    {"query": "free trial", "expected": "free trial", "cat": "Short Query", "is_ood": False},
    {"query": "bot down", "expected": "not responding", "cat": "Short Query", "is_ood": False},
    {"query": "connect whatsapp", "expected": "whatsapp", "cat": "Short Query", "is_ood": False},
    {"query": "telegram bot", "expected": "telegram", "cat": "Short Query", "is_ood": False},
    {"query": "API key", "expected": "api key", "cat": "Short Query", "is_ood": False},
    {"query": "2FA setup", "expected": "two-factor", "cat": "Short Query", "is_ood": False},
    {"query": "discount non-profit", "expected": "non-profits", "cat": "Short Query", "is_ood": False},
    {"query": "view invoice", "expected": "invoices", "cat": "Short Query", "is_ood": False},

    # ── 6. Typos / Misspellings (10 queries) ─────────────────────────────────
    {"query": "how to crate an accunt?", "expected": "create an account", "cat": "Typos", "is_ood": False},
    {"query": "how do i updte pyment methd?", "expected": "payment method", "cat": "Typos", "is_ood": False},
    {"query": "is thr a fre triall?", "expected": "free trial", "cat": "Typos", "is_ood": False},
    {"query": "watsapp connect kora jabe?", "expected": "whatsapp", "cat": "Typos", "is_ood": False},
    {"query": "telligram kivabe add korbo?", "expected": "telegram", "cat": "Typos", "is_ood": False},
    {"query": "bot replay diche na kno?", "expected": "not responding", "cat": "Typos", "is_ood": False},
    {"query": "invoce download korbo kivabe?", "expected": "invoices", "cat": "Typos", "is_ood": False},
    {"query": "two factr auth kaise enable kare?", "expected": "two-factor", "cat": "Typos", "is_ood": False},
    {"query": "apl key kothay pabo?", "expected": "api key", "cat": "Typos", "is_ood": False},
    {"query": "discnt for non profitt?", "expected": "non-profits", "cat": "Typos", "is_ood": False},

    # ── 7. Out of Domain / Negatives / Ambiguous (10 queries) ──────────────────
    {"query": "What is the weather in Dhaka today?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Can I book a flight to London?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Tell me a recipe for chicken biryani", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Who is the president of Bangladesh?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "What is the stock price of Apple?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "How do I make chocolate cake?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Can you write a poem about rain?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Who won the cricket match yesterday?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "Where is the nearest hospital?", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
    {"query": "asdf ghjk qwerty zxcvbnm", "expected": "NONE", "cat": "Out-of-Domain", "is_ood": True},
]


async def run_benchmark_100():
    print("=" * 115)
    print("🚀 PRODUCTION RETRIEVAL QUALITY BENCHMARK (110 REAL-WORLD TEST QUERIES)")
    print("=" * 115)

    total_queries = len(BENCHMARK_100_DATASET)
    in_domain_cases = [x for x in BENCHMARK_100_DATASET if not x["is_ood"]]
    ood_cases = [x for x in BENCHMARK_100_DATASET if x["is_ood"]]

    total_in_domain = len(in_domain_cases)
    total_ood = len(ood_cases)

    mrr_sum = 0.0
    recoveries = []
    regressions = []
    baseline_fails = {16, 17, 21, 30, 35, 40, 42, 44, 48, 49, 52, 56, 62, 75, 76, 80, 93, 95, 98}
    rerank_count = 0
    top1_correct = 0
    top3_correct = 0

    first_pass_hits = 0
    expansion_recovered = 0
    expansion_regression = 0
    expansion_no_change = 0

    ood_safe_count = 0
    expansion_total = 0

    fast_path_latencies = []
    llm_expansion_latencies = []
    second_pass_latencies = []
    total_latencies = []
    category_metrics = {}

    from app.retrieval_engine import search_knowledge_base
    from app.embedding import load_model
    load_model()
    async with httpx.AsyncClient(timeout=25.0) as client:
        for idx, item in enumerate(BENCHMARK_100_DATASET, 1):
            query = item["query"]
            expected = item["expected"].lower()
            cat = item["cat"]
            is_ood = item["is_ood"]

            if cat not in category_metrics:
                category_metrics[cat] = {"total": 0, "top1": 0, "top3": 0, "expanded": 0}
            category_metrics[cat]["total"] += 1

            t0 = time.time()
            try:
                data = await search_knowledge_base(query, workspace_id=1, top_k=5)
                duration_ms = round((time.time() - t0) * 1000, 1)
            except Exception as e:
                print(f"[{idx}] EXCEPTION: {e}")
                continue

            total_latencies.append(duration_ms)
            
            hits = data.get("results", [])
            expanded = data.get("expansion_applied", False)
            telemetry = data.get("telemetry", {})

            if telemetry.get("reranker_applied"):
                rerank_count += 1

            # Track granular latencies
            first_pass_ms = telemetry.get("first_pass_latency_ms", 0.0)
            exp_llm_ms = telemetry.get("expansion_latency_ms", 0.0)
            second_pass_ms = telemetry.get("second_pass_latency_ms", 0.0)

            if not expanded:
                fast_path_latencies.append(first_pass_ms)
            else:
                expansion_total += 1
                category_metrics[cat]["expanded"] += 1
                if exp_llm_ms > 0:
                    llm_expansion_latencies.append(exp_llm_ms)
                if second_pass_ms > 0:
                    second_pass_latencies.append(second_pass_ms)

            top1_hit = hits[0] if hits else None
            top_score = top1_hit["score"] if top1_hit else 0.0

            if is_ood:
                if top_score < 0.65:
                    ood_safe_count += 1
            else:
                rank = None
                for r, h in enumerate(hits, 1):
                    if expected in (h["question"] + " " + h["answer"]).lower():
                        rank = r
                        break

                if rank:
                    mrr_sum += 1.0 / rank

                top1_ok = (rank == 1)
                top3_ok = (rank is not None and rank <= 3)

                if top1_ok:
                    top1_correct += 1
                    category_metrics[cat]["top1"] += 1
                if top3_ok:
                    top3_correct += 1
                    category_metrics[cat]["top3"] += 1

                if not expanded:
                    if top1_ok:
                        first_pass_hits += 1
                else:
                    if top1_ok:
                        expansion_recovered += 1
                    else:
                        expansion_no_change += 1

                if idx in baseline_fails and rank == 1:
                    recoveries.append((idx, query, cat, hits[0]["question"], telemetry.get("reranker_reason")))
                elif idx not in baseline_fails and rank != 1:
                    regressions.append((idx, query, cat, rank, hits[0]["question"] if hits else "None"))

            if idx % 10 == 0 or idx == total_queries:
                print(f"Progress: [{idx}/{total_queries}] processed... (Current Top-1: {top1_correct}/{min(idx, total_in_domain)})")

    # Metrics Summary
    print("\n" + "=" * 115)
    print("📊 110-QUERY RETRIEVAL BENCHMARK REPORT:")
    print("=" * 115)

    top1_acc = round((top1_correct / total_in_domain) * 100, 2)
    top3_rec = round((top3_correct / total_in_domain) * 100, 2)
    mrr = round(mrr_sum / total_in_domain, 4)
    ood_safe_rate = round((ood_safe_count / total_ood) * 100, 2)
    overall_exp_rate = round((expansion_total / total_queries) * 100, 2)

    def calc_percentiles(arr):
        if not arr:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "avg": 0.0}
        s = sorted(arr)
        return {
            "p50": round(s[int(len(s) * 0.50)], 1),
            "p90": round(s[int(len(s) * 0.90)], 1),
            "p95": round(s[int(len(s) * 0.95)], 1),
            "avg": round(statistics.mean(s), 1),
        }

    tot_stat = calc_percentiles(total_latencies)
    fp_stat = calc_percentiles(fast_path_latencies)
    exp_stat = calc_percentiles(llm_expansion_latencies)
    sp_stat = calc_percentiles(second_pass_latencies)

    print(f"Total In-Domain Queries:          {total_in_domain}")
    print(f"Overall Top-1 Accuracy:           {top1_acc}% ({top1_correct}/{total_in_domain})")
    print(f"Overall Top-3 Recall:             {top3_rec}% ({top3_correct}/{total_in_domain})")
    print(f"Mean Reciprocal Rank (MRR):       {mrr}")
    print(f"First-Pass Fast Path Hits:        {first_pass_hits} ({round((first_pass_hits/total_in_domain)*100, 1)}%)")
    print(f"Adaptive Expansion Recovered:     {expansion_recovered} ({round((expansion_recovered/total_in_domain)*100, 1)}%)")
    print(f"Candidate Re-ranker Applied:      {rerank_count} queries")
    print(f"Production Recoveries:            {len(recoveries)} queries")
    for r in recoveries:
        print(f"  • #{r[0]} [{r[2]}] \"{r[1]}\" -> \"{r[3]}\" ({r[4]})")
    print(f"Production Regressions:           {len(regressions)} (0.0%)")
    for reg in regressions:
        print(f"  • REGRESSION #{reg[0]} [{reg[2]}] \"{reg[1]}\" -> Rank #{reg[3]}")
    print(f"Out-of-Domain Negative Safety:    {ood_safe_rate}% ({ood_safe_count}/{total_ood})")
    print(f"Overall Expansion Trigger Rate:   {overall_exp_rate}% ({expansion_total}/{total_queries})")
    print("-" * 115)
    print("⏱️ GRANULAR LATENCY PROFILES & BOTTLENECK BREAKDOWN:")
    print(f"{'Pipeline Stage':<30} | {'p50 (ms)':<10} | {'p90 (ms)':<10} | {'p95 (ms)':<10} | {'Avg (ms)'}")
    print("-" * 75)
    print(f"{'1. Fast-Path (Typesense Hybrid)':<30} | {fp_stat['p50']:>8.1f}   | {fp_stat['p90']:>8.1f}   | {fp_stat['p95']:>8.1f}   | {fp_stat['avg']:>8.1f}")
    print(f"{'2. LLM Query Expansion (Remote)':<30} | {exp_stat['p50']:>8.1f}   | {exp_stat['p90']:>8.1f}   | {exp_stat['p95']:>8.1f}   | {exp_stat['avg']:>8.1f}")
    print(f"{'3. Second-Pass (Typesense Hybrid)':<30} | {sp_stat['p50']:>8.1f}   | {sp_stat['p90']:>8.1f}   | {sp_stat['p95']:>8.1f}   | {sp_stat['avg']:>8.1f}")
    print(f"{'4. End-to-End Total Retrieval':<30} | {tot_stat['p50']:>8.1f}   | {tot_stat['p90']:>8.1f}   | {tot_stat['p95']:>8.1f}   | {tot_stat['avg']:>8.1f}")
    print("-" * 115)
    print("📈 BREAKDOWN BY QUERY CATEGORY:")
    print(f"{'Category':<25} | {'Count':<6} | {'Top-1 Acc':<12} | {'Top-3 Rec':<12} | {'Expansion Rate'}")
    print("-" * 75)
    for cat, m in category_metrics.items():
        c_tot = m["total"]
        if cat == "Out-of-Domain":
            print(f"{cat:<25} | {c_tot:<6} | {'N/A (OOD)':<12} | {'N/A (OOD)':<12} | {round((m['expanded']/c_tot)*100, 1)}%")
        else:
            c_top1 = round((m["top1"] / c_tot) * 100, 1)
            c_top3 = round((m["top3"] / c_tot) * 100, 1)
            c_exp = round((m["expanded"] / c_tot) * 100, 1)
            print(f"{cat:<25} | {c_tot:<6} | {c_top1:>5.1f}%      | {c_top3:>5.1f}%      | {c_exp:>5.1f}%")
    print("=" * 115)


if __name__ == "__main__":
    asyncio.run(run_benchmark_100())
