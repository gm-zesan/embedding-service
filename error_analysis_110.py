import asyncio
import io
import json
import sys
import time
from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEARCH_URL = "http://127.0.0.1:8001/api/v1/search"

# Import dataset from benchmark_100_queries
from benchmark_100_queries import BENCHMARK_100_DATASET


async def run_error_analysis():
    print("=" * 115)
    print("🔬 RETRIEVAL ERROR ANALYSIS & ROOT-CAUSE CLASSIFICATION (110 QUERIES)")
    print("=" * 115)

    failures = []
    successes = []

    async with httpx.AsyncClient(timeout=25.0) as client:
        for idx, item in enumerate(BENCHMARK_100_DATASET, 1):
            query = item["query"]
            expected = item["expected"].lower()
            cat = item["cat"]
            is_ood = item["is_ood"]

            if is_ood:
                continue  # OOD is evaluated separately for false positives

            try:
                resp = await client.post(SEARCH_URL, json={
                    "query": query,
                    "workspace_id": 1,
                    "top_k": 5,
                })
                data = resp.json()
            except Exception as e:
                print(f"[{idx}] Error: {e}")
                continue

            hits = data.get("results", [])
            expanded = data.get("expansion_applied", False)
            expanded_query = data.get("expanded_query")
            telemetry = data.get("telemetry", {})

            # Check matches in top 5
            correct_rank = None
            correct_hit = None
            for r_idx, h in enumerate(hits, 1):
                h_text = (h["question"] + " " + h["answer"]).lower()
                if expected in h_text:
                    correct_rank = r_idx
                    correct_hit = h
                    break

            top1_hit = hits[0] if hits else None
            top1_ok = (correct_rank == 1)

            if top1_ok:
                successes.append({
                    "idx": idx,
                    "query": query,
                    "category": cat,
                    "top1_question": top1_hit["question"],
                    "top1_score": top1_hit["score"],
                    "expanded": expanded,
                })
            else:
                # Classify Failure Mode
                if correct_rank is None:
                    fail_class = "A. Recall Failure (Not in Top-5)"
                elif correct_rank in (2, 3):
                    score_diff = round(top1_hit["score"] - correct_hit["score"], 4)
                    if score_diff < 0.05:
                        fail_class = "B1. Close Ranking Precision Failure (Δ < 0.05)"
                    else:
                        fail_class = "B2. Distant Ranking Precision Failure (Δ ≥ 0.05)"
                else:
                    fail_class = "B3. Deep Ranking Failure (Rank 4-5)"

                failures.append({
                    "idx": idx,
                    "query": query,
                    "category": cat,
                    "expected_keyword": expected,
                    "correct_rank": correct_rank,
                    "fail_class": fail_class,
                    "expansion_applied": expanded,
                    "expanded_query": expanded_query,
                    "top1_question": top1_hit["question"] if top1_hit else "None",
                    "top1_score": top1_hit["score"] if top1_hit else 0.0,
                    "top2_question": hits[1]["question"] if len(hits) > 1 else "None",
                    "top2_score": hits[1]["score"] if len(hits) > 1 else 0.0,
                    "top3_question": hits[2]["question"] if len(hits) > 2 else "None",
                    "top3_score": hits[2]["score"] if len(hits) > 2 else 0.0,
                    "correct_question": correct_hit["question"] if correct_hit else "Not Found",
                    "correct_score": correct_hit["score"] if correct_hit else 0.0,
                    "score_delta": round(top1_hit["score"] - correct_hit["score"], 4) if (top1_hit and correct_hit) else None,
                })

    print(f"\nTotal In-Domain Queries Evaluated: {len(successes) + len(failures)}")
    print(f"Top-1 Hits:                        {len(successes)} ({round((len(successes)/(len(successes)+len(failures)))*100, 1)}%)")
    print(f"Top-1 Failures:                    {len(failures)} ({round((len(failures)/(len(successes)+len(failures)))*100, 1)}%)")

    # Classification Summary
    class_counts = {}
    for f in failures:
        c = f["fail_class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    print("\n" + "=" * 115)
    print("📊 ROOT-CAUSE FAILURE CLASSIFICATION BREAKDOWN:")
    print("=" * 115)
    for c, cnt in sorted(class_counts.items()):
        print(f"  • {c:<55}: {cnt} cases ({round((cnt/len(failures))*100, 1)}% of all failures)")

    print("\n" + "=" * 115)
    print("📋 DETAILED FAILURE CASE-BY-CASE AUDIT TABLE:")
    print("=" * 115)
    for f in failures:
        print(f"\n[Case #{f['idx']}] Query: \"{f['query']}\" ({f['category']})")
        print(f"  ❌ Class:        {f['fail_class']}")
        print(f"  🎯 Correct FAQ:  \"{f['correct_question']}\" (Rank #{f['correct_rank']}, Score: {f['correct_score']})")
        print(f"  🥇 #1 Result:    \"{f['top1_question']}\" (Score: {f['top1_score']})")
        if f['score_delta'] is not None:
            print(f"  ⚖️ Score Delta:  Δ = {f['score_delta']}")
        print(f"  🥈 #2 Result:    \"{f['top2_question']}\" (Score: {f['top2_score']})")
        print(f"  🥉 #3 Result:    \"{f['top3_question']}\" (Score: {f['top3_score']})")
        print(f"  🔄 Expansion:    Triggered={f['expansion_applied']} | Expanded Query: \"{f['expanded_query']}\"")
        print("-" * 100)


if __name__ == "__main__":
    asyncio.run(run_error_analysis())
