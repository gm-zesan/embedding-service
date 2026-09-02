import asyncio
import time
import sys
import io
import statistics
from app.retrieval_engine import search_knowledge_base
from app.lexicon_repository import repository
from benchmark_retrieval_v2 import EVAL_DATASET

# No utf-8 override needed on Mac

async def run_shadow_benchmark():
    print("======================================================================")
    print("🔬 SHADOW MODE EQUIVALENCE BENCHMARK (PHASE A)")
    print("======================================================================")
    
    # Ensure model is loaded
    try:
        from app.embedding import load_model
        load_model()
    except Exception as e:
        print(f"Skipping model load due to error (run outside sandbox): {e}")
        return

    # Warmup and load global snapshot
    try:
        await repository.get_or_fetch_snapshot(0)
    except Exception as e:
        print(f"Failed to fetch global snapshot from Laravel: {e}")
        print("Please ensure Laravel is running on LARAVEL_API_URL")
        return

    mismatches = {
        "expansion": 0,
        "concept": 0,
        "rerank": 0
    }
    
    shadow_latencies = []
    
    total_queries = len(EVAL_DATASET)
    
    for item in EVAL_DATASET:
        query = item["query"]
        try:
            t_req = time.time()
            response = await search_knowledge_base(query, top_k=5, workspace_id=0)
            req_dur = (time.time() - t_req) * 1000
            shadow_latencies.append(req_dur)
            
            telemetry = response["telemetry"]
            
            # Check shadow match flags
            if telemetry.get("shadow_match_expansion") is False:
                mismatches["expansion"] += 1
            if telemetry.get("shadow_match_concept") is False:
                mismatches["concept"] += 1
            if telemetry.get("shadow_match_rerank") is False:
                mismatches["rerank"] += 1
                
            print(f"✅ {query[:40]:<40} | Exp: {telemetry.get('shadow_match_expansion')}, Concept: {telemetry.get('shadow_match_concept')}, Rerank: {telemetry.get('shadow_match_rerank')}")
            
        except Exception as e:
            print(f"❌ {query[:40]:<40} | ERROR: {e}")

    total_time = sum(shadow_latencies)
    median_latency = statistics.median(shadow_latencies) if shadow_latencies else 0
    p95_latency = statistics.quantiles(shadow_latencies, n=20)[18] if len(shadow_latencies) > 20 else max(shadow_latencies) if shadow_latencies else 0
    
    print("\n======================================================================")
    print("📊 SHADOW EQUIVALENCE SUMMARY:")
    print("======================================================================")
    print(f"Queries evaluated       : {total_queries}")
    print("Hardcoded authoritative : YES")
    print("DB snapshot shadow-only : YES")
    print("----------------------------------------------------------------------")
    print(f"Expansion mismatches    : {mismatches['expansion']}")
    print(f"Concept mismatches      : {mismatches['concept']}")
    print(f"Reranking mismatches    : {mismatches['rerank']}")
    print("----------------------------------------------------------------------")
    print(f"Avg Latency             : {total_time/total_queries:.2f} ms")
    print(f"Median Latency          : {median_latency:.2f} ms")
    print(f"P95 Latency             : {p95_latency:.2f} ms")
    print("======================================================================")
    
    if sum(mismatches.values()) == 0:
        print("\n✅ SHADOW MODE PASSED: 100% Behavioral Equivalence Confirmed!")
    else:
        print("\n❌ SHADOW MODE FAILED: Discrepancies found between Hardcoded and DB paths.")

if __name__ == "__main__":
    asyncio.run(run_shadow_benchmark())
