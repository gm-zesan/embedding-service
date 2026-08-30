import asyncio
import io
import json
import sys
import time

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
    search_knowledge_base,
)

async def inspect_case():
    print("=================================================================================================")
    print("🔬 DEEP INSPECTION OF TIER 3 QUERY: \"How do you ensure our customer records remain unreadable?\"")
    print("=================================================================================================\n")

    load_model()
    client = get_typesense_client()

    query = "How do you ensure our customer records remain unreadable to unauthorized third parties?"
    clean_q = preprocess_query(query)

    print(f"Original Query: \"{query}\"")
    print(f"Target Ground Truth FAQ: \"How is my data encrypted?\"\n")

    # Step 1: Tier 1 Raw Search
    t0 = time.time()
    v1 = embed(clean_q)
    s1 = execute_hybrid_search(client, clean_q, v1, workspace_id=1, top_k=5)
    h1 = parse_typesense_hits(s1.get("hits", []))
    t1_ms = round((time.time() - t0) * 1000, 2)
    print("1️⃣  TIER 1 (Raw Dense Vector + Hybrid):")
    print(f"    - Latency: {t1_ms} ms")
    print(f"    - Top 3 Hits:")
    for i, h in enumerate(h1[:3], 1):
        print(f"       #{i}: \"{h['question']}\" | Score: {h['score']:.4f} | Match Type: {h['match_type']}")

    # Step 2: Tier 2 Local Expansion
    t0 = time.time()
    local_exp = expand_locally(clean_q)
    t2_applied = local_exp != clean_q
    print(f"\n2️⃣  TIER 2 (Local Domain Expansion):")
    print(f"    - Expanded Text: \"{local_exp}\"")
    print(f"    - Dictionary Triggered: {'YES' if t2_applied else 'NO (Query absent from local lexicon)'}")

    # Step 3: Tier 3 LLM Expansion
    t0 = time.time()
    llm_terms = None
    if config.LLM_EXPANSION_API_KEY:
        llm_terms = await expand_query_via_llm(clean_q)
    t3_llm_ms = round((time.time() - t0) * 1000, 2)
    print(f"\n3️⃣  TIER 3 (LLM Query Reformulation):")
    print(f"    - LLM Generated Keywords: \"{llm_terms}\"")
    print(f"    - LLM Reformulation Latency: {t3_llm_ms} ms")

    if llm_terms:
        exp_q = f"{clean_q} {llm_terms}"
        v3 = embed(exp_q)
        s3 = execute_hybrid_search(client, exp_q, v3, workspace_id=1, top_k=5)
        h3 = parse_typesense_hits(s3.get("hits", []))
        print(f"    - Search with Reformulated Keywords:")
        for i, h in enumerate(h3[:3], 1):
            print(f"       #{i}: \"{h['question']}\" | Score: {h['score']:.4f} | Match Type: {h['match_type']}")

    # Step 4: Answerability Gate Verification
    res = await search_knowledge_base(query, workspace_id=1, top_k=5)
    final_hits = res["results"]
    top_hit = final_hits[0] if final_hits else None
    final_score = top_hit["score"] if top_hit else 0.0

    print(f"\n4️⃣  ANSWERABILITY GATE DECISION:")
    print(f"    - Final Top Hit: \"{top_hit['question'] if top_hit else 'None'}\"")
    print(f"    - Final Score: {final_score:.4f}")
    print(f"    - Answerability Threshold: >= 0.45")
    gate_passed = final_score >= 0.45
    print(f"    - Gate Status: {'⚠️ PASSED (Grounded Candidate)' if gate_passed else '🛑 BLOCKED (< 0.45) ➜ Route safely diverted to Safe Fallback / Handoff'}")
    print(f"    - False Grounding Citation Occurred: {'YES ❌ (Dangerous)' if gate_passed and 'encrypted' not in top_hit['question'].lower() else 'NO ✅ (Answerability Gate successfully protected the user)'}")

if __name__ == "__main__":
    asyncio.run(inspect_case())
