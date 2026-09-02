import asyncio
import json
from app.retrieval_engine import expand_locally, rerank_candidate_hits
from app.linguistics import CanonicalConceptMapper, LanguageProfiler
from app.lexicon_repository import repository

async def main():
    # Force load global snapshot from JSON file (bypassing Laravel API)
    with open("snapshot.json", "r") as f:
        snapshot = json.load(f)
        
    repository.snapshots[0] = snapshot
    
    query = "koto din lagbe"
    clean_query = query
    print(f"Query: {query}")
    
    lang_profile = LanguageProfiler.profile(clean_query)
    
    telemetry = {
        "shadow_match_concept": None,
        "shadow_match_expansion": None,
        "shadow_match_rerank": None,
    }
    
    enrichment = CanonicalConceptMapper.map_concepts(clean_query, lang_profile, snapshot, telemetry)
    print(f"Shadow Concept Match: {telemetry['shadow_match_concept']}")
    
    norm_sig = enrichment.metadata.get("normalized_signal")
    local_expanded = expand_locally(clean_query, normalized_signal=norm_sig, contextual_signal=None, snapshot=snapshot, telemetry=telemetry)
    print(f"Shadow Expansion Match: {telemetry['shadow_match_expansion']}")
    print(f"Expansion Result: {local_expanded}")
    
    # Fake Hits
    fake_hits = [
        {"id": "1", "question": "koto din lagbe?", "answer": "10 days", "score": 0.9, "priority": 1, "document_type": "delivery_policy"},
        {"id": "2", "question": "ferot dibo kivabe?", "answer": "return it", "score": 0.88, "priority": 1, "document_type": "return_policy"},
    ]
    
    reranked, applied, reason = rerank_candidate_hits(clean_query, fake_hits, enrichment=enrichment, snapshot=snapshot, telemetry=telemetry)
    print(f"Shadow Rerank Match: {telemetry['shadow_match_rerank']}")

if __name__ == "__main__":
    asyncio.run(main())
