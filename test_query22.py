import asyncio
from app.retrieval_engine import search_knowledge_base
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/Users/zesan/.cache/huggingface"

async def main():
    hits = await search_knowledge_base("Is our customer data secure and stored safely?", top_k=3, workspace_id=0)
    for i, h in enumerate(hits['results']):
        print(f"Rank {i+1}: {h['question']} (score: {h['score']}, match: {h['match_type']}, id: {h['id']})")
    print(f"telemetry: {hits['telemetry']}")

asyncio.run(main())
