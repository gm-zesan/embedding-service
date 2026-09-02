import asyncio
import time
import json
import logging
from app.lexicon_repository import repository
from app.retrieval_engine import search_knowledge_base
# pyrefly: ignore [missing-import]
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safety_test")

async def test_workspace_isolation():
    logger.info("=== 1. Workspace Isolation ===")
    
    # Fake global snapshot
    global_snap = {
        "domain_entries": {"global_concept": [{"pattern": "global_term", "expansion": "global_exp"}]},
        "concept_patterns": {},
        "action_mappings": {},
        "policy_mappings": {}
    }
    # Fake WS 1
    ws1_snap = {
        "domain_entries": {"ws1_concept": [{"pattern": "ws1_term", "expansion": "ws1_exp"}]},
        "concept_patterns": {}, "action_mappings": {}, "policy_mappings": {}
    }
    
    repository.snapshots[0] = global_snap
    repository.snapshots[1] = ws1_snap
    
    s1 = await repository.get_or_fetch_snapshot(1)
    s999 = await repository.get_or_fetch_snapshot(999)
    
    assert "ws1_concept" in s1["domain_entries"], "WS1 must have WS1 terms"
    assert "ws1_concept" not in s999["domain_entries"], "WS999 must NOT have WS1 terms"
    assert s999 is global_snap, "WS999 must fallback to global"
    logger.info("Workspace isolation PASSED")

async def test_atomic_reload_failure():
    logger.info("=== 2. Atomic Reload Failure ===")
    old_snap = repository.snapshots[0]
    
    # Mock HTTP failure for workspace 2
    try:
        await repository.fetch_and_reload(2)
    except Exception:
        pass
        
    assert 2 not in repository.snapshots, "Failed reload must not create active snapshot"
    assert repository.snapshots[0] is old_snap, "Global must remain untouched"
    
    # Mock invalid snapshot
    class FakeResponse:
        def json(self): return {"invalid": "missing_keys"}
        def raise_for_status(self): pass
    
    import httpx
    original_get = httpx.AsyncClient.get
    
    async def mock_get(*args, **kwargs):
        return FakeResponse()
        
    httpx.AsyncClient.get = mock_get
    
    try:
        await repository.fetch_and_reload(0)
    except Exception as e:
        logger.info(f"Caught expected validation error: {e}")
        
    assert repository.snapshots[0] is old_snap, "Invalid reload must leave old snapshot untouched"
    httpx.AsyncClient.get = original_get
    logger.info("Atomic reload failure PASSED")

async def test_performance_overhead():
    logger.info("=== 4. Performance Overhead ===")
    # Load actual snapshot for realistic test (dummy for sandbox)
    real_snap = {
        "domain_entries": {"koto_din": [{"pattern": "koto din lagbe return korte", "expansion": "return timeframe 7 days"}]},
        "concept_patterns": {"RETURN_POLICY": {"target_doc_type": "return_policy", "phrases": ["return koto din"]}},
        "action_mappings": {"view": {"actions": ["view"], "target_phrase": "how", "penalty_phrase": "none"}},
        "policy_mappings": {}
    }
    
    repository.snapshots[0] = real_snap
    
    # Warmup
    await search_knowledge_base("how to return")
    
    # Measure with shadow
    t0 = time.time()
    for _ in range(50):
        await search_knowledge_base("koto din lagbe return korte?")
    shadow_dur = (time.time() - t0) / 50 * 1000
    
    # Measure without shadow (remove snapshot)
    del repository.snapshots[0]
    repository.get_or_fetch_snapshot = lambda *args, **kwargs: asyncio.sleep(0) # Mock no DB
    
    t1 = time.time()
    for _ in range(50):
        await search_knowledge_base("koto din lagbe return korte?", workspace_id=99)
    no_shadow_dur = (time.time() - t1) / 50 * 1000
    
    logger.info(f"Avg latency WITHOUT shadow: {no_shadow_dur:.2f} ms")
    logger.info(f"Avg latency WITH shadow: {shadow_dur:.2f} ms")
    logger.info(f"Shadow overhead: {shadow_dur - no_shadow_dur:.2f} ms")

async def test_failure_path():
    logger.info("=== 5. Failure-Path Behavior ===")
    # If fetch fails, retrieval must gracefully use old/global snapshot or nothing, and NOT crash
    repository.snapshots.clear()
    
    import httpx
    original_get = httpx.AsyncClient.get
    async def mock_fail_get(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")
    httpx.AsyncClient.get = mock_fail_get
    
    res = await search_knowledge_base("hello", workspace_id=1)
    assert res is not None, "Retrieval must not crash on DB failure"
    assert res["query"] == "hello", "Retrieval must return results normally using hardcoded"
    
    httpx.AsyncClient.get = original_get
    logger.info("Failure-path behavior PASSED")


async def main():
    await test_workspace_isolation()
    await test_atomic_reload_failure()
    await test_performance_overhead()
    await test_failure_path()
    
if __name__ == "__main__":
    asyncio.run(main())
