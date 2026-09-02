import asyncio
from app.lexicon_repository import repository
import time
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

async def run_verifications():
    print("======================================================================")
    print("🧪 SYSTEM PROPERTIES VERIFICATION SUITE")
    print("======================================================================")
    
    # 1. Global Fallback
    print("\n[1] Validating Global Fallback (workspace_id = None -> 0)...")
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        # Mock success for workspace 0
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "workspace_id": 0,
            "concept_patterns": {"TEST": {"target_doc_type": "faq", "positive_phrases": ["test"]}},
            "domain_entries": {},
            "action_mappings": {},
            "policy_mappings": {}
        }
        mock_get.return_value = mock_resp
        
        # Clear snapshots
        repository.snapshots = {}
        
        snap = await repository.get_or_fetch_snapshot(None)
        assert snap["workspace_id"] == 0, "Fallback failed to resolve workspace_id to 0"
        print("✅ Global fallback to workspace 0 successful.")

    # 2. Cross-Tenant Leakage & Workspace Isolation
    print("\n[2] Validating Workspace Isolation & No Cross-Tenant Leakage...")
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_resp_w1 = MagicMock()
        mock_resp_w1.raise_for_status = lambda: None
        mock_resp_w1.json.return_value = {
            "workspace_id": 1,
            "concept_patterns": {"W1_CONCEPT": {"target_doc_type": "faq", "positive_phrases": ["test"]}},
            "domain_entries": {},
            "action_mappings": {},
            "policy_mappings": {}
        }
        mock_get.return_value = mock_resp_w1
        
        snap_w1 = await repository.get_or_fetch_snapshot(1)
        assert snap_w1["workspace_id"] == 1
        assert "W1_CONCEPT" in snap_w1["concept_patterns"]
        
        # Now mock workspace 2
        mock_resp_w2 = MagicMock()
        mock_resp_w2.raise_for_status = lambda: None
        mock_resp_w2.json.return_value = {
            "workspace_id": 2,
            "concept_patterns": {"W2_CONCEPT": {"target_doc_type": "faq", "positive_phrases": ["test"]}},
            "domain_entries": {},
            "action_mappings": {},
            "policy_mappings": {}
        }
        mock_get.return_value = mock_resp_w2
        
        snap_w2 = await repository.get_or_fetch_snapshot(2)
        assert snap_w2["workspace_id"] == 2
        assert "W2_CONCEPT" in snap_w2["concept_patterns"]
        
        # Verify isolation
        assert "W1_CONCEPT" not in snap_w2["concept_patterns"], "Cross-tenant leakage detected: W1 leaked into W2!"
        assert "W2_CONCEPT" not in snap_w1["concept_patterns"], "Cross-tenant leakage detected: W2 leaked into W1!"
        print("✅ Workspace isolation strictly maintained. Zero cross-tenant leakage.")

    # 3. Atomic Reload/Swap Behavior
    print("\n[3] Validating Atomic Reload/Swap Behavior...")
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        # Mock slow fetch
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(0.5)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = lambda: None
            mock_resp.json.return_value = {
                "workspace_id": 1,
                "concept_patterns": {"NEW_W1_CONCEPT": {"target_doc_type": "faq", "positive_phrases": ["test"]}},
                "domain_entries": {},
                "action_mappings": {},
                "policy_mappings": {}
            }
            return mock_resp
            
        mock_get.side_effect = slow_fetch
        
        # Active snapshot is still snap_w1 ("W1_CONCEPT")
        assert "W1_CONCEPT" in repository.snapshots[1]["concept_patterns"]
        
        # Start background reload
        reload_task = asyncio.create_task(repository.fetch_and_reload(1))
        
        # Concurrently read while reloading (should NOT block, should get OLD data)
        await asyncio.sleep(0.1)
        active_snap = repository.get_snapshot(1)
        assert "W1_CONCEPT" in active_snap["concept_patterns"]
        assert "NEW_W1_CONCEPT" not in active_snap["concept_patterns"]
        print("✅ Readers are NOT blocked during background fetch.")
        
        # Wait for reload to finish
        await reload_task
        
        # Now read again
        active_snap_new = repository.get_snapshot(1)
        assert "NEW_W1_CONCEPT" in active_snap_new["concept_patterns"]
        assert "W1_CONCEPT" not in active_snap_new["concept_patterns"]
        print("✅ Atomic pointer swap completed cleanly.")

    print("\n======================================================================")
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == "__main__":
    asyncio.run(run_verifications())
