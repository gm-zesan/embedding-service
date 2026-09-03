import logging
import httpx
from typing import Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)

class LexiconRepository:
    """
    Manages in-memory snapshots of the DB-driven lexicon configuration.
    Provides atomic swaps and strict validation to ensure zero runtime DB hits.
    
    The snapshots are keyed by workspace_id. workspace_id=0 is the global fallback.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LexiconRepository, cls).__new__(cls)
                cls._instance._init()
            return cls._instance
            
    def _init(self):
        # dict[workspace_id -> dict]
        # Stores the validated snapshots
        self.snapshots: Dict[int, Dict[str, Any]] = {}
        # Protects the dictionary during atomic swaps
        self.swap_lock = threading.Lock()
        # The base URL of the Laravel application to fetch snapshots from
        # In a real app this would be injected via config
        from app import config
        # Assuming we can get LARAVEL_API_URL or APP_URL from config, defaulting to 127.0.0.1:8000
        self.laravel_url = getattr(config, "LARAVEL_API_URL", "http://127.0.0.1:8000")

    def get_snapshot(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve the currently active snapshot for a workspace. Non-blocking."""
        return self.snapshots.get(workspace_id)

    async def get_or_fetch_snapshot(self, workspace_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the snapshot for the workspace. If not loaded, fetches it atomically.
        If workspace_id is None, it defaults to 0.
        If fetching fails, attempts to fallback to 0.
        """
        wid = 0 if workspace_id is None else workspace_id
        
        # Check if already loaded
        snap = self.get_snapshot(wid)
        if snap is not None:
            return snap
            
        # Fetch if missing
        try:
            snap = await self.fetch_and_reload(wid)
            return snap
        except Exception as e:
            logger.warning("Failed to lazy-load snapshot for workspace %s: %s", wid, e)
            # If not global, fallback to global
            if wid != 0:
                snap_global = self.get_snapshot(0)
                if snap_global is not None:
                    return snap_global
                # Try to fetch global if missing
                try:
                    return await self.fetch_and_reload(0)
                except Exception:
                    pass
        return None

    def validate_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Enforce strict invariants before allowing a snapshot to go live.
        Raises AssertionError if any constraint is violated.
        """
        assert "concept_patterns" in snapshot, "Missing concept_patterns in snapshot"
        assert "domain_entries" in snapshot, "Missing domain_entries in snapshot"
        assert "action_mappings" in snapshot, "Missing action_mappings in snapshot"
        assert "policy_mappings" in snapshot, "Missing policy_mappings in snapshot"
        
        for concept_key, data in snapshot["concept_patterns"].items():
            if concept_key == "MULTI_ENTITY_DETECTION":
                continue  # no meta required for detection-only concepts
            
            assert data.get("target_doc_type") is not None, f"Missing CONCEPT_META for {concept_key}"
            assert len(data.get("positive_phrases", [])) > 0, f"No POSITIVE phrases for {concept_key}"

    def load_snapshot(self, workspace_id: int, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and atomically swaps a provided snapshot into memory.
        Does not require external HTTP fetch.
        """
        self.validate_snapshot(snapshot)
        with self.swap_lock:
            self.snapshots[workspace_id] = snapshot

        logger.info(
            "Successfully loaded lexicon snapshot for workspace %s (Version: %s)",
            workspace_id,
            snapshot.get("snapshot_version", "unknown")
        )
        return snapshot

    async def fetch_and_reload(self, workspace_id: int) -> Dict[str, Any]:
        """
        Fetches the latest snapshot for a workspace from the Laravel API,
        validates it, and atomically swaps it into memory.
        """
        candidate_urls = [
            self.laravel_url,
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8002",
        ]
        # Deduplicate preserving order
        candidate_urls = list(dict.fromkeys(candidate_urls))

        snapshot = None
        last_error = None

        for base_url in candidate_urls:
            endpoint = f"{base_url.rstrip('/')}/api/v1/internal/lexicon/snapshot?workspace_id={workspace_id}"
            try:
                async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        snapshot = response.json()
                        break
                    else:
                        last_error = f"HTTP {response.status_code} from {endpoint}"
            except Exception as e:
                last_error = str(e) or type(e).__name__
                continue

        if snapshot is None:
            logger.error("Failed to fetch lexicon snapshot for workspace %s: %s", workspace_id, last_error)
            raise RuntimeError(f"Could not reach Laravel API to fetch snapshot: {last_error}")

        return self.load_snapshot(workspace_id, snapshot)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current loaded versions and sizes for all workspaces."""
        status = {}
        for ws_id, snap in self.snapshots.items():
            status[ws_id] = {
                "snapshot_version": snap.get("snapshot_version"),
                "global_version": snap.get("global_version"),
                "workspace_version": snap.get("workspace_version"),
                "entry_counts": {
                    "domain_entries": len(snap.get("domain_entries", {})),
                    "concept_patterns": len(snap.get("concept_patterns", {})),
                    "action_mappings": len(snap.get("action_mappings", {})),
                    "policy_mappings": len(snap.get("policy_mappings", {})),
                }
            }
        return status

# Singleton instance
repository = LexiconRepository()
