"""
Workspace-Scoped Data Source Registry.
Manages metadata, persistence, and isolation for uploaded analytical data sources.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from .data_source import DataSource, Dataset, DataSourceFormat

logger = logging.getLogger(__name__)


class DataSourceRegistry:
    """
    Registry for managing and discovering analytical data sources strictly isolated by workspace_id.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            self.storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "storage",
                "excel_databases"
            )
        else:
            self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_metadata_path(self, workspace_id: int, source_id: str) -> str:
        return os.path.join(self.storage_dir, f"ws_{workspace_id}_{source_id}.json")

    def register_source(self, source: DataSource) -> DataSource:
        """Persists source metadata to workspace-isolated storage."""
        meta_path = self._get_metadata_path(source.workspace_id, source.source_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(source.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"[DataSourceRegistry] Registered source {source.source_id} for workspace {source.workspace_id}")
        return source

    def get_source(self, workspace_id: int, source_id: str) -> Optional[DataSource]:
        """Retrieves a source ensuring strict workspace ownership."""
        if not source_id:
            return None
        meta_path = self._get_metadata_path(workspace_id, source_id)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self._parse_source_dict(data, workspace_id, source_id)
            except Exception as e:
                logger.error(f"[DataSourceRegistry] Error reading metadata for {source_id}: {e}")
                return None

        # Check for legacy hash descriptor
        for fname in os.listdir(self.storage_dir):
            if fname.startswith(f"ws_{workspace_id}_") and fname.endswith(".json"):
                if source_id in fname:
                    try:
                        with open(os.path.join(self.storage_dir, fname), "r", encoding="utf-8") as f:
                            data = json.load(f)
                        return self._parse_source_dict(data, workspace_id, source_id)
                    except Exception:
                        pass
        return None

    def list_sources(self, workspace_id: int) -> List[DataSource]:
        """Lists all analytical data sources belonging to the given workspace_id."""
        sources: List[DataSource] = []
        if not os.path.exists(self.storage_dir):
            return sources

        for fname in os.listdir(self.storage_dir):
            if fname.startswith(f"ws_{workspace_id}_") and fname.endswith(".json"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    source_id = data.get("source_id") or data.get("file_id") or fname.replace(f"ws_{workspace_id}_", "").replace(".json", "")
                    src = self._parse_source_dict(data, workspace_id, source_id)
                    if src:
                        sources.append(src)
                except Exception as e:
                    logger.warning(f"[DataSourceRegistry] Skipping unreadable descriptor {fname}: {e}")

        # Sort by uploaded_at descending
        sources.sort(key=lambda s: s.uploaded_at, reverse=True)
        return sources

    def find_source_by_filename(self, workspace_id: int, filename: str) -> Optional[DataSource]:
        """Finds a source by exact or stem filename within a workspace."""
        clean_target = filename.strip().lower()
        target_stem = os.path.splitext(clean_target)[0]

        sources = self.list_sources(workspace_id)
        # 1. Exact match
        for s in sources:
            if s.original_filename.lower() == clean_target:
                return s
        # 2. Stem match
        for s in sources:
            src_stem = os.path.splitext(s.original_filename.lower())[0]
            if src_stem == target_stem or target_stem in src_stem or src_stem in target_stem:
                return s
        return None

    def delete_source(self, workspace_id: int, source_id: str) -> bool:
        """Deletes metadata and physical SQLite file for a source with workspace security."""
        src = self.get_source(workspace_id, source_id)
        if not src:
            return False

        meta_path = self._get_metadata_path(workspace_id, source_id)
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass

        if src.db_path and os.path.exists(src.db_path):
            try:
                os.remove(src.db_path)
            except OSError:
                pass

        return True

    def _parse_source_dict(self, data: Dict[str, Any], workspace_id: int, fallback_source_id: str) -> Optional[DataSource]:
        """Helper to parse legacy or modern metadata dictionaries into DataSource."""
        if int(data.get("workspace_id", workspace_id)) != workspace_id:
            return None

        # Build Datasets
        datasets: List[Dataset] = []
        if "datasets" in data and isinstance(data["datasets"], list):
            for d in data["datasets"]:
                datasets.append(Dataset(**d))
        elif "tables" in data and isinstance(data["tables"], dict):
            # Legacy format conversion
            for tbl_name, tbl_meta in data["tables"].items():
                datasets.append(Dataset(
                    dataset_id=tbl_name,
                    dataset_name=tbl_name.replace("_", " ").title(),
                    sheet_name=tbl_meta.get("original_sheet", tbl_name),
                    columns=tbl_meta.get("columns", []),
                    numeric_measures=tbl_meta.get("numeric_measures", []),
                    text_dimensions=tbl_meta.get("text_dimensions", []),
                    date_fields=[],
                    row_count=tbl_meta.get("row_count", 0),
                    storage_reference=tbl_name,
                ))

        fmt_str = data.get("format", "xlsx").lower()
        if fmt_str not in ("xlsx", "xls", "csv", "pdf"):
            fmt_str = "xlsx"

        return DataSource(
            workspace_id=workspace_id,
            source_id=data.get("source_id") or data.get("file_id") or fallback_source_id,
            original_filename=data.get("original_filename") or data.get("filename") or "uploaded_data.xlsx",
            format=DataSourceFormat(fmt_str),
            stored_path=data.get("stored_path", ""),
            uploaded_at=data.get("uploaded_at", ""),
            file_hash=data.get("file_hash", fallback_source_id),
            datasets=datasets,
            schema_summary=data.get("schema_summary", ""),
            total_rows=data.get("total_rows", 0),
            db_path=data.get("db_path"),
        )


# Global Singleton Registry
source_registry = DataSourceRegistry()
