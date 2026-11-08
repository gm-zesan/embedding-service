"""
Virtual Catalog Builder for Phase 3 Contract 3.0.0.
Builds DynamicSchemaCatalog from SQLite database files or registered data sources.
Provides deterministic, safe SQL identifiers for arbitrary multi-sheet spreadsheets
(handling Bangla, English, special characters, duplicate headers, and reserved words).
"""

from __future__ import annotations

import re
import os
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional

from app.analytics.contract_v3 import (
    DynamicSchemaCatalog,
    VirtualTableMeta,
    VirtualColumnMeta,
    VirtualColumnType,
)
from app.analytics.semantic_schema import SpreadsheetRelationshipGraph


class VirtualCatalogBuilder:
    """
    Constructs safe DynamicSchemaCatalog instances for VirtualRelationalCompiler.
    """

    RESERVED_SQLITE_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY", "ORDER",
        "LIMIT", "TABLE", "INDEX", "VIEW", "TRIGGER", "INSERT", "UPDATE",
        "DELETE", "DROP", "ALTER", "CREATE", "SET", "AND", "OR", "NOT",
        "IN", "IS", "NULL", "LIKE", "BETWEEN", "CASE", "WHEN", "THEN", "ELSE", "END",
    }

    @classmethod
    def sanitize_identifier(cls, raw: str, prefix: str = "col") -> str:
        """
        Converts arbitrary header string (including Bangla or punctuation)
        into a safe alphanumeric ASCII SQL identifier.
        """
        clean = re.sub(r"[^\w]", "_", str(raw).strip()).strip("_")
        # If contains non-ascii (e.g. Bangla) or is empty
        if not clean or not re.match(r"^[a-zA-Z_]", clean):
            # Generate deterministic hash suffix
            h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]
            clean = f"{prefix}_{h}"

        clean = re.sub(r"_+", "_", clean)
        if clean.upper() in cls.RESERVED_SQLITE_KEYWORDS:
            clean = f"{clean}_val"

        return clean[:64]

    @classmethod
    def build_from_sqlite(
        cls,
        db_path: str,
        workspace_id: int,
        dataset_id: str,
        generation_id: str,
        relationship_graph: Optional[SpreadsheetRelationshipGraph] = None,
        source_descriptors: Optional[Dict[str, Any]] = None,
    ) -> DynamicSchemaCatalog:
        """
        Inspects live SQLite tables and columns via PRAGMA table_info and builds DynamicSchemaCatalog.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database file not found: {db_path}")

        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
        cursor = conn.cursor()

        # List all user tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables_rows = cursor.fetchall()

        tables_meta: Dict[str, VirtualTableMeta] = {}
        fingerprint_parts: List[str] = []

        for (table_name,) in tables_rows:
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            col_rows = cursor.fetchall()

            columns_meta: Dict[str, VirtualColumnMeta] = {}
            pks: List[str] = []

            for cid, col_name, col_type, notnull, dflt_value, pk in col_rows:
                ptype = VirtualColumnType.NUMERIC if any(t in col_type.upper() for t in ("INT", "NUM", "REAL", "FLOAT", "DOUBLE")) else VirtualColumnType.TEXT
                is_pk = bool(pk)
                if is_pk:
                    pks.append(col_name)

                # Check if descriptor exists for this column
                desc = (source_descriptors or {}).get(col_name, {})
                orig_header = desc.get("original_header", col_name)
                role = desc.get("semantic_role", "attribute")
                sensitivity = desc.get("sensitivity", "restricted")

                meta = VirtualColumnMeta(
                    name=col_name,
                    original_header=orig_header,
                    physical_type=ptype,
                    semantic_role=role,
                    sensitivity=sensitivity,
                    is_primary_key=is_pk,
                    is_nullable=not bool(notnull),
                )
                columns_meta[col_name] = meta
                fingerprint_parts.append(f"{table_name}.{col_name}:{ptype.value}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
            row_count = cursor.fetchone()[0]

            tables_meta[table_name] = VirtualTableMeta(
                name=table_name,
                sheet_name=table_name,
                row_count=row_count,
                columns=columns_meta,
                primary_key=pks,
            )

        conn.close()

        schema_fingerprint = hashlib.sha256(";".join(sorted(fingerprint_parts)).encode("utf-8")).hexdigest()

        return DynamicSchemaCatalog(
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            generation_id=generation_id,
            schema_fingerprint=schema_fingerprint,
            tables=tables_meta,
            relationships=relationship_graph,
        )


virtual_catalog_builder = VirtualCatalogBuilder()

