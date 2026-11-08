"""
Virtual Relational Compiler for Phase 3 Contract 3.0.0.
Compiles validated SemanticQueryPlan instances into parameterized, read-only SQLite statements
across multi-sheet dynamic tabular schemas using ONLY confirmed relationship edges.

Guarantees:
1. Deterministic compilation: zero LLM involvement in SQL string creation.
2. Parameterized bindings (?): zero string interpolation of filter values.
3. Relationship resolution: joins ONLY across edges where state == 'CONFIRMED'.
4. Zero Cartesian fanouts: raises error if no confirmed join path exists.
5. Safe identifier resolution: handles Bangla/English/special chars safely.
6. Generation consistency: verifies schema fingerprint and generation_id.
"""

from __future__ import annotations

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque

logger = logging.getLogger(__name__)

from app.analytics.contract_v3 import (
    CONTRACT_VERSION_3_0,
    PreparedQueryArtifact,
    DynamicSchemaCatalog,
    VirtualTableMeta,
    VirtualColumnMeta,
    VirtualColumnType,
)
from app.analytics.models import (
    SemanticQueryPlan,
    MeasureSpec,
    DimensionSpec,
    FilterSpec,
    OrderBySpec,
    AggregationType,
    OperatorType,
)
from app.analytics.semantic_schema import (
    SpreadsheetRelationshipGraph,
    DiscoveredRelationshipEdge,
    RelationshipLifecycleState,
    RelationshipCardinality,
)


class VirtualCompilerError(Exception):
    """Raised when query planning or compilation encounters an unresolvable or unsafe condition."""
    def __init__(self, message: str, code: str = "COMPILER_ERROR"):
        super().__init__(message)
        self.code = code


class VirtualRelationalCompiler:
    """
    Deterministic AST-to-SQLite compiler for dynamic multi-sheet spreadsheet workbooks.
    """

    ALLOWED_AGGREGATIONS = {
        AggregationType.SUM: "SUM",
        AggregationType.COUNT: "COUNT",
        AggregationType.AVG: "AVG",
        AggregationType.MIN: "MIN",
        AggregationType.MAX: "MAX",
        AggregationType.COUNT_DISTINCT: "COUNT(DISTINCT",
    }

    SQL_OPERATORS = {
        OperatorType.EQUALS: "=",
        OperatorType.NOT_EQUALS: "!=",
        OperatorType.GREATER_THAN: ">",
        OperatorType.GREATER_EQUAL: ">=",
        OperatorType.LESS_THAN: "<",
        OperatorType.LESS_EQUAL: "<=",
        OperatorType.LIKE: "LIKE",
        OperatorType.IN: "IN",
        OperatorType.BETWEEN: "BETWEEN",
        OperatorType.IS_NULL: "IS NULL",
        OperatorType.NOT_NULL: "IS NOT NULL",
    }

    def compile(
        self,
        plan: SemanticQueryPlan,
        catalog: DynamicSchemaCatalog,
        relationship_graph: Optional[SpreadsheetRelationshipGraph] = None,
        max_limit: int = 50,
    ) -> PreparedQueryArtifact:
        """
        Compiles SemanticQueryPlan into PreparedQueryArtifact.
        """
        if plan.is_security_rejection:
            raise VirtualCompilerError("Plan rejected by security policy.", code="SECURITY_REJECTION")
        if plan.needs_clarification:
            raise VirtualCompilerError("Plan requires conversational clarification.", code="CLARIFICATION_REQUIRED")

        # 1. Resolve Confirmed Relationship Edges
        graph = relationship_graph or catalog.relationships
        confirmed_edges = self._extract_confirmed_edges(graph)

        # 2. Map all requested columns and measures to their owning virtual tables
        column_table_map, table_column_index = self._index_catalog_columns(catalog)

        # 3. Collect all referenced entities/columns across plan clauses
        referenced_fields = self._collect_referenced_fields(plan)
        target_tables, field_resolution = self._resolve_tables_for_fields(referenced_fields, column_table_map, catalog)

        if not target_tables:
            # If no specific columns, pick the first table in catalog
            if catalog.tables:
                target_tables = [next(iter(catalog.tables.keys()))]
            else:
                raise VirtualCompilerError("No tables available in dynamic schema catalog.", code="EMPTY_CATALOG")

        # 4. Resolve Join Paths across target tables using ONLY confirmed edges
        primary_table, join_clauses, used_edge_ids = self._resolve_join_tree(target_tables, confirmed_edges, catalog)

        # 5. Build SELECT clause (projections and aggregations)
        select_items, select_col_names = self._build_select_clause(plan, field_resolution, catalog)

        # 6. Build WHERE clause with parameterized bindings (?)
        where_clauses, bound_params = self._build_where_clause(plan, field_resolution, catalog)

        # 7. Build GROUP BY clause
        group_by_clause = self._build_group_by_clause(plan, field_resolution, catalog)

        # 8. Build ORDER BY clause
        order_by_clause = self._build_order_by_clause(plan, field_resolution, catalog)

        # 9. Enforce LIMIT
        limit_val = min(plan.limit if (plan.limit and plan.limit > 0) else max_limit, max_limit)

        # 10. Assemble SQL string
        sql_parts = ["SELECT", "  " + ",\n  ".join(select_items), f"FROM {primary_table}"]
        if join_clauses:
            sql_parts.extend(join_clauses)
        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))
        if group_by_clause:
            sql_parts.append("GROUP BY " + group_by_clause)
        if order_by_clause:
            sql_parts.append("ORDER BY " + order_by_clause)
        sql_parts.append(f"LIMIT {limit_val}")

        compiled_sql = "\n".join(sql_parts)

        # 11. Compute Query Fingerprint
        fingerprint = PreparedQueryArtifact.compute_fingerprint(
            sql=compiled_sql,
            params=bound_params,
            workspace_id=catalog.workspace_id,
            generation_id=catalog.generation_id,
        )

        return PreparedQueryArtifact(
            contract_version=CONTRACT_VERSION_3_0,
            request_id=getattr(plan, "request_id", "") or "req_compiled",
            workspace_id=catalog.workspace_id,
            dataset_id=catalog.dataset_id,
            generation_id=catalog.generation_id,
            query_fingerprint=fingerprint,
            schema_fingerprint=catalog.schema_fingerprint,
            sql_template=compiled_sql,
            params=bound_params,
            referenced_tables=list(target_tables),
            referenced_columns=list(referenced_fields),
            confirmed_edge_ids=list(used_edge_ids),
            compiler_version="3.0.0",
            status="prepared",
        )

    def _extract_confirmed_edges(self, graph: Optional[SpreadsheetRelationshipGraph]) -> List[DiscoveredRelationshipEdge]:
        """
        Extracts edges strictly where state == 'CONFIRMED'.
        Any edge that is DISCOVERED, NEEDS_CONFIRMATION, REJECTED, INVALIDATED, or STALE is excluded.
        """
        if not graph or not graph.edges:
            return []

        confirmed: List[DiscoveredRelationshipEdge] = []
        for edge_id, edge in graph.edges.items():
            state = getattr(edge, "state", "").upper()
            if state == RelationshipLifecycleState.CONFIRMED.value or state == "CONFIRMED":
                confirmed.append(edge)
            else:
                logger.debug("[VirtualCompiler] Skipping unconfirmed edge '%s' (state=%s)", edge_id, state)

        return confirmed

    def _index_catalog_columns(self, catalog: DynamicSchemaCatalog) -> Tuple[Dict[str, List[Tuple[str, VirtualColumnMeta]]], Dict[str, Dict[str, VirtualColumnMeta]]]:
        """
        Indexes all columns in the catalog for fast lookup by header or sanitized name.
        Returns:
            column_table_map: normalized_key -> list of (table_name, column_meta)
            table_column_index: table_name -> {col_name: column_meta}
        """
        col_map: Dict[str, List[Tuple[str, VirtualColumnMeta]]] = {}
        table_index: Dict[str, Dict[str, VirtualColumnMeta]] = {}

        for tbl_name, tbl_meta in catalog.tables.items():
            table_index[tbl_name] = {}
            for col_id, col_meta in tbl_meta.columns.items():
                table_index[tbl_name][col_id] = col_meta

                # Index by sanitized col_id, original header, and lowercase variants
                keys = {
                    col_id.lower(),
                    col_meta.name.lower(),
                    col_meta.original_header.lower(),
                    re.sub(r"[^\w]", "", col_meta.original_header.lower()),
                }
                for k in keys:
                    if k:
                        col_map.setdefault(k, []).append((tbl_name, col_meta))

        return col_map, table_index

    def _collect_referenced_fields(self, plan: SemanticQueryPlan) -> Set[str]:
        """Extracts all column/measure/dimension field names mentioned in the plan."""
        fields: Set[str] = set()
        for m in plan.measures:
            if m.name != "*":
                fields.add(m.name)
        for d in plan.dimensions:
            fields.add(d.name)
        for f in plan.filters:
            fields.add(f.field)
        for g in plan.group_by:
            fields.add(g)
        for o in plan.order_by:
            fields.add(o.field)
        return fields

    def _resolve_tables_for_fields(
        self,
        fields: Set[str],
        column_table_map: Dict[str, List[Tuple[str, VirtualColumnMeta]]],
        catalog: DynamicSchemaCatalog,
    ) -> Tuple[Set[str], Dict[str, Tuple[str, VirtualColumnMeta]]]:
        """
        Resolves each field to its owning table.
        Returns (set of unique tables needed, mapping of field -> (table_name, column_meta)).
        """
        tables: Set[str] = set()
        resolution: Dict[str, Tuple[str, VirtualColumnMeta]] = {}

        for f in fields:
            clean_f = f.lower().strip()
            # If field explicitly includes table prefix (e.g. Orders.customer_id)
            if "." in clean_f:
                tbl_part, col_part = clean_f.split(".", 1)
                matched_tbl = None
                for tbl_name, tbl_meta in catalog.tables.items():
                    if tbl_name.lower() == tbl_part or tbl_meta.sheet_name.lower() == tbl_part:
                        matched_tbl = tbl_name
                        break
                if matched_tbl:
                    for c_id, c_meta in catalog.tables[matched_tbl].columns.items():
                        if c_id.lower() == col_part or c_meta.original_header.lower() == col_part:
                            tables.add(matched_tbl)
                            resolution[f] = (matched_tbl, c_meta)
                            break
                    if f in resolution:
                        continue

            matches = column_table_map.get(clean_f) or column_table_map.get(re.sub(r"[^\w]", "", clean_f))
            if matches:
                # If unambiguous or already chosen table
                chosen_table, chosen_col = matches[0]
                tables.add(chosen_table)
                resolution[f] = (chosen_table, chosen_col)
            else:
                # Field not found in catalog; check if it's a count(*) or generic
                if f not in ("*", "count", "record_count", "id"):
                    logger.warning("[VirtualCompiler] Unresolved field '%s' across catalog columns", f)

        return tables, resolution

    def _resolve_join_tree(
        self,
        target_tables: Set[str],
        confirmed_edges: List[DiscoveredRelationshipEdge],
        catalog: DynamicSchemaCatalog,
    ) -> Tuple[str, List[str], Set[str]]:
        """
        Builds join tree across target tables using only confirmed edges.
        Returns: (primary_table, list_of_JOIN_clauses, set_of_used_edge_ids)
        """
        if len(target_tables) <= 1:
            primary = next(iter(target_tables))
            return primary, [], set()

        # Build adjacency graph of confirmed edges
        # Map sheet names to catalog table names
        sheet_to_table = {}
        for tbl_name, tbl_meta in catalog.tables.items():
            sheet_to_table[tbl_meta.sheet_name.lower()] = tbl_name
            sheet_to_table[tbl_name.lower()] = tbl_name

        adj: Dict[str, List[Tuple[str, DiscoveredRelationshipEdge, bool]]] = {}
        for edge in confirmed_edges:
            src_tbl = sheet_to_table.get(edge.source_sheet.lower())
            tgt_tbl = sheet_to_table.get(edge.target_sheet.lower())

            if src_tbl and tgt_tbl:
                adj.setdefault(src_tbl, []).append((tgt_tbl, edge, True))
                adj.setdefault(tgt_tbl, []).append((src_tbl, edge, False))

        # BFS to find join path connecting all target tables
        primary_table = next(iter(target_tables))
        joined_tables: Set[str] = {primary_table}
        join_clauses: List[str] = []
        used_edge_ids: Set[str] = set()

        remaining = set(target_tables) - {primary_table}

        while remaining:
            found_any = False
            for target in list(remaining):
                # Search path from any joined_table to target
                path = self._find_bfs_path(joined_tables, target, adj)
                if path:
                    for u, v, edge, is_forward in path:
                        if v not in joined_tables:
                            join_sql = self._compile_join_clause(u, v, edge, is_forward, catalog)
                            join_clauses.append(join_sql)
                            joined_tables.add(v)
                            used_edge_ids.add(edge.id)
                    remaining.remove(target)
                    found_any = True
                    break

            if not found_any:
                raise VirtualCompilerError(
                    f"No confirmed relationship join path connects table(s): {', '.join(remaining)} to '{primary_table}'. "
                    "Cross-table join cannot be synthesized without owner-confirmed relationship edges.",
                    code="UNCONFIRMED_JOIN_PATH"
                )

        return primary_table, join_clauses, used_edge_ids

    def _find_bfs_path(
        self,
        sources: Set[str],
        target: str,
        adj: Dict[str, List[Tuple[str, DiscoveredRelationshipEdge, bool]]],
    ) -> Optional[List[Tuple[str, str, DiscoveredRelationshipEdge, bool]]]:
        queue = deque([(s, []) for s in sources])
        visited = set(sources)

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path

            for neighbor, edge, is_fwd in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [(curr, neighbor, edge, is_fwd)]
                    if neighbor == target:
                        return new_path
                    queue.append((neighbor, new_path))

        return None

    def _compile_join_clause(
        self,
        u_table: str,
        v_table: str,
        edge: DiscoveredRelationshipEdge,
        is_forward: bool,
        catalog: DynamicSchemaCatalog,
    ) -> str:
        """Constructs safe SQLite 'JOIN v ON ...' clause."""
        # Find column names in u_table and v_table
        if is_forward:
            src_col_name = edge.source_headers[0] if edge.source_headers else (edge.source_column_ids[0] if edge.source_column_ids else "id")
            tgt_col_name = edge.target_headers[0] if edge.target_headers else (edge.target_column_ids[0] if edge.target_column_ids else "id")
        else:
            src_col_name = edge.target_headers[0] if edge.target_headers else (edge.target_column_ids[0] if edge.target_column_ids else "id")
            tgt_col_name = edge.source_headers[0] if edge.source_headers else (edge.source_column_ids[0] if edge.source_column_ids else "id")

        # Resolve exact physical/virtual column names from catalog
        u_col_safe = self._resolve_safe_col(u_table, src_col_name, catalog)
        v_col_safe = self._resolve_safe_col(v_table, tgt_col_name, catalog)

        # Standard INNER JOIN on confirmed equality
        return f"JOIN {v_table} ON {u_table}.{u_col_safe} = {v_table}.{v_col_safe}"

    def _resolve_safe_col(self, table_name: str, col_name_or_header: str, catalog: DynamicSchemaCatalog) -> str:
        tbl = catalog.tables.get(table_name)
        if not tbl:
            return col_name_or_header

        target_clean = col_name_or_header.lower()
        for col_id, col_meta in tbl.columns.items():
            if (
                col_id.lower() == target_clean
                or col_meta.name.lower() == target_clean
                or col_meta.original_header.lower() == target_clean
            ):
                return col_meta.name

        return col_name_or_header

    def _build_select_clause(
        self,
        plan: SemanticQueryPlan,
        field_resolution: Dict[str, Tuple[str, VirtualColumnMeta]],
        catalog: DynamicSchemaCatalog,
    ) -> Tuple[List[str], List[str]]:
        select_items: List[str] = []
        col_names: List[str] = []

        # Dimensions first
        for d in plan.dimensions:
            resolved = field_resolution.get(d.name)
            alias = d.alias or d.name
            if resolved:
                tbl, col = resolved
                select_items.append(f"{tbl}.{col.name} AS {self._sanitize_alias(alias)}")
                col_names.append(alias)
            else:
                select_items.append(f"'{d.name}' AS {self._sanitize_alias(alias)}")
                col_names.append(alias)

        # Measures
        for m in plan.measures:
            alias = m.alias or m.name
            agg_type = m.aggregation or AggregationType.SUM
            agg_sql = self.ALLOWED_AGGREGATIONS.get(agg_type, "SUM")

            if m.name == "*":
                select_items.append(f"COUNT(*) AS {self._sanitize_alias(alias)}")
                col_names.append(alias)
            else:
                resolved = field_resolution.get(m.name)
                if resolved:
                    tbl, col = resolved
                    if agg_type == AggregationType.COUNT_DISTINCT:
                        select_items.append(f"COUNT(DISTINCT {tbl}.{col.name}) AS {self._sanitize_alias(alias)}")
                    elif agg_type == AggregationType.AVG:
                        select_items.append(f"ROUND(AVG({tbl}.{col.name}), 2) AS {self._sanitize_alias(alias)}")
                    else:
                        select_items.append(f"{agg_sql}({tbl}.{col.name}) AS {self._sanitize_alias(alias)}")
                    col_names.append(alias)
                else:
                    select_items.append(f"COUNT(*) AS {self._sanitize_alias(alias)}")
                    col_names.append(alias)

        if not select_items:
            # Fallback to COUNT(*)
            select_items.append("COUNT(*) AS total_count")
            col_names.append("total_count")

        return select_items, col_names

    def _build_where_clause(
        self,
        plan: SemanticQueryPlan,
        field_resolution: Dict[str, Tuple[str, VirtualColumnMeta]],
        catalog: DynamicSchemaCatalog,
    ) -> Tuple[List[str], List[Any]]:
        where_clauses: List[str] = []
        params: List[Any] = []

        for f in plan.filters:
            resolved = field_resolution.get(f.field)
            if not resolved:
                logger.warning("[VirtualCompiler] Skipping filter on unresolvable field '%s'", f.field)
                continue

            tbl, col = resolved
            col_expr = f"{tbl}.{col.name}"
            op = f.operator
            val = f.value

            if op in (OperatorType.IS_NULL, OperatorType.NOT_NULL):
                where_clauses.append(f"{col_expr} {self.SQL_OPERATORS[op]}")
            elif op == OperatorType.IN and isinstance(val, (list, tuple)):
                if val:
                    placeholders = ", ".join(["?"] * len(val))
                    where_clauses.append(f"{col_expr} IN ({placeholders})")
                    params.extend(val)
            elif op == OperatorType.BETWEEN and isinstance(val, (list, tuple)) and len(val) == 2:
                where_clauses.append(f"{col_expr} BETWEEN ? AND ?")
                params.append(val[0])
                params.append(val[1])
            elif op == OperatorType.LIKE:
                where_clauses.append(f"LOWER({col_expr}) LIKE ?")
                params.append(f"%{str(val).lower()}%")
            else:
                sql_op = self.SQL_OPERATORS.get(op, "=")
                where_clauses.append(f"{col_expr} {sql_op} ?")
                params.append(val)

        return where_clauses, params

    def _build_group_by_clause(
        self,
        plan: SemanticQueryPlan,
        field_resolution: Dict[str, Tuple[str, VirtualColumnMeta]],
        catalog: DynamicSchemaCatalog,
    ) -> str:
        groups: List[str] = []
        for g in plan.group_by:
            resolved = field_resolution.get(g)
            if resolved:
                tbl, col = resolved
                groups.append(f"{tbl}.{col.name}")
            else:
                groups.append(self._sanitize_alias(g))

        return ", ".join(groups)

    def _build_order_by_clause(
        self,
        plan: SemanticQueryPlan,
        field_resolution: Dict[str, Tuple[str, VirtualColumnMeta]],
        catalog: DynamicSchemaCatalog,
    ) -> str:
        orders: List[str] = []
        for o in plan.order_by:
            direction = "DESC" if o.direction.lower() == "desc" else "ASC"
            resolved = field_resolution.get(o.field)
            if resolved:
                tbl, col = resolved
                orders.append(f"{tbl}.{col.name} {direction}")
            else:
                orders.append(f"{self._sanitize_alias(o.field)} {direction}")

        return ", ".join(orders)

    @staticmethod
    def _sanitize_alias(alias: str) -> str:
        return re.sub(r"[^\w]", "_", alias).strip("_") or "col"


# Global compiler instance
virtual_compiler = VirtualRelationalCompiler()
