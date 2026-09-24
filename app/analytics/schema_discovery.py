"""
Dynamic Database Schema Discovery & Auto-Cataloging Engine.
Introspects MySQL information_schema (tables, columns, data types, foreign keys).
Automatically discovers and registers Measures, Dimensions, Relationships, and Capability Matrices at runtime
so that any new table or column is immediately queryable without manual code changes.
"""

import time
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import pymysql
import pymysql.cursors
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AutoCatalogingEngine:
    """
    Introspects database schema and auto-populates semantic catalog registries dynamically.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl = cache_ttl_seconds
        self._last_refresh = 0.0
        self._schema_cache: Dict[str, Any] = {}
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.user = os.getenv("DB_USERNAME", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_DATABASE", "chatbot_db")

        # Excluded internal technical tables
        self.excluded_tables: Set[str] = {
            "failed_jobs", "job_batches", "jobs", "migrations",
            "password_reset_tokens", "sessions", "cache", "cache_locks"
        }
        # Excluded sensitive authentication / secret columns
        self.excluded_columns: Set[str] = {
            "password", "password_hash", "remember_token", "access_token",
            "refresh_token", "secret", "api_key", "token"
        }

    def _get_connection(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=3,
        )

    def discover_and_catalog(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Reads information_schema and generates live semantic measures, dimensions, relationships,
        and capability matrix for all tables.
        """
        now = time.time()
        if not force_refresh and self._schema_cache and (now - self._last_refresh < self.cache_ttl):
            return self._schema_cache

        discovered: Dict[str, Any] = {
            "tables": {},
            "measures": {},
            "dimensions": {},
            "relationships": [],
            "capability_matrix": {},
            "field_types": {},
            "prompt_catalog": "",
            "discovered_at": now,
        }

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Fetch all columns and data types from information_schema
                    sql_columns = """
                        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = %s
                        ORDER BY TABLE_NAME, ORDINAL_POSITION;
                    """
                    cursor.execute(sql_columns, (self.database,))
                    col_rows = cursor.fetchall()

                    # 2. Fetch Foreign Key Constraints & Relationships
                    sql_fks = """
                        SELECT 
                            TABLE_NAME AS source_table,
                            COLUMN_NAME AS source_column,
                            REFERENCED_TABLE_NAME AS target_table,
                            REFERENCED_COLUMN_NAME AS target_column
                        FROM information_schema.KEY_COLUMN_USAGE
                        WHERE TABLE_SCHEMA = %s 
                          AND REFERENCED_TABLE_NAME IS NOT NULL;
                    """
                    cursor.execute(sql_fks, (self.database,))
                    fk_rows = cursor.fetchall()

                    # Organize columns by table
                    tables_map: Dict[str, List[Dict[str, Any]]] = {}
                    for row in col_rows:
                        t_name = row["TABLE_NAME"]
                        if t_name in self.excluded_tables:
                            continue
                        if t_name not in tables_map:
                            tables_map[t_name] = []
                        tables_map[t_name].append(row)

                    prompt_lines = ["### Live Auto-Discovered Database Schema & Catalog:"]

                    for t_name, cols in tables_map.items():
                        # Determine domain name from table name (e.g. analytics_products -> product, analytics_orders -> sales)
                        domain_key = t_name.replace("analytics_", "").rstrip("s")
                        if domain_key == "order":
                            domain_key = "sales"

                        discovered["capability_matrix"][domain_key] = {
                            "table": t_name,
                            "measures": set(),
                            "dimensions": set(),
                            "group_by": set(),
                            "derived_metrics": set(),
                        }

                        table_meta = {
                            "table_name": t_name,
                            "domain": domain_key,
                            "columns": {},
                            "measures": [],
                            "dimensions": [],
                            "dates": [],
                        }

                        has_workspace = any(c["COLUMN_NAME"] == "workspace_id" for c in cols)
                        table_meta["has_workspace_id"] = has_workspace

                        for col in cols:
                            c_name = col["COLUMN_NAME"]
                            d_type = col["DATA_TYPE"].lower()
                            if c_name in self.excluded_columns:
                                continue
                            c_key = col["COLUMN_KEY"]

                            # Auto-Classify field types
                            is_primary = (c_key == "PRI")
                            is_fk = c_name.endswith(("_id", "Id")) and not is_primary
                            
                            # Categorize: numeric, date, or string
                            if d_type in ("decimal", "float", "double", "int", "bigint", "mediumint", "smallint", "tinyint"):
                                if is_primary or is_fk or c_name in ("is_active", "status", "workspace_id", "channel_id", "user_id"):
                                    sem_type = "integer" if is_primary else "identifier"
                                    # ID fields can be used as dimensions for counting/filtering
                                    dim_name = c_name
                                    discovered["dimensions"][dim_name] = {
                                        "name": dim_name,
                                        "table": t_name,
                                        "column": c_name,
                                        "data_type": "integer",
                                    }
                                    discovered["field_types"][dim_name] = "numeric"
                                    discovered["capability_matrix"][domain_key]["dimensions"].add(dim_name)
                                    table_meta["dimensions"].append(dim_name)
                                else:
                                    sem_type = "numeric"
                                    # Auto-register as Measure
                                    measure_name = c_name
                                    discovered["measures"][measure_name] = {
                                        "name": measure_name,
                                        "domain": domain_key,
                                        "table": t_name,
                                        "column": c_name,
                                        "data_type": "currency" if any(kw in c_name for kw in ("amount", "price", "cost", "total", "spent", "due", "revenue", "salary", "balance")) else "numeric",
                                    }
                                    discovered["field_types"][measure_name] = "numeric"
                                    discovered["capability_matrix"][domain_key]["measures"].add(measure_name)
                                    table_meta["measures"].append(measure_name)

                            elif d_type in ("date", "datetime", "timestamp", "time", "year"):
                                sem_type = "date"
                                dim_name = c_name
                                discovered["dimensions"][dim_name] = {
                                    "name": dim_name,
                                    "table": t_name,
                                    "column": c_name,
                                    "data_type": "date",
                                }
                                discovered["field_types"][dim_name] = "date"
                                discovered["capability_matrix"][domain_key]["dimensions"].add(dim_name)
                                discovered["capability_matrix"][domain_key]["group_by"].add(dim_name)
                                table_meta["dates"].append(dim_name)

                            else:  # varchar, text, enum, json, char
                                sem_type = "string"
                                dim_name = c_name
                                discovered["dimensions"][dim_name] = {
                                    "name": dim_name,
                                    "table": t_name,
                                    "column": c_name,
                                    "data_type": "string",
                                }
                                discovered["field_types"][dim_name] = "string"
                                discovered["capability_matrix"][domain_key]["dimensions"].add(dim_name)
                                discovered["capability_matrix"][domain_key]["group_by"].add(dim_name)
                                table_meta["dimensions"].append(dim_name)

                            table_meta["columns"][c_name] = {
                                "data_type": d_type,
                                "semantic_type": sem_type,
                                "is_nullable": col["IS_NULLABLE"] == "YES",
                            }

                        # Add domain count measure by default
                        count_measure = f"{domain_key}_count"
                        discovered["measures"][count_measure] = {
                            "name": count_measure,
                            "domain": domain_key,
                            "table": t_name,
                            "column": "id",
                            "data_type": "integer",
                        }
                        discovered["capability_matrix"][domain_key]["measures"].add(count_measure)
                        discovered["field_types"][count_measure] = "numeric"

                        discovered["tables"][t_name] = table_meta

                        # Build markdown schema description
                        prompt_lines.append(f"- **Table `{t_name}`** (Domain: `{domain_key}`):")
                        if table_meta["measures"]:
                            prompt_lines.append(f"  - Measures (Numeric): {', '.join(table_meta['measures'])}")
                        if table_meta["dimensions"]:
                            prompt_lines.append(f"  - Dimensions (Attributes): {', '.join(table_meta['dimensions'][:8])}")
                        if table_meta["dates"]:
                            prompt_lines.append(f"  - Dates: {', '.join(table_meta['dates'])}")

                    # Auto-register Relationships from Foreign Keys + naming conventions
                    for fk in fk_rows:
                        s_table = fk["source_table"]
                        t_table = fk["target_table"]
                        s_col = fk["source_column"]
                        t_col = fk["target_column"]

                        if s_table in self.excluded_tables or t_table in self.excluded_tables:
                            continue

                        discovered["relationships"].append({
                            "source_table": s_table,
                            "target_table": t_table,
                            "source_key": s_col,
                            "target_key": t_col,
                            "join_sql": f"JOIN `{t_table}` ON `{t_table}`.`{t_col}` = `{s_table}`.`{s_col}`",
                        })

                    discovered["prompt_catalog"] = "\n".join(prompt_lines)
                    self._schema_cache = discovered
                    self._last_refresh = now
                    logger.info(f"[AutoCatalogingEngine] Auto-cataloged {len(tables_map)} tables, {len(discovered['measures'])} measures, {len(discovered['dimensions'])} dimensions.")

        except Exception as e:
            logger.error(f"[AutoCatalogingEngine] Failed to auto-catalog schema: {e}", exc_info=True)
            if not self._schema_cache:
                self._schema_cache = {
                    "tables": {}, "measures": {}, "dimensions": {}, "relationships": [],
                    "capability_matrix": {}, "field_types": {}, "prompt_catalog": "", "discovered_at": now
                }

        return self._schema_cache

    def get_prompt_schema_context(self) -> str:
        return self.get_prompt_catalog()

    def get_prompt_catalog(self) -> str:
        """Returns the dynamic prompt catalog string for LLM system prompt injection."""
        data = self.discover_and_catalog()
        return data.get("prompt_catalog", "")


# Global Singleton
auto_catalog = AutoCatalogingEngine()

schema_discovery = auto_catalog
