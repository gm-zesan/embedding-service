"""
Generic Document & Data Source Analytics Engine.
Ingests multi-sheet Excel (.xlsx, .xls) and CSV files into isolated virtual SQLite databases,
auto-discovers schema, registers data sources, and performs conversational semantic natural language Q&A.
"""

import os
import re
import json
import time
import sqlite3
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from .data_source import (
    DataSource,
    Dataset,
    DataSourceFormat,
    DocumentAnalyticsContext,
    ResolutionStatus,
    SourceResolutionResult,
)
from .source_registry import DataSourceRegistry, source_registry
from .source_resolver import DataSourceResolver, source_resolver

load_dotenv()
logger = logging.getLogger(__name__)


class ExcelDatabaseEngine:
    """
    Manages multi-sheet Excel and CSV file ingestion, SQLite dynamic database creation,
    workspace isolation, and format-aware AI Q&A.
    """

    def __init__(self, registry: Optional[DataSourceRegistry] = None, resolver: Optional[DataSourceResolver] = None):
        self.registry = registry or source_registry
        self.resolver = resolver or source_resolver
        self.storage_dir = self.registry.storage_dir

        api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy-key-for-init"
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

    def _sanitize_name(self, name: str) -> str:
        """Converts raw string to safe snake_case SQL identifier."""
        s = re.sub(r"[^\w\s]", "", str(name).strip())
        s = re.sub(r"\s+", "_", s).lower()
        if not s or s[0].isdigit():
            s = "col_" + s
        return s[:64]

    def ingest_excel_file(
        self,
        file_path: str,
        original_filename: str,
        workspace_id: int,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reads all sheets/tabs from Excel file or CSV, loads them as tables into an isolated SQLite database,
        and generates live schema metadata in the DataSourceRegistry.
        """
        ext = os.path.splitext(original_filename)[1].lower()
        if ext in (".xlsx", ".xls"):
            fmt = DataSourceFormat.XLSX if ext == ".xlsx" else DataSourceFormat.XLS
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
        elif ext == ".csv":
            fmt = DataSourceFormat.CSV
            sheet_names = ["Sheet1"]
            xls = None
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only .xlsx, .xls, and .csv are supported.")

        file_hash = hashlib.md5(f"{workspace_id}_{original_filename}_{time.time()}".encode()).hexdigest()[:12]
        db_filename = f"ws_{workspace_id}_{file_hash}.sqlite"
        db_path = os.path.join(self.storage_dir, db_filename)

        conn = sqlite3.connect(db_path)
        datasets: List[Dataset] = []
        tables_meta: Dict[str, Any] = {}
        total_rows = 0

        try:
            for sheet in sheet_names:
                table_name = self._sanitize_name(sheet) or "data_table"
                if xls:
                    df = pd.read_excel(xls, sheet_name=sheet)
                else:
                    df = pd.read_csv(file_path)

                original_cols = list(df.columns)
                clean_cols = [self._sanitize_name(c) for c in original_cols]
                seen = {}
                final_cols = []
                for c in clean_cols:
                    if c in seen:
                        seen[c] += 1
                        final_cols.append(f"{c}_{seen[c]}")
                    else:
                        seen[c] = 0
                        final_cols.append(c)

                df.columns = final_cols

                # Save dataframe to SQLite
                df.to_sql(table_name, conn, if_exists="replace", index=False)

                numeric_cols = []
                text_cols = []
                data_types = {}
                for col in final_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric_cols.append(col)
                        data_types[col] = "NUMERIC"
                    else:
                        text_cols.append(col)
                        data_types[col] = "TEXT"

                row_count = len(df)
                total_rows += row_count

                dataset = Dataset(
                    dataset_id=table_name,
                    dataset_name=sheet if sheet != "Sheet1" else os.path.splitext(original_filename)[0].replace("_", " ").title(),
                    sheet_name=sheet,
                    columns=final_cols,
                    data_types=data_types,
                    numeric_measures=numeric_cols,
                    text_dimensions=text_cols,
                    date_fields=[c for c in final_cols if "date" in c.lower() or "time" in c.lower()],
                    row_count=row_count,
                    storage_reference=table_name,
                )
                datasets.append(dataset)

                tables_meta[table_name] = {
                    "original_sheet": sheet,
                    "columns": final_cols,
                    "numeric_measures": numeric_cols,
                    "text_dimensions": text_cols,
                    "row_count": row_count,
                }
        finally:
            conn.close()

        # Build schema catalog summary
        schema_lines = [
            f"### Uploaded File: `{original_filename}` (Virtual Database)",
            f"Total Sheets/Tables: {len(datasets)}, Total Records: {total_rows}\n"
        ]
        for d in datasets:
            schema_lines.append(f"- **Table/Sheet `{d.storage_reference}`** (Original: \"{d.sheet_name}\", {d.row_count} rows):")
            schema_lines.append(f"  - Columns: {', '.join(d.columns)}")
            if d.numeric_measures:
                schema_lines.append(f"  - Numeric Measures: {', '.join(d.numeric_measures)}")
            if d.text_dimensions:
                schema_lines.append(f"  - Text/Dimensions: {', '.join(d.text_dimensions)}")

        schema_summary = "\n".join(schema_lines)

        source = DataSource(
            workspace_id=workspace_id,
            source_id=file_hash,
            original_filename=original_filename,
            format=fmt,
            stored_path=file_path,
            uploaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            file_hash=file_hash,
            datasets=datasets,
            schema_summary=schema_summary,
            total_rows=total_rows,
            db_path=db_path,
        )

        # Register in workspace-scoped registry
        self.registry.register_source(source)

        return {
            "success": True,
            "message": f"Successfully ingested '{original_filename}' with {len(datasets)} sheets and {total_rows} rows.",
            "file_id": file_hash,
            "source_id": file_hash,
            "filename": original_filename,
            "sheets": [d.storage_reference for d in datasets],
            "sheets_count": len(datasets),
            "tables": tables_meta,
            "total_rows": total_rows,
            "schema_summary": schema_summary,
            "db_path": db_path,
        }

    def get_file_metadata(self, workspace_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves file metadata from DataSourceRegistry with workspace boundary enforcement."""
        src = self.registry.get_source(workspace_id, file_id)
        if src:
            return src.model_dump()
        return None

    def query_excel(
        self,
        workspace_id: int,
        file_id: Optional[str] = None,
        question: str = "",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes format-aware AI analytics against the resolved data source and datasets.
        """
        t_start = time.perf_counter()

        # Build DocumentAnalyticsContext
        ctx = DocumentAnalyticsContext(
            workspace_id=workspace_id,
            active_source_id=file_id if file_id else None,
            history=history,
        )

        # ── 1. Resolve Data Source & Datasets ─────────────────────────
        resolution = self.resolver.resolve(
            query=question,
            workspace_id=workspace_id,
            explicit_source_id=file_id,
            context=ctx,
        )

        if resolution.status == ResolutionStatus.AMBIGUOUS:
            return {
                "success": True,
                "is_ambiguous": True,
                "report": resolution.clarification_message,
                "clarification_options": resolution.clarification_options,
                "candidates": resolution.candidates,
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        if resolution.status == ResolutionStatus.UNSUPPORTED_CROSS_SOURCE:
            return {
                "success": True,
                "is_cross_source_unsupported": True,
                "report": resolution.clarification_message,
                "clarification_options": resolution.clarification_options,
                "candidates": resolution.candidates,
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        if resolution.status == ResolutionStatus.NOT_FOUND or not resolution.source:
            return {
                "success": False,
                "report": resolution.clarification_message or "⚠️ **Data Source Notice**: কোনো আপলোড করা ফাইল বা ডেটাসেট পাওয়া যায়নি।",
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # ── 2. Build Bounded Dataset Schema for LLM Planner ───────────
        target_source = resolution.source
        db_path = target_source.db_path
        if not db_path or not os.path.exists(db_path):
            return {
                "success": False,
                "report": f"⚠️ **Database Error**: Physical storage file for '{target_source.original_filename}' is missing.",
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # Isolate schema to selected datasets
        target_datasets = resolution.selected_datasets or target_source.datasets
        schema_lines = [
            f"### Active Data Source: `{target_source.original_filename}`",
            f"Format: {target_source.format.value.upper()}",
            "Available Tables for Querying:\n"
        ]
        for d in target_datasets:
            schema_lines.append(f"- **Table `{d.storage_reference}`** ({d.row_count} rows):")
            schema_lines.append(f"  - Columns: {', '.join(d.columns)}")
            if d.numeric_measures:
                schema_lines.append(f"  - Numeric Measures: {', '.join(d.numeric_measures)}")
            if d.text_dimensions:
                schema_lines.append(f"  - Text/Dimensions: {', '.join(d.text_dimensions)}")

        schema_catalog = "\n".join(schema_lines)

        # ── 3. Build Conversational History Block ─────────────────────
        history_block = ""
        if history:
            recent_turns = history[-4:]
            lines = []
            for h in recent_turns:
                role = "User" if h.get("role") == "user" else "Assistant"
                content = str(h.get("content", "")).strip()
                if content:
                    lines.append(f"{role}: {content[:150]}")
            if lines:
                history_block = "<RECENT_CONVERSATION_CONTEXT>\n" + "\n".join(lines) + "\n</RECENT_CONVERSATION_CONTEXT>\n\n"

        system_prompt = f"""You are an Expert SQLite Database Analytics Engineer.
Your task is to translate natural language business questions into valid, optimized, read-only SQLite SQL queries against an uploaded spreadsheet database.

<DATABASE_SCHEMA>
{schema_catalog}
</DATABASE_SCHEMA>

STRICT RULES:
1. Generate ONLY SELECT statements. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, DETACH, or PRAGMA statements.
2. Use exact table names and column names from the schema catalog above.
3. For text filtering, use LIKE '%value%' or LOWER(column) = LOWER('value') to ensure case-insensitive matching. Never include Bengali grammatical particles or inflections (e.g. 'er', 'r', 'ta', 'te', 'এর', 'র', 'টা') in the SQL filter value (e.g. for "Rice er stock" or "Rice-er stock", filter by product_name LIKE '%Rice%').
4. If the question asks about a specific product/person/entity (e.g. "Oil er stock", "Rice er price"), include a WHERE filter for that entity (e.g. WHERE LOWER(product_name) LIKE '%oil%'). If the user asks for an overall sheet total/count (e.g. "total stock", "all products"), aggregate across the whole table without an entity filter.
5. If the user asks for a total, use SUM(column). If count, use COUNT(*). If average, use AVG(column).
5. Always round averages or monetary aggregations to 2 decimal places: ROUND(AVG(col), 2).
6. When comparing or grouping across sheets/tabs, perform a standard JOIN on matching ID or name columns.
7. Return ONLY a valid JSON object:
{{
  "sql": "SELECT ...",
  "explanation": "Short summary of what this query calculates"
}}
"""

        user_content = f"{history_block}Current Business Question: {question}"

        # ── 4. Call LLM for Query Planning ───────────────────────────
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )

        res_text = response.choices[0].message.content.strip()
        parsed = json.loads(res_text)
        sql = parsed.get("sql", "").strip()

        # ── 5. Security & Read-Only Sandbox Guard ─────────────────────
        clean_sql = sql.lower().strip()
        forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "pragma", "vacuum"]
        if any(re.search(rf"\b{kw}\b", clean_sql) for kw in forbidden_keywords) or (not clean_sql.startswith("select") and not clean_sql.startswith("with")):
            return {
                "success": False,
                "report": "🛡️ **Security Notice**: Only read-only SELECT queries are allowed on uploaded data sources.",
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # ── 6. Execute against Sandboxed SQLite DB ────────────────────
        rows = []
        col_names = []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            if results:
                col_names = results[0].keys()
                rows = [dict(r) for r in results[:50]]
        except Exception as e:
            logger.error(f"[ExcelDatabaseEngine] SQL Execution error: {e} for SQL: {sql}")
            return {
                "success": False,
                "report": f"⚠️ **Query Execution Notice**: {str(e)}",
                "sql": sql,
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }
        finally:
            conn.close()

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        report = self._format_results(question, target_source.original_filename, rows, col_names, total_latency)

        return {
            "success": True,
            "report": report,
            "sql": sql,
            "rows": rows,
            "filename": target_source.original_filename,
            "source_id": target_source.source_id,
            "latency_ms": total_latency,
        }

    def _format_results(
        self,
        question: str,
        filename: str,
        rows: List[Dict[str, Any]],
        columns: List[str],
        latency_ms: float,
    ) -> str:
        """Formats query rows into a beautiful markdown report."""
        if not rows:
            return (
                f"📋 **Data Analysis Report** (`{filename}`)\n\n"
                f"> **Question:** {question}\n\n"
                f"*কোনো ডেটা পাওয়া যায়নি (0 records matched).*\n\n"
                f"*(Latency: {latency_ms}ms)*"
            )

        if len(rows) == 1 and len(columns) <= 3:
            r = rows[0]
            lines = [
                f"📊 **Data Analytics Summary** (`{filename}`)\n",
            ]
            for col in columns:
                val = r[col]
                col_label = col.replace("_", " ").title()
                if isinstance(val, (int, float)):
                    if any(kw in col.lower() for kw in ("amount", "price", "cost", "salary", "spent", "total", "fee", "revenue", "due", "cod")):
                        lines.append(f"- **{col_label}:** ৳{val:,.2f}" if val >= 0 else f"- **{col_label}:** -৳{abs(val):,.2f}")
                    else:
                        lines.append(f"- **{col_label}:** {val:,.2f}" if isinstance(val, float) else f"- **{col_label}:** {val:,}")
                else:
                    lines.append(f"- **{col_label}:** {val}")

            lines.append(f"\n*(Latency: {latency_ms}ms)*")
            return "\n".join(lines)

        headers = [c.replace("_", " ").title() for c in columns]
        lines = [
            f"📋 **Data Analytics Report** (`{filename}`)\n",
            f"> **Question:** {question}\n",
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(['---'] * len(headers))} |",
        ]

        for r in rows[:25]:
            row_vals = []
            for col in columns:
                val = r[col]
                if val is None:
                    row_vals.append("-")
                elif isinstance(val, float):
                    if any(kw in col.lower() for kw in ("amount", "price", "cost", "salary", "spent", "total", "revenue", "due", "cod")):
                        row_vals.append(f"৳{val:,.2f}")
                    else:
                        row_vals.append(f"{val:,.2f}")
                elif isinstance(val, int) and any(kw in col.lower() for kw in ("amount", "price", "cost", "salary", "spent", "total", "revenue", "due", "cod")):
                    row_vals.append(f"৳{val:,}")
                else:
                    row_vals.append(str(val))

            lines.append(f"| {' | '.join(row_vals)} |")

        row_count_str = f"Found: {len(rows)} rows" if len(rows) < 25 else "Showing first 25 rows"
        lines.append(f"\n*({row_count_str} | Latency: {latency_ms}ms)*")
        return "\n".join(lines)


# Global Singleton
excel_engine = ExcelDatabaseEngine()
