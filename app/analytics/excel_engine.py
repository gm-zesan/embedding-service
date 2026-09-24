"""
Dynamic Excel-to-Database AI Q&A Engine.
Ingests multi-sheet Excel (.xlsx, .xls) and CSV files into isolated virtual SQLite databases,
auto-discovers schema, and performs conversational semantic natural language Q&A.
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
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ExcelDatabaseEngine:
    """
    Manages multi-sheet Excel file ingestion, SQLite dynamic database creation, and AI Q&A.
    """

    def __init__(self):
        self.storage_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "storage",
            "excel_databases"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

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
        Reads all sheets/tabs from Excel file, loads them as tables into an isolated SQLite database,
        and generates live schema metadata.
        """
        file_hash = hashlib.md5(f"{workspace_id}_{original_filename}_{time.time()}".encode()).hexdigest()[:12]
        db_filename = f"ws_{workspace_id}_{file_hash}.sqlite"
        db_path = os.path.join(self.storage_dir, db_filename)

        # 1. Read sheets
        ext = os.path.splitext(original_filename)[1].lower()
        if ext in (".xlsx", ".xls"):
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
        elif ext == ".csv":
            sheet_names = ["Sheet1"]
            xls = None
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only .xlsx, .xls, and .csv are supported.")

        conn = sqlite3.connect(db_path)
        tables_meta: Dict[str, Any] = {}
        total_rows = 0

        try:
            for sheet in sheet_names:
                table_name = self._sanitize_name(sheet) or "data_table"
                if xls:
                    df = pd.read_excel(xls, sheet_name=sheet)
                else:
                    df = pd.read_csv(file_path)

                # Clean dataframe column names
                original_cols = list(df.columns)
                clean_cols = [self._sanitize_name(c) for c in original_cols]
                # Ensure unique column names
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

                # Clean NaN / NaT
                df = df.where(pd.notnull(df), None)

                # Write to SQLite table
                df.to_sql(table_name, conn, if_exists="replace", index=False)

                row_count = len(df)
                total_rows += row_count

                # Analyze column types
                measures = []
                dimensions = []
                for c in df.columns:
                    dtype = str(df[c].dtype).lower()
                    if "int" in dtype or "float" in dtype:
                        measures.append(c)
                    else:
                        dimensions.append(c)

                tables_meta[table_name] = {
                    "sheet_title": sheet,
                    "table_name": table_name,
                    "row_count": row_count,
                    "columns": final_cols,
                    "original_columns": original_cols,
                    "measures": measures,
                    "dimensions": dimensions,
                    "sample_preview": df.head(3).to_dict(orient="records"),
                }

            conn.commit()
        finally:
            conn.close()

        # Build prompt catalog
        catalog_lines = [
            f"### Uploaded Excel File: `{original_filename}` (Virtual Database)",
            f"Total Sheets/Tables: {len(tables_meta)}, Total Records: {total_rows}",
            "",
        ]
        for t_name, meta in tables_meta.items():
            catalog_lines.append(f"- **Table/Sheet `{t_name}`** (Original: \"{meta['sheet_title']}\", {meta['row_count']} rows):")
            catalog_lines.append(f"  - Columns: {', '.join(meta['columns'])}")
            if meta["measures"]:
                catalog_lines.append(f"  - Numeric Measures: {', '.join(meta['measures'])}")
            if meta["dimensions"]:
                catalog_lines.append(f"  - Text/Dimensions: {', '.join(meta['dimensions'][:8])}")

        schema_catalog = "\n".join(catalog_lines)

        metadata = {
            "file_id": file_hash,
            "filename": original_filename,
            "db_path": db_path,
            "db_filename": db_filename,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "sheets_count": len(sheet_names),
            "total_rows": total_rows,
            "tables": tables_meta,
            "schema_catalog": schema_catalog,
            "created_at": time.time(),
        }

        # Save metadata JSON alongside SQLite DB
        meta_path = os.path.join(self.storage_dir, f"ws_{workspace_id}_{file_hash}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[ExcelDatabaseEngine] Successfully ingested '{original_filename}' into SQLite DB: {db_filename}")
        return metadata

    def get_file_metadata(self, workspace_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves stored metadata for an uploaded Excel database."""
        meta_path = os.path.join(self.storage_dir, f"ws_{workspace_id}_{file_id}.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return None

    def query_excel(
        self,
        workspace_id: int,
        file_id: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes AI-powered natural language SQL analytics against the uploaded Excel virtual database.
        """
        t_start = time.perf_counter()
        meta = self.get_file_metadata(workspace_id, file_id)
        if not meta:
            # Look for most recent file in workspace
            all_files = [
                f for f in os.listdir(self.storage_dir)
                if f.startswith(f"ws_{workspace_id}_") and f.endswith(".json")
            ]
            if all_files:
                all_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.storage_dir, x)), reverse=True)
                with open(os.path.join(self.storage_dir, all_files[0]), "r") as f:
                    meta = json.load(f)
            else:
                return {
                    "success": False,
                    "report": "⚠️ **Excel Database Notice**: কোনো আপলোড করা এক্সেল ফাইল পাওয়া যায়নি। অনুগ্রহ করে প্রথমে একটি এক্সেল ফাইল আপলোড করুন।",
                    "rows": [],
                    "latency_ms": 0.0,
                }

        db_path = meta["db_path"]
        schema_catalog = meta["schema_catalog"]

        system_prompt = f"""You are an Expert SQLite Database Analytics Engineer.
Your task is to translate natural language business questions into valid, optimized, read-only SQLite SQL queries against an uploaded Excel spreadsheet database.

<DATABASE_SCHEMA>
{schema_catalog}
</DATABASE_SCHEMA>

STRICT RULES:
1. Generate ONLY SELECT statements. NEVER generate INSERT, UPDATE, DELETE, DROP, or ALTER statements.
2. Use exact table names and column names from the schema catalog above.
3. For text filtering, use LIKE '%value%' or LOWER(column) = LOWER('value') to ensure case-insensitive matching.
4. If the user asks for a total, use SUM(column). If count, use COUNT(*). If average, use AVG(column).
5. Always round averages or monetary aggregations to 2 decimal places: ROUND(AVG(col), 2).
6. When comparing or grouping across sheets/tabs, perform a standard JOIN on matching ID or name columns.
7. Return ONLY a valid JSON object:
{{
  "sql": "SELECT ...",
  "explanation": "Short summary of what this query calculates"
}}
"""

        user_content = f"Business Question: {question}"

        # Call LLM
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

        # Security check: Read-Only SQLite
        clean_sql = sql.lower()
        if not clean_sql.startswith("select") and not clean_sql.startswith("with"):
            return {
                "success": False,
                "report": "🛡️ **Security Notice**: Only read-only SELECT queries are allowed on uploaded files.",
                "rows": [],
                "latency_ms": 0.0,
            }

        # Execute on SQLite
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

        # Format report into markdown table or summary card
        report = self._format_results(question, meta["filename"], rows, col_names, total_latency)

        return {
            "success": True,
            "report": report,
            "sql": sql,
            "rows": rows,
            "filename": meta["filename"],
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
                f"📋 **Excel Analysis Report** (`{filename}`)\n\n"
                f"> **Question:** {question}\n\n"
                f"*কোনো ডেটা পাওয়া যায়নি (0 records matched).*\n\n"
                f"*(Latency: {latency_ms}ms)*"
            )

        # Single row scalar summary (e.g. Total sum, single count, or single value)
        if len(rows) == 1 and len(columns) <= 3:
            r = rows[0]
            lines = [
                f"📊 **Excel Analytics Summary** (`{filename}`)\n",
            ]
            for col in columns:
                val = r[col]
                col_label = col.replace("_", " ").title()
                if isinstance(val, (int, float)):
                    if any(kw in col.lower() for kw in ("amount", "price", "cost", "salary", "spent", "total", "fee", "revenue")):
                        lines.append(f"- **{col_label}:** ৳{val:,.2f}" if val >= 0 else f"- **{col_label}:** -৳{abs(val):,.2f}")
                    else:
                        lines.append(f"- **{col_label}:** {val:,.2f}" if isinstance(val, float) else f"- **{col_label}:** {val:,}")
                else:
                    lines.append(f"- **{col_label}:** {val}")

            lines.append(f"\n*(Latency: {latency_ms}ms)*")
            return "\n".join(lines)

        # Multi-row Table Report
        headers = [c.replace("_", " ").title() for c in columns]
        lines = [
            f"📋 **Excel Analytics Report** (`{filename}`)\n",
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
                    if any(kw in col.lower() for kw in ("amount", "price", "cost", "salary", "spent", "total", "revenue")):
                        row_vals.append(f"৳{val:,.2f}")
                    else:
                        row_vals.append(f"{val:,.2f}")
                elif isinstance(val, int) and any(kw in col.lower() for kw in ("amount", "price", "cost", "salary", "spent", "total", "revenue")):
                    row_vals.append(f"৳{val:,}")
                else:
                    row_vals.append(str(val))

            lines.append(f"| {' | '.join(row_vals)} |")

        row_count_str = f"Found: {len(rows)} rows" if len(rows) < 25 else "Showing first 25 rows"
        lines.append(f"\n*({row_count_str} | Latency: {latency_ms}ms)*")
        return "\n".join(lines)


# Global Singleton
excel_engine = ExcelDatabaseEngine()
