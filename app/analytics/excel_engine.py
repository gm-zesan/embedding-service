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
from pathlib import Path
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
    DatasetCategory,
    DatasetClassification,
)
from .source_registry import DataSourceRegistry, source_registry
from .source_resolver import DataSourceResolver, source_resolver
from .dataset_classifier import dataset_classifier, DatasetClassifier
from .semantic_schema import profile_dataset_columns
from .semantic_interpreter import get_semantic_interpreter
from app import config

load_dotenv()
logger = logging.getLogger(__name__)


class ExcelDatabaseEngine:
    """
    Manages multi-sheet Excel and CSV file ingestion, SQLite dynamic database creation,
    workspace isolation, and format-aware AI Q&A.
    """

    def __init__(
        self,
        registry: Optional[DataSourceRegistry] = None,
        resolver: Optional[DataSourceResolver] = None,
        max_sheets: Optional[int] = None,
        max_rows_per_sheet: Optional[int] = None,
        query_timeout: Optional[float] = None,
    ):
        self.registry = registry or source_registry
        self.resolver = resolver or source_resolver
        self.storage_dir = self.registry.storage_dir

        self.max_sheets = max_sheets if max_sheets is not None else getattr(config, "ANALYTICS_MAX_SHEETS", 10)
        self.max_rows_per_sheet = max_rows_per_sheet if max_rows_per_sheet is not None else getattr(config, "ANALYTICS_MAX_ROWS_PER_SHEET", 100000)
        self.query_timeout = query_timeout if query_timeout is not None else getattr(config, "ANALYTICS_QUERY_TIMEOUT_SECONDS", 15.0)

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
        conversation_id: Optional[str] = None,
        source_id: Optional[str] = None,
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
            if len(sheet_names) > self.max_sheets:
                raise ValueError(
                    f"Workbook exceeds maximum allowed sheets limit ({self.max_sheets}). Found {len(sheet_names)} sheets."
                )
        elif ext == ".csv":
            fmt = DataSourceFormat.CSV
            sheet_names = ["Sheet1"]
            xls = None
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only .xlsx, .xls, and .csv are supported.")

        if source_id and str(source_id).strip():
            file_hash = re.sub(r"[^\w\-]", "", str(source_id).strip())[:64]
        else:
            hasher = hashlib.md5()
            try:
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                content_digest = hasher.hexdigest()[:12]
            except Exception:
                content_digest = "content"
            file_hash = hashlib.md5(f"{workspace_id}_{original_filename}_{content_digest}".encode()).hexdigest()[:12]

        db_filename = f"ws_{workspace_id}_{file_hash}.sqlite"
        db_path = os.path.join(self.storage_dir, db_filename)

        conn = sqlite3.connect(db_path)
        datasets: List[Dataset] = []
        tables_meta: Dict[str, Any] = {}
        failed_sheets: List[Dict[str, str]] = []
        total_rows = 0
        seen_tables: Dict[str, int] = {}

        try:
            for sheet in sheet_names:
                base_table = self._sanitize_name(sheet) or "data_table"
                if base_table in seen_tables:
                    seen_tables[base_table] += 1
                    table_name = f"{base_table}_{seen_tables[base_table]}"
                else:
                    seen_tables[base_table] = 0
                    table_name = base_table
                if xls:
                    try:
                        # Enforce row bounds during streaming/reading to prevent unbounded memory allocation
                        df = pd.read_excel(xls, sheet_name=sheet, nrows=self.max_rows_per_sheet + 1)
                    except Exception as sheet_err:
                        logger.warning(f"[ExcelEngine] Error reading sheet '{sheet}' from {original_filename}: {sheet_err}")
                        failed_sheets.append({"sheet": sheet, "error": str(sheet_err)})
                        continue
                else:
                    try:
                        df = pd.read_csv(file_path, encoding="utf-8", nrows=self.max_rows_per_sheet + 1)
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, encoding="utf-8-sig", nrows=self.max_rows_per_sheet + 1)
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, encoding="latin1", nrows=self.max_rows_per_sheet + 1)

                if len(df) > self.max_rows_per_sheet:
                    raise ValueError(
                        f"Worksheet '{sheet}' exceeds maximum allowed data rows limit ({self.max_rows_per_sheet:,})."
                    )

                if df.empty and len(df.columns) == 0:
                    failed_sheets.append({"sheet": sheet, "error": "Empty sheet or no columns"})
                    continue

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

                date_cols = [c for c in final_cols if "date" in c.lower() or "time" in c.lower()]

                # Deterministic profiling & semantic interpretation
                dataset_display_name = sheet if sheet != "Sheet1" else os.path.splitext(original_filename)[0].replace("_", " ").title()
                sample_data = df.head(10).to_dict(orient="records") if len(df) > 0 else []
                profile = profile_dataset_columns(
                    columns=final_cols,
                    data_types=data_types,
                    raw_sample_data=sample_data,
                    row_count=row_count,
                    dataset_name=dataset_display_name,
                    sheet_name=sheet,
                    original_filename=original_filename,
                    workspace_id=workspace_id,
                )
                interpreter = get_semantic_interpreter()
                semantic_schema = interpreter.interpret_schema(profile, dataset_id=table_name)

                # Classify the dataset (backward-compatible)
                classification = dataset_classifier.classify(
                    columns=final_cols,
                    data_types=data_types,
                    numeric_measures=numeric_cols,
                    text_dimensions=text_cols,
                    date_fields=date_cols,
                    sheet_name=sheet,
                    original_filename=original_filename,
                    sample_rows_count=row_count,
                )

                dataset = Dataset(
                    dataset_id=table_name,
                    dataset_name=dataset_display_name,
                    sheet_name=sheet,
                    columns=final_cols,
                    data_types=data_types,
                    numeric_measures=numeric_cols,
                    text_dimensions=text_cols,
                    date_fields=date_cols,
                    row_count=row_count,
                    storage_reference=table_name,
                    parent_source_id=file_hash,
                    workspace_id=workspace_id,
                    original_filename=original_filename,
                    classification=classification,
                    semantic_schema=semantic_schema,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                datasets.append(dataset)

                tables_meta[table_name] = {
                    "original_sheet": sheet,
                    "columns": final_cols,
                    "numeric_measures": numeric_cols,
                    "text_dimensions": text_cols,
                    "row_count": row_count,
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "domain": semantic_schema.domain,
                    "purpose": semantic_schema.purpose,
                    "primary_entity": semantic_schema.primary_entity,
                }
        except Exception:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError:
                    pass
            raise
        finally:
            conn.close()

        if not datasets:
            raise ValueError(f"No readable tabular data found in '{original_filename}'.")

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

        # Generate rich searchable metadata and index into Typesense analytics catalog
        from .metadata_generator import metadata_generator
        from app.typesense_engine import get_typesense_client, upsert_analytics_metadata

        meta = metadata_generator.generate_metadata(source, document_id=source_id, use_llm=True)
        try:
            ts_client = get_typesense_client()
            upsert_analytics_metadata(ts_client, meta.model_dump())
            logger.info("Successfully indexed analytics metadata for source %s into Typesense", file_hash)
        except Exception as ts_err:
            logger.warning("Typesense analytics metadata indexing warning (non-fatal): %s", ts_err)

        msg = (
            f"Successfully ingested '{original_filename}' with {len(datasets)} sheets and {total_rows} rows."
            if not failed_sheets
            else f"Partially ingested '{original_filename}': {len(datasets)} sheets succeeded, {len(failed_sheets)} sheets failed."
        )

        return {
            "success": True,
            "message": msg,
            "file_id": file_hash,
            "source_id": file_hash,
            "filename": original_filename,
            "sheets": [d.storage_reference for d in datasets],
            "sheets_count": len(datasets),
            "failed_sheets": failed_sheets,
            "tables": tables_meta,
            "total_rows": total_rows,
            "schema_summary": schema_summary,
            "db_path": db_path,
            "metadata_catalog": meta.model_dump(),
        }

    def get_file_metadata(self, workspace_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves file metadata from DataSourceRegistry with workspace boundary enforcement."""
        src = self.registry.get_source(workspace_id, file_id)
        if src:
            return src.model_dump()
        return None

    def execute_sql_query(
        self,
        db_path: str,
        sql: str,
        timeout: Optional[float] = None,
        max_results: int = 50,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Executes a read-only SQL query against an isolated SQLite database file.
        Enforces read-only PRAGMA and query execution deadline via set_progress_handler.
        Returns (rows, column_names).
        Raises TimeoutError on query deadline exceeded.
        """
        effective_timeout = timeout if timeout is not None else self.query_timeout

        try:
            db_uri = f"{Path(os.path.abspath(db_path)).as_uri()}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        except Exception:
            conn = sqlite3.connect(db_path, timeout=5.0)

        conn.row_factory = sqlite3.Row

        deadline = time.monotonic() + effective_timeout
        timed_out = False

        def progress_handler() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        # Check every 1,000 SQLite VM opcodes
        conn.set_progress_handler(progress_handler, 1000)

        try:
            conn.execute("PRAGMA query_only = ON;")
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            col_names = list(results[0].keys()) if results else []
            rows = [dict(r) for r in results[:max_results]]
            return rows, col_names
        except sqlite3.OperationalError as e:
            if timed_out or "interrupted" in str(e).lower():
                raise TimeoutError(
                    f"The database query exceeded the execution time limit ({effective_timeout}s) and was safely stopped."
                ) from e
            raise
        finally:
            try:
                conn.set_progress_handler(None, 0)
            except Exception:
                pass
            conn.close()

    def query_excel(
        self,
        workspace_id: int,
        file_id: Optional[str] = None,
        question: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        currency_hint: Optional[str] = None,
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

        # ── 6. Execute against Sandboxed SQLite DB (Read-Only & Timeout Enforced) ─
        try:
            rows, col_names = self.execute_sql_query(
                db_path=db_path,
                sql=sql,
                timeout=self.query_timeout,
                max_results=50,
            )
        except TimeoutError as te:
            logger.warning(f"[ExcelDatabaseEngine] Query timed out after {self.query_timeout}s: {te}")
            return {
                "success": False,
                "report": f"⚠️ **Query Timeout Notice**: The database query exceeded the maximum execution time limit ({self.query_timeout}s) and was safely stopped.",
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }
        except Exception as e:
            logger.error(f"[ExcelDatabaseEngine] SQL Execution error: {e} for SQL: {sql}")
            return {
                "success": False,
                "report": "⚠️ **Query Execution Notice**: The query could not be executed due to a database syntax or execution error.",
                "rows": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        effective_currency = currency_hint
        if not effective_currency and target_source and getattr(target_source, "metadata", None):
            effective_currency = target_source.metadata.get("currency") or target_source.metadata.get("currency_hint")
        report = self._format_results(question, target_source.original_filename, rows, col_names, total_latency, currency_hint=effective_currency)

        return {
            "success": True,
            "report": report,
            "sql": sql,
            "rows": rows,
            "filename": target_source.original_filename,
            "source_id": target_source.source_id,
            "latency_ms": total_latency,
        }

    @staticmethod
    def _is_monetary_column(col: str) -> bool:
        """
        Determines if a column name represents a monetary currency value.
        Preserves non-monetary classifications (quantities, counts, ratings, IDs)
        while correctly identifying currency aliases and compound metrics.
        """
        lower = col.lower().strip()

        # 1. Non-monetary exclusions (counts, quantities, ratings, identifiers, dates, percentages, rates)
        non_monetary_indicators = (
            "count", "quantity", "qty", "units", "items", "orders",
            "rating", "score", "phone", "code", "id", "status",
            "ratio", "rate", "percentage", "percent", "pct", "days", "hours",
            "year", "month", "rank", "p_val", "index"
        )
        if any(non_money in lower for non_money in non_monetary_indicators):
            return False

        # 1b. Margin percentage exclusions (profit_margin, gross_margin, net_margin unless amount/val is explicit)
        margin_percentage_indicators = (
            "profit_margin", "gross_margin", "net_margin", "operating_margin",
            "ebitda_margin", "margin_rate", "margin_ratio", "margin_pct"
        )
        if any(m in lower for m in margin_percentage_indicators):
            if not any(amt in lower for amt in ("amount", "val", "value", "cash", "bdt", "usd")):
                return False

        # 2. Direct monetary keyword matches
        monetary_keywords = (
            "amount", "price", "cost", "salary", "spent", "fee", "charge",
            "revenue", "due", "cod", "tax", "vat", "discount", "cash",
            "payout", "budget", "valuation", "balance", "aov", "gmv",
            "sales", "profit", "margin", "worth"
        )
        if any(kw in lower for kw in monetary_keywords):
            return True

        # 3. Exact "total" or "grand_total" (without non-monetary suffix)
        if lower in ("total", "grand_total", "net_total", "sub_total", "subtotal"):
            return True

        # 4. Specific monetary value compounds (e.g., avg_order_val, order_value, cart_value, monetary_val)
        monetary_value_compounds = (
            "order_val", "order_value", "cart_val", "cart_value",
            "ticket_val", "ticket_value", "monetary_val", "monetary_value",
            "gross_val", "net_val", "total_val", "total_value", "market_value"
        )
        if any(compound in lower for compound in monetary_value_compounds):
            return True

        return False

    @staticmethod
    def _get_currency_symbol(col: str, default: str = "৳", currency_hint: Optional[str] = None) -> str:
        """
        Determines currency symbol based on:
        1. Explicit currency_hint / workspace metadata (highest precedence)
        2. Column name suffix/indicator (USD, EUR, GBP, INR, BDT)
        3. Default tenant currency fallback (e.g. '৳')
        """
        if currency_hint:
            hint = currency_hint.strip().upper()
            currency_map = {
                "USD": "$", "$": "$",
                "EUR": "€", "€": "€",
                "GBP": "£", "£": "£",
                "INR": "₹", "₹": "₹",
                "BDT": "৳", "৳": "৳", "TK": "৳", "TAKA": "৳"
            }
            if hint in currency_map:
                return currency_map[hint]
            return currency_hint

        lower = col.lower()
        if "usd" in lower or "dollar" in lower:
            return "$"
        if "eur" in lower or "euro" in lower:
            return "€"
        if "inr" in lower or "rupee" in lower:
            return "₹"
        if "gbp" in lower or "pound" in lower:
            return "£"
        return default

    def _format_results(
        self,
        question: str,
        filename: str,
        rows: List[Dict[str, Any]],
        columns: List[str],
        latency_ms: float,
        currency_hint: Optional[str] = None,
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
                    if self._is_monetary_column(col):
                        sym = self._get_currency_symbol(col, currency_hint=currency_hint)
                        lines.append(f"- **{col_label}:** {sym}{val:,.2f}" if val >= 0 else f"- **{col_label}:** -{sym}{abs(val):,.2f}")
                    else:
                        lines.append(f"- **{col_label}:** {val:,.2f}" if isinstance(val, float) else f"- **{col_label}:** {val:,}")
                else:
                    lines.append(f"- **{col_label}:** {val}")

            lines.append(f"\n*(Latency: {latency_ms}ms)*")
            return "\n".join(lines)

        headers = [c.replace("_", " ").title() for c in columns]
        lines = [
            f"📋 **Data Analysis Report** (`{filename}`)\n",
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
                    if self._is_monetary_column(col):
                        sym = self._get_currency_symbol(col, currency_hint=currency_hint)
                        row_vals.append(f"{sym}{val:,.2f}")
                    else:
                        row_vals.append(f"{val:,.2f}")
                elif isinstance(val, int) and self._is_monetary_column(col):
                    sym = self._get_currency_symbol(col, currency_hint=currency_hint)
                    row_vals.append(f"{sym}{val:,}")
                else:
                    row_vals.append(str(val))

            lines.append(f"| {' | '.join(row_vals)} |")

        row_count_str = f"Found: {len(rows)} rows" if len(rows) < 25 else "Showing first 25 rows"
        lines.append(f"\n*({row_count_str} | Latency: {latency_ms}ms)*")
        return "\n".join(lines)


# Global Singleton
excel_engine = ExcelDatabaseEngine()
