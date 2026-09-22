"""
Safe Read-Only MySQL Executor for Analytics.
Enforces read-only statement validation, query timeout, and strict parameterization.
"""

import os
import re
import time
from decimal import Decimal
from typing import List, Dict, Any, Tuple
import pymysql
import pymysql.cursors
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()


class SecurityViolationError(Exception):
    """Raised when an illegal or non-read-only SQL statement is detected."""
    pass


class AnalyticsExecutor:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.user = os.getenv("DB_USERNAME", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_DATABASE", "chatbot_db")
        self.timeout = 5  # seconds
        self.max_rows = 100

    def _validate_safety(self, sql: str) -> None:
        """
        Validates that the SQL statement is read-only and free of dangerous operations.
        """
        clean_sql = sql.strip().rstrip(";").strip()
        
        # Check for multiple statements
        if ";" in clean_sql:
            raise SecurityViolationError("Multi-statement queries are strictly prohibited.")

        # Must start with SELECT or WITH
        if not (clean_sql.upper().startswith("SELECT") or clean_sql.upper().startswith("WITH")):
            raise SecurityViolationError("Only SELECT / read-only queries are permitted.")

        # Prohibited DDL/DML keywords (whole word matching)
        forbidden = [
            r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
            r"\bALTER\b", r"\bTRUNCATE\b", r"\bREPLACE\b", r"\bGRANT\b",
            r"\bREVOKE\b", r"\bCREATE\b", r"\bEXEC\b", r"\bEXECUTE\b",
        ]
        for pattern in forbidden:
            if re.search(pattern, clean_sql, re.IGNORECASE):
                raise SecurityViolationError(f"Prohibited SQL operation detected: {pattern}")

    def execute(self, sql: str, params: List[Any] = None) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes a parameterized read-only SQL query safely.
        Returns (rows, execution_time_ms).
        """
        if not sql or not sql.strip():
            return [], 0.0

        self._validate_safety(sql)
        params = params or []

        # PyMySQL requires %s instead of ? for parameter placeholders
        pymysql_sql = sql.replace("?", "%s")

        start_time = time.perf_counter()

        connection = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=self.timeout,
            read_timeout=self.timeout,
            autocommit=True,
        )

        try:
            with connection.cursor() as cursor:
                cursor.execute(pymysql_sql, tuple(params))
                rows = cursor.fetchmany(self.max_rows)
        finally:
            connection.close()

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Sanitize decimals and dates for clean JSON serialization
        sanitized_rows = []
        for row in rows:
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, Decimal):
                    clean_row[k] = float(v)
                elif hasattr(v, "isoformat"):
                    clean_row[k] = v.isoformat()
                else:
                    clean_row[k] = v
            sanitized_rows.append(clean_row)

        return sanitized_rows, execution_time_ms
