"""
Fuzzy Entity Resolver & Normalizer.
Resolves misspelled, partial, or phonetic entity names (salespersons, customers, products, categories, payment methods)
to their canonical database values using token similarity, substring containment, and SequenceMatcher.
"""

import time
import difflib
import logging
from typing import Dict, List, Optional, Tuple, Any
import pymysql
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class FuzzyEntityResolver:
    """
    High-performance in-memory cached fuzzy entity resolver.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl = cache_ttl_seconds
        # Structure: {workspace_id: {"timestamp": float, "entities": {entity_type: [canonical_names]}}}
        self._cache: Dict[int, Dict[str, Any]] = {}
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.user = os.getenv("DB_USERNAME", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_DATABASE", "chatbot_db")

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

    def _load_workspace_entities(self, workspace_id: int) -> Dict[str, List[str]]:
        """Loads canonical entity lists from MySQL for the workspace."""
        entities: Dict[str, List[str]] = {
            "salesperson": [],
            "customer": [],
            "product": [],
            "category": [],
            "payment_method": ["cash", "bank", "bkash", "nagad", "card", "rocket"],
            "status": ["completed", "pending", "cancelled", "processing"],
        }

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Salespersons
                    cursor.execute(
                        "SELECT name FROM analytics_salespersons WHERE workspace_id = %s AND (is_active = 1 OR is_active IS NULL)",
                        (workspace_id,),
                    )
                    entities["salesperson"] = [row["name"] for row in cursor.fetchall() if row.get("name")]

                    # 2. Customers
                    cursor.execute(
                        "SELECT name FROM analytics_customers WHERE workspace_id = %s",
                        (workspace_id,),
                    )
                    entities["customer"] = [row["name"] for row in cursor.fetchall() if row.get("name")]

                    # 3. Products & Categories
                    cursor.execute(
                        "SELECT name, category FROM analytics_products WHERE workspace_id = %s",
                        (workspace_id,),
                    )
                    rows = cursor.fetchall()
                    entities["product"] = [r["name"] for r in rows if r.get("name")]
                    entities["category"] = list(set(r["category"] for r in rows if r.get("category")))

        except Exception as e:
            logger.warning(f"[FuzzyEntityResolver] Could not load entity catalog from DB: {e}")

        return entities

    def get_entities(self, workspace_id: int) -> Dict[str, List[str]]:
        now = time.time()
        cached = self._cache.get(workspace_id)
        if cached and (now - cached["timestamp"] < self.cache_ttl):
            return cached["entities"]

        entities = self._load_workspace_entities(workspace_id)
        self._cache[workspace_id] = {
            "timestamp": now,
            "entities": entities,
        }
        return entities

    def normalize(self, workspace_id: int, field_type: str, raw_value: str, threshold: float = 0.55) -> Tuple[str, float]:
        """
        Normalizes raw input value to the closest matching canonical entity in DB.
        Returns: (canonical_name, similarity_score)
        """
        if not raw_value or not isinstance(raw_value, str):
            return raw_value, 1.0

        raw_str = raw_value.strip()
        raw_lower = raw_str.lower()

        # Map field aliases
        f_type = field_type.lower()
        if f_type in ("collector", "assigned_collector", "salesperson_name", "salesman", "sales_rep"):
            f_type = "salesperson"
        elif f_type in ("client", "customer_name"):
            f_type = "customer"
        elif f_type in ("item", "product_name"):
            f_type = "product"
        elif f_type in ("method", "payment"):
            f_type = "payment_method"

        all_entities = self.get_entities(workspace_id)
        candidates = all_entities.get(f_type, [])

        if not candidates:
            return raw_str, 1.0

        # 1. Exact match (case-insensitive)
        for c in candidates:
            if c.lower() == raw_lower:
                return c, 1.0

        # 2. Substring / Token containment
        for c in candidates:
            c_lower = c.lower()
            if raw_lower in c_lower or c_lower in raw_lower:
                return c, 0.90

        # 3. Token-level SequenceMatcher & Word Similarity
        best_candidate = None
        best_score = 0.0

        for c in candidates:
            c_lower = c.lower()
            # Full string comparison
            full_ratio = difflib.SequenceMatcher(None, raw_lower, c_lower).ratio()
            if full_ratio > best_score:
                best_score = full_ratio
                best_candidate = c

            # Token-to-token comparison for multi-word products/names
            c_tokens = [t.strip() for t in c_lower.split() if len(t.strip()) > 2]
            raw_tokens = [t.strip() for t in raw_lower.split() if len(t.strip()) > 2] or [raw_lower]

            for rt in raw_tokens:
                for ct in c_tokens:
                    token_ratio = difflib.SequenceMatcher(None, rt, ct).ratio()
                    if token_ratio > best_score:
                        best_score = token_ratio
                        best_candidate = c

        if best_candidate and best_score >= threshold:
            logger.info(f"[FuzzyEntityResolver] Fuzzy resolved '{raw_str}' -> '{best_candidate}' (score: {best_score:.2f})")
            return best_candidate, best_score

        return raw_str, 0.0

    def normalize_plan(self, plan: Any, workspace_id: int) -> Any:
        """
        Iterates over all filters in a SemanticQueryPlan and normalizes entity values.
        """
        if not hasattr(plan, "filters") or not plan.filters:
            return plan

        for f in plan.filters:
            if f.field in ("salesperson", "collector", "assigned_collector", "customer", "product", "category", "payment_method", "status"):
                if isinstance(f.value, str):
                    canonical_val, score = self.normalize(workspace_id, f.field, f.value)
                    if score >= 0.55:
                        f.value = canonical_val
                elif isinstance(f.value, list):
                    normalized_list = []
                    for item in f.value:
                        if isinstance(item, str):
                            c_val, score = self.normalize(workspace_id, f.field, item)
                            normalized_list.append(c_val if score >= 0.55 else item)
                        else:
                            normalized_list.append(item)
                    f.value = normalized_list

        return plan


# Global Singleton Instance
fuzzy_resolver = FuzzyEntityResolver()
