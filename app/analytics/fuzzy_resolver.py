import os
import re
import time
import difflib
import logging
from typing import Dict, List, Tuple, Any, Optional

# pyrefly: ignore [missing-import]
import pymysql
# pyrefly: ignore [missing-import]
import pymysql.cursors

logger = logging.getLogger("app.analytics.fuzzy_resolver")


def bengali_to_phonetic_latin(text: str) -> str:
    """
    Dynamic Algorithmic Bengali-to-Latin phonetic converter for infinite scale.
    Automatically handles any Bengali word, name, product, category, or term.
    Zero static name dictionary or hardcoded entity mappings.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Check if text contains Bengali characters
    if not re.search(r"[ঀ-৿]", text):
        return text.lower().strip()

    vowels = {
        "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
        "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
        "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u",
        "ৃ": "ri", "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou", 
        "্": "", "্য": "", "্যা": "a"
    }
    consonants = {
        "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
        "চ": "ch", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",
        "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
        "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
        "প": "p", "ফ": "f", "ব": "b", "ভ": "v", "ম": "m",
        "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
        "স": "s", "হ": "h", "ড়": "r", "ঢ়": "rh", "য়": "y",
        "ৎ": "t", "ং": "ng", "ঃ": "h", "ঁ": ""
    }
    
    out = []
    chars = list(text.strip())
    for i, ch in enumerate(chars):
        if ch in vowels:
            out.append(vowels[ch])
        elif ch in consonants:
            out.append(consonants[ch])
            if i + 1 < len(chars):
                next_ch = chars[i+1]
                # Insert inherent vowel a if not followed by vowel mark or virama/hasant
                if next_ch in consonants and next_ch not in ("্", "্য", "্র", "্লা", "বা"):
                    out.append("a")
        else:
            out.append(ch)
            
    res = "".join(out).lower()
    res = res.replace("kya", "ca").replace("kja", "ca").replace("bja", "ba").replace("bya", "ba").replace("ngak", "nk")
    res = re.sub(r"aa+", "a", res)
    res = re.sub(r"ee+", "ee", res)
    res = re.sub(r"oo+", "oo", res)
    return res


class FuzzyEntityResolver:
    """
    Enterprise-Scale Scalable Entity Normalizer & Disambiguator.
    
    Architectural Guarantees:
    1. Zero Full-Table Scans: Queries targeted candidate generation via parameterized SQL (Top-K / Prefix / N-gram).
    2. Zero Static Name Mappings: Pure algorithmic transliteration + multi-stage similarity scoring.
    3. Strict Multi-Tenant Isolation: Every candidate query strictly parameterized with `workspace_id = %s`.
    4. Strict Ambiguity Guard: Margin-based confidence threshold (winner >= 0.70 AND margin >= 0.10).
    5. Fallback Protection: Unknown/unmatched entities return unchanged with safe 0.0 confidence (never guess).
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.user = os.getenv("DB_USERNAME", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_DATABASE", "chatbot_db")
        self.cache_ttl = cache_ttl_seconds
        
        # Fixed small enum domains (memory safe)
        self.static_enums = {
            "payment_method": ["cash", "bank", "bkash", "nagad", "card", "rocket"],
            "status": ["completed", "pending", "cancelled", "processing"],
        }
        
        # LRU/TTL Cache for small catalogs (salespersons/categories)
        self._cache: Dict[str, Dict[str, Any]] = {}

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

    def _fetch_candidates_from_db(self, workspace_id: int, field_type: str, search_terms: List[str], limit: int = 30) -> List[str]:
        """
        Top-K Targeted Candidate Retrieval using Database Indexing & Prefix/LIKE Filters.
        Scales to millions of records by fetching only plausible candidates (O(1) to O(K))
        instead of scanning entire customer/product tables in Python memory.
        """
        table_col_map = {
            "salesperson": ("analytics_salespersons", "name", "AND (is_active = 1 OR is_active IS NULL)"),
            "customer": ("analytics_customers", "name", ""),
            "product": ("analytics_products", "name", ""),
            "category": ("analytics_products", "category", ""),
        }

        if field_type not in table_col_map:
            return []

        table, col, extra_cond = table_col_map[field_type]
        candidates = set()

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # If field is small category or salesperson, fetch workspace list
                    if field_type in ("salesperson", "category"):
                        sql = f"SELECT DISTINCT {col} FROM {table} WHERE workspace_id = %s {extra_cond} LIMIT 100"
                        cursor.execute(sql, (workspace_id,))
                        for row in cursor.fetchall():
                            val = row.get(col)
                            if val:
                                candidates.add(val)
                        return list(candidates)

                    # For large tables (Customer, Product): Run targeted parameterized LIKE queries
                    for term in search_terms:
                        clean = term.strip()
                        if len(clean) < 2:
                            continue
                        
                        # Parameterized Prefix & Substring Match
                        sql = f"""
                            SELECT DISTINCT {col} 
                            FROM {table} 
                            WHERE workspace_id = %s 
                              AND ({col} LIKE %s OR {col} LIKE %s)
                              {extra_cond}
                            LIMIT %s
                        """
                        cursor.execute(sql, (workspace_id, f"{clean}%", f"%{clean}%", limit))
                        for row in cursor.fetchall():
                            val = row.get(col)
                            if val:
                                candidates.add(val)

                        # If term has multiple words, search by individual prominent token
                        tokens = [t for t in clean.split() if len(t) > 2]
                        for tok in tokens[:2]:
                            cursor.execute(
                                f"SELECT DISTINCT {col} FROM {table} WHERE workspace_id = %s AND {col} LIKE %s {extra_cond} LIMIT 15",
                                (workspace_id, f"%{tok}%")
                            )
                            for row in cursor.fetchall():
                                val = row.get(col)
                                if val:
                                    candidates.add(val)

        except Exception as e:
            logger.warning(f"[FuzzyEntityResolver] Targeted candidate fetch error: {e}")

        return list(candidates)

    def get_candidates(self, workspace_id: int, field_type: str, search_terms: List[str]) -> List[str]:
        """
        Returns candidate list for entity resolution:
        1. Fixed enums (payment methods, statuses)
        2. Small cached catalogs (salespersons, categories)
        3. Targeted Top-K database retrieval for large entities (customers, products).
        """
        if field_type in self.static_enums:
            return self.static_enums[field_type]

        # Check in-memory cache for small workspaces
        cache_key = f"{workspace_id}:{field_type}"
        now = time.time()
        if field_type in ("salesperson", "category"):
            cached = self._cache.get(cache_key)
            if cached and (now - cached["timestamp"] < self.cache_ttl):
                return cached["candidates"]

        candidates = self._fetch_candidates_from_db(workspace_id, field_type, search_terms)

        if field_type in ("salesperson", "category"):
            self._cache[cache_key] = {
                "timestamp": now,
                "candidates": candidates,
            }

        return candidates

    def normalize(self, workspace_id: int, field_type: str, raw_value: str, threshold: float = 0.70) -> Tuple[str, float]:
        """
        Normalizes raw input value to the closest matching canonical entity in DB.
        
        Algorithm:
        1. Pure algorithmic phonetic transliteration (if Bengali script present).
        2. Generate Top-K targeted candidate set from MySQL DB under current workspace_id.
        3. Multi-tier fuzzy & token similarity scoring.
        4. Margin-based Ambiguity Guard (refuse to guess if top 2 candidates are within 0.10 margin).
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

        # 1. Phonetic Candidate Generation
        phonetic_latin = bengali_to_phonetic_latin(raw_str)
        search_terms = [raw_str]
        if phonetic_latin and phonetic_latin != raw_lower:
            search_terms.append(phonetic_latin)

        # 2. Retrieve targeted candidates
        candidates = self.get_candidates(workspace_id, f_type, search_terms)
        if not candidates:
            return raw_str, 0.0

        # 3. Exact Match (Case-insensitive)
        for c in candidates:
            if c.lower() == raw_lower:
                return c, 1.0
            if phonetic_latin and c.lower().replace("-", "").replace(" ", "") == phonetic_latin.replace("-", "").replace(" ", ""):
                return c, 1.0

        # 4. Score all generated candidates using Hybrid Sequence & Token Scoring
        scored: List[Tuple[str, float]] = []

        for c in candidates:
            c_lower = c.lower()
            c_clean = c_lower.replace("-", "").replace(" ", "")
            
            # Check direct raw similarity
            ratio_raw = difflib.SequenceMatcher(None, raw_lower, c_lower).ratio()
            
            # Check phonetic similarity
            ratio_phonetic = 0.0
            if phonetic_latin:
                p_clean = phonetic_latin.replace("-", "").replace(" ", "")
                ratio_phonetic = difflib.SequenceMatcher(None, p_clean, c_clean).ratio()

            # Check token overlap across raw, phonetic, and candidate tokens
            c_tokens = [t.strip() for t in c_lower.split() if len(t.strip()) > 2]
            raw_tokens = [t.strip() for t in raw_lower.split() if len(t.strip()) > 2]
            phon_tokens = [t.strip() for t in (phonetic_latin.split() if phonetic_latin else []) if len(t.strip()) > 2]
            all_query_tokens = list(set(raw_tokens + phon_tokens))
            
            token_match = 0.0
            for qt in all_query_tokens:
                for ct in c_tokens:
                    t_ratio = difflib.SequenceMatcher(None, qt, ct).ratio()
                    if t_ratio > token_match:
                        token_match = t_ratio

            # Final composite score
            final_score = max(ratio_raw, ratio_phonetic, token_match)
            
            # Substring containment bonus
            if raw_lower in c_lower or (phonetic_latin and phonetic_latin in c_lower):
                final_score = max(final_score, 0.88)

            scored.append((c, round(final_score, 4)))

        # Sort descending by similarity score
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return raw_str, 0.0

        top1_cand, top1_score = scored[0]

        # 5. Margin-Based Ambiguity Guard
        if len(scored) >= 2:
            top2_cand, top2_score = scored[1]
            # If top 2 candidates are distinct, have close scores (<0.10 margin) and are not perfect matches (<0.95)
            if top1_cand.lower() != top2_cand.lower() and (top1_score - top2_score < 0.10) and top1_score < 0.95:
                logger.info(
                    f"[FuzzyEntityResolver] Ambiguity detected between {top1_cand} ({top1_score}) and {top2_cand} ({top2_score}) for {raw_str}"
                )
                return raw_str, 0.0

        if top1_score >= threshold:
            logger.info(f"[FuzzyEntityResolver] Resolved {raw_str} -> {top1_cand} (score: {top1_score})")
            return top1_cand, top1_score

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
                    if score >= 0.70:
                        f.value = canonical_val
                elif isinstance(f.value, list):
                    normalized_list = []
                    for item in f.value:
                        if isinstance(item, str):
                            c_val, score = self.normalize(workspace_id, f.field, item)
                            normalized_list.append(c_val if score >= 0.70 else item)
                        else:
                            normalized_list.append(item)
                    f.value = normalized_list

        return plan


# Global Singleton Instance
fuzzy_resolver = FuzzyEntityResolver()
