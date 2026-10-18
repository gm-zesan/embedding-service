"""
Deterministic Dataset Classification Engine for E-Commerce / F-Commerce Analytics.
Infers business meaning (Products, Sales, Orders, Customers, Inventory, Payments, General Knowledge)
using column lexicon, inferred types, schema topology, and limited representative-value evidence
without requiring external LLM dependencies.
"""

import os
import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from .data_source import DatasetCategory, DatasetClassification

logger = logging.getLogger(__name__)


# Category lexicon mapping (normalized lower-case regex or tokens)
LEXICON_SIGNALS: Dict[DatasetCategory, Dict[str, Any]] = {
    DatasetCategory.PRODUCTS: {
        "primary_columns": [
            "sku", "product_id", "item_id", "product_name", "product_title",
            "item_name", "title", "unit_price", "selling_price", "mrp", "cost_price",
            "brand", "category", "sub_category", "description", "color", "size",
            "weight", "dimensions", "barcode", "upc", "specification",
            # Bengali / Banglish
            "পণ্য", "পণ্যের_নাম", "দাম", "মূল্য", "ব্র্যান্ড", "সাইজ", "রং",
            "ponno", "dam", "mullo", "item_nam", "product_nam"
        ],
        "strong_combos": [
            {"sku", "price"},
            {"sku", "unit_price"},
            {"product_name", "price"},
            {"item_name", "unit_price"},
            {"product_id", "price"},
            {"title", "price"},
        ],
        "negative_columns": [
            "order_id", "invoice_no", "sale_date", "order_date", "buyer", "customer_id",
            "transaction_id", "trx_id", "payment_method", "shipping_fee"
        ],
        "name_hints": ["product", "catalog", "item", "sku", "price_list", "পণ্য", "প্রোডাক্ট"]
    },
    DatasetCategory.SALES: {
        "primary_columns": [
            "sale_id", "sales_id", "invoice_no", "invoice_id", "invoice_number",
            "sale_date", "sales_date", "sold_at", "total_amount", "grand_total",
            "sales_amount", "revenue", "gross_sales", "net_sales", "sold_qty",
            "units_sold", "discount", "tax", "vat", "seller", "salesperson",
            # Bengali / Banglish
            "বিক্রয়", "বিক্রি", "মোট_টাকা", "মোট_বিক্রয়", "bikkroy", "mot_taka", "bikri"
        ],
        "strong_combos": [
            {"invoice_no", "total_amount"},
            {"sale_date", "revenue"},
            {"sold_qty", "total_amount"},
            {"sale_date", "total_amount"},
        ],
        "negative_columns": ["reorder_level", "bin_no", "warehouse_location"],
        "name_hints": ["sale", "sales", "revenue", "বিক্রয়", "বিক্রি", "সেলস"]
    },
    DatasetCategory.ORDERS: {
        "primary_columns": [
            "order_id", "order_no", "order_number", "order_date", "order_time",
            "customer_id", "buyer", "customer_name", "order_status", "delivery_status",
            "shipping_address", "shipping_city", "shipping_fee", "delivery_charge",
            "courier", "tracking_code", "courier_status", "delivery_date",
            # Bengali / Banglish
            "অর্ডার", "অর্ডার_আইডি", "অর্ডার_তারিখ", "ডেলিভারি", "order_id", "order_tarikh"
        ],
        "strong_combos": [
            {"order_id", "order_date"},
            {"order_id", "customer_id"},
            {"order_id", "order_status"},
            {"order_no", "delivery_status"},
        ],
        "negative_columns": ["bin_no", "warehouse_location", "reorder_level"],
        "name_hints": ["order", "orders", "delivery", "অর্ডার", "ডেলিভারি"]
    },
    DatasetCategory.INVENTORY: {
        "primary_columns": [
            "stock", "stock_qty", "stock_quantity", "current_stock", "available_qty",
            "available_stock", "quantity_in_stock", "qty_in_stock", "on_hand",
            "reorder_level", "reorder_point", "safety_stock", "warehouse",
            "warehouse_id", "location", "bin_no", "shelf", "damaged_qty", "restock_date",
            # Bengali / Banglish
            "মজুদ", "স্টক", "পরিমাণ", "গুদাম", "stok", "mojud", "poriman"
        ],
        "strong_combos": [
            {"sku", "stock"},
            {"sku", "stock_qty"},
            {"product_name", "available_qty"},
            {"sku", "reorder_level"},
            {"warehouse", "stock"},
        ],
        "negative_columns": ["invoice_no", "customer_id", "buyer", "payment_method", "trx_id"],
        "name_hints": ["inventory", "stock", "warehouse", "ইনভেন্টরি", "স্টক", "মজুদ"]
    },
    DatasetCategory.CUSTOMERS: {
        "primary_columns": [
            "customer_id", "client_id", "user_id", "customer_name", "client_name",
            "full_name", "first_name", "last_name", "phone", "mobile", "contact_no",
            "email", "address", "city", "district", "thana", "postal_code", "zip_code",
            "gender", "dob", "age", "loyalty_points", "tier",
            # Bengali / Banglish
            "গ্রাহক", "নাম", "ফোন", "মোবাইল", "ঠিকানা", "জেলা", "grahok", "thikana"
        ],
        "strong_combos": [
            {"customer_id", "phone"},
            {"customer_id", "email"},
            {"customer_name", "phone"},
            {"phone", "address"},
        ],
        "negative_columns": ["sku", "unit_price", "mrp", "reorder_level", "stock_qty"],
        "name_hints": ["customer", "customers", "client", "clients", "user", "গ্রাহক", "কাস্টমার"]
    },
    DatasetCategory.PAYMENTS: {
        "primary_columns": [
            "payment_id", "transaction_id", "trx_id", "tran_id", "payment_method",
            "gateway", "payment_status", "paid_amount", "due_amount", "paid_at",
            "card_type", "bank", "bkash", "nagad", "rocket", "currency",
            # Bengali / Banglish
            "পেমেন্ট", "লেনদেন", "পেমেন্ট_মেথড", "বিকাশ", "নগদ", "payment", "lenden"
        ],
        "strong_combos": [
            {"transaction_id", "paid_amount"},
            {"trx_id", "payment_method"},
            {"payment_id", "payment_status"},
        ],
        "negative_columns": ["sku", "reorder_level", "shelf"],
        "name_hints": ["payment", "payments", "transaction", "trx", "পেমেন্ট", "লেনদেন"]
    },
    DatasetCategory.GENERAL_KNOWLEDGE: {
        "primary_columns": [
            "faq", "question", "answer", "topic", "category", "policy", "guideline",
            "rule", "terms", "condition", "description", "details", "content",
            # Bengali / Banglish
            "প্রশ্ন", "উত্তর", "নীতিমালা", "নিয়ম", "proshno", "uttor"
        ],
        "strong_combos": [
            {"question", "answer"},
            {"topic", "content"},
            {"faq", "answer"},
        ],
        "negative_columns": ["sku", "unit_price", "order_id", "invoice_no", "stock_qty"],
        "name_hints": ["faq", "policy", "knowledge", "help", "guide", "প্রশ্ন", "নীতিমালা"]
    },
}


class DatasetClassifier:
    """
    Deterministic rule-based dataset classifier.
    Assigns business meaning, confidence score, and matched evidence to tabular datasets.
    """

    VERSION = "2.0.0"
    METHOD = "deterministic_heuristic_v2"

    def classify(
        self,
        columns: List[str],
        data_types: Optional[Dict[str, str]] = None,
        numeric_measures: Optional[List[str]] = None,
        text_dimensions: Optional[List[str]] = None,
        date_fields: Optional[List[str]] = None,
        sheet_name: Optional[str] = None,
        original_filename: Optional[str] = None,
        sample_rows_count: int = 0,
        inferred_types: Optional[Dict[str, str]] = None,
        filename: Optional[str] = None,
    ) -> DatasetClassification:
        """
        Classifies a tabular dataset into an e-commerce business category.
        """
        data_types = data_types or inferred_types or {}
        original_filename = original_filename or filename
        numeric_measures = numeric_measures or []
        text_dimensions = text_dimensions or []
        date_fields = date_fields or []

        if not columns:
            return DatasetClassification(
                category=DatasetCategory.UNKNOWN,
                confidence=0.0,
                method=self.METHOD,
                version=self.VERSION,
                status="unclassified",
                reason="Empty column list",
                classified_at=datetime.now(timezone.utc).isoformat(),
            )

        # Normalize column tokens
        norm_cols = [self._normalize_col(c) for c in columns]
        col_set = set(norm_cols)

        # Collect category scores
        category_scores: Dict[DatasetCategory, Tuple[float, List[str]]] = {}

        for cat, spec in LEXICON_SIGNALS.items():
            score, signals = self._score_category(
                cat=cat,
                spec=spec,
                norm_cols=norm_cols,
                col_set=col_set,
                data_types=data_types,
                numeric_measures=numeric_measures,
                date_fields=date_fields,
                sheet_name=sheet_name,
                original_filename=original_filename,
            )
            category_scores[cat] = (score, signals)

        # Disambiguation between Transactional (Sales/Orders) vs Static (Products/Inventory)
        self._apply_structural_disambiguation(category_scores, col_set, date_fields)

        # Sort by score descending
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1][0], reverse=True)
        top_cat, (top_score, top_signals) = sorted_cats[0]

        # Confidence normalization: min 0.0, max 1.0
        normalized_confidence = round(min(1.0, max(0.0, top_score / 10.0)), 2)

        # Thresholds:
        # > 0.50 -> Clear match
        # 0.35 - 0.50 -> Ambiguous / low confidence
        # < 0.35 -> UNKNOWN
        if normalized_confidence < 0.35:
            return DatasetClassification(
                category=DatasetCategory.UNKNOWN,
                confidence=normalized_confidence,
                method=self.METHOD,
                version=self.VERSION,
                status="unclassified",
                matched_signals=top_signals,
                reason=f"Insufficient matching evidence (top score: {top_score:.1f} for {top_cat.value})",
                classified_at=datetime.now(timezone.utc).isoformat(),
            )

        # Check for tie with second category
        if len(sorted_cats) > 1:
            second_cat, (second_score, second_signals) = sorted_cats[1]
            second_conf = round(min(1.0, max(0.0, second_score / 10.0)), 2)
            # If the top two categories are virtually identical and confidence is moderate, mark ambiguous
            if abs(normalized_confidence - second_conf) <= 0.05 and normalized_confidence < 0.65:
                # Exception: SALES and ORDERS are sibling transaction categories
                if {top_cat, second_cat} != {DatasetCategory.SALES, DatasetCategory.ORDERS}:
                    return DatasetClassification(
                        category=DatasetCategory.OTHER,
                        confidence=normalized_confidence,
                        method=self.METHOD,
                        version=self.VERSION,
                        status="ambiguous",
                        matched_signals=top_signals + [f"tied_with_{second_cat.value}"],
                        reason=f"Ambiguous category match between {top_cat.value} and {second_cat.value}",
                        classified_at=datetime.now(timezone.utc).isoformat(),
                    )

        reason = f"Classified as {top_cat.value.upper()} with evidence: {', '.join(top_signals[:5])}"

        return DatasetClassification(
            category=top_cat,
            confidence=normalized_confidence,
            method=self.METHOD,
            version=self.VERSION,
            status="classified",
            matched_signals=top_signals,
            reason=reason,
            classified_at=datetime.now(timezone.utc).isoformat(),
        )

    def _normalize_col(self, col: str) -> str:
        clean = col.lower().strip()
        clean = re.sub(r"[^\w\u0980-\u09FF]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean

    def _score_category(
        self,
        cat: DatasetCategory,
        spec: Dict[str, Any],
        norm_cols: List[str],
        col_set: set,
        data_types: Dict[str, str],
        numeric_measures: List[str],
        date_fields: List[str],
        sheet_name: Optional[str],
        original_filename: Optional[str],
    ) -> Tuple[float, List[str]]:
        score = 0.0
        signals = []

        primary_cols = spec.get("primary_columns", [])
        strong_combos = spec.get("strong_combos", [])
        negative_cols = spec.get("negative_columns", [])
        name_hints = spec.get("name_hints", [])

        # 1. Primary column matches
        matched_primary = []
        for c in norm_cols:
            for p in primary_cols:
                if c == p or c.endswith(f"_{p}") or c.startswith(f"{p}_"):
                    matched_primary.append(c)
                    score += 2.0
                    signals.append(f"col:{c}")
                    break

        # 2. Strong combination matches
        for combo in strong_combos:
            combo_matched = True
            for required_token in combo:
                if not any(required_token in c for c in norm_cols):
                    combo_matched = False
                    break
            if combo_matched:
                score += 3.5
                signals.append(f"combo:{'+'.join(combo)}")

        # 3. Negative indicators
        for neg in negative_cols:
            for c in norm_cols:
                if c == neg or c.endswith(f"_{neg}") or c.startswith(f"{neg}_"):
                    score -= 2.0
                    signals.append(f"neg_col:{c}")
                    break

        # 4. Sheet name and Filename hints (contributory only, max 2.0 points)
        name_targets = []
        if sheet_name:
            name_targets.append(sheet_name.lower())
        if original_filename:
            stem = os.path.splitext(original_filename.lower())[0]
            name_targets.append(stem)

        for target in name_targets:
            for hint in name_hints:
                if hint in target:
                    score += 1.5
                    signals.append(f"name_hint:{hint}")
                    break

        # 5. Type-based boosters
        if cat in (DatasetCategory.SALES, DatasetCategory.ORDERS):
            if date_fields:
                score += 1.5
                signals.append("has_date_field")
            if numeric_measures:
                score += 1.0

        if cat == DatasetCategory.PRODUCTS:
            if not date_fields:
                score += 1.0
            if any("price" in c or "mrp" in c or "cost" in c for c in norm_cols):
                score += 1.5

        if cat == DatasetCategory.INVENTORY:
            if any("stock" in c or "qty" in c or "quantity" in c for c in norm_cols):
                score += 1.5

        return max(0.0, score), signals

    def _apply_structural_disambiguation(
        self,
        category_scores: Dict[DatasetCategory, Tuple[float, List[str]]],
        col_set: set,
        date_fields: List[str],
    ) -> None:
        """
        Disambiguates between transactional order/sales tables and static product catalogs or inventory snapshots.
        """
        has_transaction_id = any(
            re.search(r"\b(order_id|invoice_no|order_no|sale_id|invoice_id)\b", c) for c in col_set
        )
        has_dates = bool(date_fields)
        has_buyer = any(re.search(r"\b(customer_id|buyer|customer_name|client_id)\b", c) for c in col_set)

        # If it has transaction IDs and dates, penalize static Product catalog
        if has_transaction_id or (has_dates and has_buyer):
            prod_score, prod_sig = category_scores.get(DatasetCategory.PRODUCTS, (0.0, []))
            category_scores[DatasetCategory.PRODUCTS] = (
                max(0.0, prod_score - 4.0),
                prod_sig + ["demoted_by_transaction_ids"],
            )

        # If it has warehouse / bin / reorder and no customer/order, boost Inventory
        has_warehouse = any(re.search(r"\b(warehouse|bin_no|shelf|reorder_level|reorder_point)\b", c) for c in col_set)
        if has_warehouse and not has_transaction_id:
            inv_score, inv_sig = category_scores.get(DatasetCategory.INVENTORY, (0.0, []))
            category_scores[DatasetCategory.INVENTORY] = (
                inv_score + 3.0,
                inv_sig + ["boosted_by_warehouse_topology"],
            )

    classify_dataset = classify


# Global Singleton Classifier
dataset_classifier = DatasetClassifier()
