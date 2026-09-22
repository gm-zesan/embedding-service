"""
Analytics Report Formatter.
Translates structured query results and plan metadata into clean, screenshot-ready Markdown reports.
"""

from typing import List, Dict, Any
from app.analytics.models import SemanticQueryPlan


class AnalyticsFormatter:
    """
    Formats analytical results into professional, human-readable reports and Markdown tables.
    """

    @staticmethod
    def format_currency(amount: float) -> str:
        return f"৳{amount:,.2f}"

    def format(
        self,
        question: str,
        plan: SemanticQueryPlan,
        results: List[Dict[str, Any]],
        latency_ms: float = 0.0,
    ) -> str:
        # 1. Security Rejection Notice
        if plan.is_security_rejection:
            reason = plan.rejection_reason or "Request cannot be fulfilled."
            return (
                f"🛡️ **Security / Policy Notice**\n\n"
                f"> **Query:** `{question}`\n"
                f"> **Status:** REJECTED\n"
                f"> **Reason:** {reason}\n"
            )

        # 2. Clarification Required Notice
        if plan.needs_clarification:
            msg = plan.clarification_message or "আপনার প্রশ্নটি নির্দিষ্ট করুন:"
            options_str = "\n".join([f"- **{opt.capitalize()}**" for opt in plan.clarification_options])
            return (
                f"❓ **Clarification Required**\n\n"
                f"{msg}\n\n"
                f"**বিকল্পসমূহ:**\n{options_str}\n"
            )

        # 3. Empty Results
        if not results:
            return (
                f"📊 **Business Analytics Report**\n\n"
                f"উক্ত অনুসন্ধানের জন্য কোনো রেকর্ড খুঁজে পাওয়া যায়নি।\n"
                f"*(Latency: {latency_ms:.1f}ms)*"
            )

        # 4. Single Value Summary (e.g., Total Cash-in, Total Due, Total Orders)
        first_row = results[0]
        if len(results) == 1 and len(first_row) <= 2:
            # Check for currency keys
            output_lines = [f"📊 **Business Analytics Summary**\n"]
            for k, v in first_row.items():
                label = k.replace("_", " ").title()
                if isinstance(v, (int, float)) and ("amount" in k or "total" in k or "sales" in k or "due" in k or "collected" in k or "cash" in k):
                    output_lines.append(f"- **{label}:** {self.format_currency(float(v))}")
                else:
                    output_lines.append(f"- **{label}:** {v}")
            output_lines.append(f"\n*(Latency: {latency_ms:.1f}ms)*")
            return "\n".join(output_lines)

        # 5. Tabular / Multi-row Report (e.g., Customer Due List, Salesperson Ranking)
        headers = list(first_row.keys())
        header_row = "| " + " | ".join([h.replace("_", " ").title() for h in headers]) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = []
        for r in results:
            row_vals = []
            for h in headers:
                val = r.get(h, "")
                if isinstance(val, (int, float)) and ("amount" in h or "total" in h or "sales" in h or "due" in h or "collected" in h):
                    row_vals.append(self.format_currency(float(val)))
                else:
                    row_vals.append(str(val))
            rows.append("| " + " | ".join(row_vals) + " |")

        table_str = "\n".join([header_row, sep_row] + rows)
        return (
            f"📋 **Business Analytics Report**\n\n"
            f"> **Question:** {question}\n\n"
            f"{table_str}\n\n"
            f"*(Found: {len(results)} rows | Latency: {latency_ms:.1f}ms)*"
        )
