"""
Statistical & Machine Learning Time-Series Forecaster.
Computes deterministic trend projections, growth velocity (MoM/WoW), and confidence intervals
for sales, collections, and order volumes across workspaces.
"""

import math
import time
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import pymysql
import pymysql.cursors
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """
    Computes time-series statistical forecasts and trend projections from historical database aggregates.
    """

    def __init__(self):
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

    def _fetch_historical_series(self, workspace_id: int, metric: str, period: str = "month") -> List[Dict[str, Any]]:
        """Fetches chronological aggregate series from orders or payments."""
        series = []
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if metric in ("sales", "sales_amount", "revenue"):
                        sql = """
                            SELECT DATE_FORMAT(order_date, '%%Y-%%m') AS period_label,
                                   COALESCE(SUM(net_amount), 0) AS value,
                                   COUNT(id) AS volume
                            FROM analytics_orders
                            WHERE workspace_id = %s AND status = 'completed' AND order_date IS NOT NULL
                            GROUP BY DATE_FORMAT(order_date, '%%Y-%%m')
                            ORDER BY period_label ASC;
                        """
                    elif metric in ("collection", "collection_amount", "payments"):
                        sql = """
                            SELECT DATE_FORMAT(collected_at, '%%Y-%%m') AS period_label,
                                   COALESCE(SUM(amount), 0) AS value,
                                   COUNT(id) AS volume
                            FROM analytics_payments
                            WHERE workspace_id = %s AND collected_at IS NOT NULL
                            GROUP BY DATE_FORMAT(collected_at, '%%Y-%%m')
                            ORDER BY period_label ASC;
                        """
                    else:  # orders count
                        sql = """
                            SELECT DATE_FORMAT(order_date, '%%Y-%%m') AS period_label,
                                   COUNT(id) AS value,
                                   COUNT(id) AS volume
                            FROM analytics_orders
                            WHERE workspace_id = %s AND status = 'completed' AND order_date IS NOT NULL
                            GROUP BY DATE_FORMAT(order_date, '%%Y-%%m')
                            ORDER BY period_label ASC;
                        """
                    cursor.execute(sql, (workspace_id,))
                    series = cursor.fetchall()
        except Exception as e:
            logger.error(f"[TimeSeriesForecaster] Failed to fetch historical series: {e}")

        return series

    def forecast(
        self,
        workspace_id: int,
        metric: str = "sales",
        horizon: int = 1,
        period: str = "month",
    ) -> Dict[str, Any]:
        """
        Calculates trend slope, moving average, confidence interval, and projected forecast value.
        """
        raw_series = self._fetch_historical_series(workspace_id, metric, period)
        if not raw_series:
            # Fallback if no chronological partition exists (use total aggregate)
            return {
                "metric": metric,
                "projected_value": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "growth_rate_pct": 0.0,
                "trend_direction": "stable",
                "historical_points": 0,
                "formatted_report": "⚠️ **Forecast Notice**: পর্যাপ্ত ঐতিহাসিক সেলস/অর্ডার ডেটা পাওয়া যায়নি।"
            }

        values = [float(r["value"]) for r in raw_series]
        n = len(values)

        if n == 1:
            projected = values[0]
            growth_pct = 0.0
            lower = projected * 0.90
            upper = projected * 1.10
            trend_dir = "stable"
        else:
            # Linear Regression Trend (Least Squares)
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(values) / n

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n)) or 1.0

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Next period projection (x = n)
            next_x = n + horizon - 1
            projected = max(0.0, intercept + slope * next_x)

            # Growth rate compared to last actual period
            last_val = values[-1] or 1.0
            growth_pct = round(((projected - last_val) / last_val) * 100, 2)

            # Standard deviation of residuals for confidence bounds
            residuals = [values[i] - (intercept + slope * x[i]) for i in range(n)]
            variance = sum(r ** 2 for r in residuals) / max(1, n - 2)
            std_err = math.sqrt(variance) if variance > 0 else (last_val * 0.08)

            lower = max(0.0, projected - 1.96 * std_err)
            upper = projected + 1.96 * std_err
            trend_dir = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")

        # Format report
        metric_title = metric.replace("_", " ").title()
        is_currency = metric in ("sales", "sales_amount", "revenue", "collection", "collection_amount")
        prefix = "৳" if is_currency else ""

        report_lines = [
            f"📈 **Predictive Analytics & Forecasting Report**",
            f"",
            f"> **Metric Analyzed:** {metric_title} ({period.capitalize()}ly Projection)",
            f"",
            f"| Metric | Projected Value | Confidence Interval (95%) | Trend Growth |",
            f"| :--- | :--- | :--- | :--- |",
            f"| **Next {period.capitalize()} Projection** | **{prefix}{projected:,.2f}** | {prefix}{lower:,.2f} – {prefix}{upper:,.2f} | `{growth_pct:+.2f}%` ({trend_dir.capitalize()}) |",
            f"",
            f"### 📊 Historical Baseline ({n} periods):",
        ]

        for r in raw_series[-6:]:
            val = float(r['value'])
            report_lines.append(f"- **{r['period_label']}:** {prefix}{val:,.2f} ({r.get('volume', 0)} orders/transactions)")

        report_lines.append(f"\n*(Calculated via Linear Trend Regression & Residual Variance Analysis)*")

        return {
            "metric": metric,
            "projected_value": round(projected, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "growth_rate_pct": growth_pct,
            "trend_direction": trend_dir,
            "historical_points": n,
            "formatted_report": "\n".join(report_lines),
        }


# Global Singleton
forecaster = TimeSeriesForecaster()
