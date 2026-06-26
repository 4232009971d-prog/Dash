"""
InsightFlow - AI Insights Engine
Generates natural-language business insights from KPIs and dataset statistics.
Works in two modes:
  1. Rule-based (always available) - fast, deterministic insights
  2. LLM-powered (when OpenAI key present) - richer, contextual narrative
"""

import logging
from dataclasses import dataclass

import pandas as pd



from header_detection.detector import DetectionReport, get_kpi_columns
from services.ingestion import DataQualityReport

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    text: str
    category: str   # "revenue" | "product" | "customer" | "quality" | "alert"
    importance: str  # "high" | "medium" | "low"
    icon: str = "💡"


class InsightsEngine:
    """
    Generates automatic business insights from cleaned data and KPIs.
    """

    def generate(
        self,
        df: pd.DataFrame,
        detection: DetectionReport,
        kpis: list[KPI],
        quality: DataQualityReport,
    ) -> list[Insight]:
        insights: list[Insight] = []

        insights += self._revenue_insights(df, detection, kpis)
        insights += self._product_insights(df, detection)
        insights += self._customer_insights(df, detection, kpis)
        insights += self._quality_insights(quality)
        insights += self._trend_insights(df, detection)

        # Sort: high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: order[x.importance])
        logger.info("Generated %d insights.", len(insights))
        return insights

    # ── Revenue ────────────────────────────────────────────────────────────────

    def _revenue_insights(
        self, df: pd.DataFrame, detection: DetectionReport, kpis: list[KPI]
    ) -> list[Insight]:
        insights = []
        groups = get_kpi_columns(detection)
        rev_cols = groups["revenue"]
        date_cols = groups["date"]

        if not rev_cols:
            return []

        rev_col = rev_cols[0]
        rev = pd.to_numeric(df[rev_col], errors="coerce").dropna()

        if rev.empty:
            return []

        total = rev.sum()

        # Month-over-month growth
        if date_cols:
            date_col = date_cols[0]
            df_t = df[[date_col, rev_col]].copy()
            df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
            df_t[rev_col] = pd.to_numeric(df_t[rev_col], errors="coerce")
            df_t.dropna(inplace=True)

            if not df_t.empty:
                df_t["Month"] = df_t[date_col].dt.to_period("M")
                monthly = df_t.groupby("Month")[rev_col].sum()

                if len(monthly) >= 2:
                    last = monthly.iloc[-1]
                    prev = monthly.iloc[-2]
                    if prev > 0:
                        pct = (last - prev) / prev * 100
                        if pct > 0:
                            insights.append(Insight(
                                text=f"Revenue grew {pct:.1f}% in the most recent month compared to the prior month.",
                                category="revenue",
                                importance="high",
                                icon="📈",
                            ))
                        else:
                            insights.append(Insight(
                                text=f"Revenue declined {abs(pct):.1f}% in the most recent month — review pricing or demand.",
                                category="revenue",
                                importance="high",
                                icon="📉",
                            ))

        # Profit margin
        profit_col = next(
            (c.raw_name for c in detection.columns if c.canonical_name == "Profit"), None
        )
        if profit_col and profit_col in df.columns:
            profit = pd.to_numeric(df[profit_col], errors="coerce").dropna().sum()
            if total > 0:
                margin = profit / total * 100
                importance = "high" if margin < 10 else "medium"
                insights.append(Insight(
                    text=f"Overall profit margin is {margin:.1f}%."
                         + (" Consider reviewing cost structure." if margin < 10 else ""),
                    category="revenue",
                    importance=importance,
                    icon="💹",
                ))

        return insights

    # ── Product ────────────────────────────────────────────────────────────────

    def _product_insights(
        self, df: pd.DataFrame, detection: DetectionReport
    ) -> list[Insight]:
        insights = []
        groups = get_kpi_columns(detection)
        product_col = next(
            (c.raw_name for c in detection.columns if c.category == "product"
             and c.raw_name in df.columns),
            None,
        )
        rev_cols = groups["revenue"]

        if not product_col or not rev_cols:
            return []

        rev_col = rev_cols[0]
        agg = (
            pd.to_numeric(df.groupby(product_col)[rev_col].sum(), errors="coerce")
            .sort_values(ascending=False)
        )

        if agg.empty:
            return []

        total_rev = agg.sum()
        if total_rev <= 0:
            return []

        # Top product share
        top = agg.iloc[0]
        top_pct = top / total_rev * 100
        if top_pct > 30:
            insights.append(Insight(
                text=f"'{agg.index[0]}' alone contributes {top_pct:.0f}% of total revenue — high concentration risk.",
                category="product",
                importance="high",
                icon="⚠️",
            ))

        # Top-3 combined share
        top3_pct = agg.head(3).sum() / total_rev * 100
        if top3_pct > 60:
            insights.append(Insight(
                text=f"The top 3 products account for {top3_pct:.0f}% of total revenue.",
                category="product",
                importance="medium",
                icon="🏆",
            ))

        # Low performers
        low_pct = (agg / total_rev * 100)
        low_count = int((low_pct < 1).sum())
        if low_count > 0:
            insights.append(Insight(
                text=f"{low_count} products each contribute less than 1% of revenue — consider portfolio review.",
                category="product",
                importance="low",
                icon="🔍",
            ))

        return insights

    # ── Customer ───────────────────────────────────────────────────────────────

    def _customer_insights(
        self, df: pd.DataFrame, detection: DetectionReport, kpis: list[KPI]
    ) -> list[Insight]:
        insights = []
        customer_col = next(
            (c.raw_name for c in detection.columns if c.category == "customer"
             and c.raw_name in df.columns),
            None,
        )
        groups = get_kpi_columns(detection)
        rev_cols = groups["revenue"]

        if not customer_col or not rev_cols:
            return []

        rev_col = rev_cols[0]
        agg = (
            pd.to_numeric(df.groupby(customer_col)[rev_col].sum(), errors="coerce")
            .sort_values(ascending=False)
        )

        if agg.empty:
            return []

        total_rev = agg.sum()
        if total_rev <= 0:
            return []

        top_cust_pct = agg.iloc[0] / total_rev * 100
        if top_cust_pct > 25:
            insights.append(Insight(
                text=f"'{agg.index[0]}' is your top customer, representing {top_cust_pct:.0f}% of total revenue.",
                category="customer",
                importance="high",
                icon="⭐",
            ))

        return insights

    # ── Data Quality ───────────────────────────────────────────────────────────

    def _quality_insights(self, quality: DataQualityReport) -> list[Insight]:
        insights = []

        if quality.duplicate_rows > 0:
            insights.append(Insight(
                text=f"{quality.duplicate_rows} duplicate rows were detected and removed during cleaning.",
                category="quality",
                importance="medium",
                icon="🔄",
            ))

        high_null_cols = [
            c for c in quality.columns if c.null_pct > 20
        ]
        if high_null_cols:
            names = ", ".join(c.column for c in high_null_cols[:3])
            insights.append(Insight(
                text=f"High missing-value rate detected in: {names}. Data completeness may affect accuracy.",
                category="quality",
                importance="high",
                icon="❌",
            ))

        if quality.quality_score < 70:
            insights.append(Insight(
                text=f"Overall data quality score is {quality.quality_score:.0f}/100. Review the Data Quality tab for details.",
                category="quality",
                importance="high",
                icon="⚠️",
            ))

        return insights

    # ── Trend ──────────────────────────────────────────────────────────────────

    def _trend_insights(
        self, df: pd.DataFrame, detection: DetectionReport
    ) -> list[Insight]:
        insights = []
        groups = get_kpi_columns(detection)
        date_cols = groups["date"]
        rev_cols = groups["revenue"]

        if not date_cols or not rev_cols:
            return []

        date_col = date_cols[0]
        rev_col = rev_cols[0]

        df_t = df[[date_col, rev_col]].copy()
        df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
        df_t[rev_col] = pd.to_numeric(df_t[rev_col], errors="coerce")
        df_t.dropna(inplace=True)

        if df_t.empty:
            return []

        df_t["Quarter"] = df_t[date_col].dt.to_period("Q")
        quarterly = df_t.groupby("Quarter")[rev_col].sum()

        if len(quarterly) >= 4:
            last_q = quarterly.iloc[-1]
            prev_q = quarterly.iloc[-2]
            if prev_q > 0:
                qoq = (last_q - prev_q) / prev_q * 100
                trend = "grew" if qoq > 0 else "declined"
                insights.append(Insight(
                    text=f"Revenue {trend} {abs(qoq):.1f}% quarter-over-quarter in the latest period.",
                    category="revenue",
                    importance="medium",
                    icon="📊",
                ))

        return insights
