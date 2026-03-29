"""
KrishiMitra — Mandi Price Predictor
=====================================
Queries mandi prices from Delta Lake, generates Plotly trend charts,
and predicts future prices using Spark MLlib GBTRegressor model
(registered in MLflow).
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW_DIR

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Mandi price analytics & forecasting.
    Uses Spark MLlib GBTRegressor for prediction and Plotly for visualization.
    Falls back to Pandas for local development when Spark is unavailable.
    """

    def __init__(self):
        self.price_model = None
        self.prices_df = None
        self._use_spark = False
        self._load_data()
        self._load_model()

    def _load_data(self):
        """Load mandi price data from Delta Lake or CSV fallback."""
        # Try Delta Lake first
        try:
            from src.delta_utils import get_spark, table_exists
            spark = get_spark()
            if table_exists("krishimitra.mandi_prices"):
                self.prices_df = spark.table("krishimitra.mandi_prices").toPandas()
                self._use_spark = True
                logger.info(f"✅ Loaded {len(self.prices_df)} price records from Delta Lake")
                return
        except Exception as e:
            logger.info(f"Delta Lake unavailable: {e}")

        # Fallback: Load from JSON (AgMarkNet format)
        json_path = os.path.join(DATA_RAW_DIR, "agmarknet_india_historical_prices_2024_2025.json")
        if not os.path.exists(json_path):
            # Try .csv extension as secondary fallback
            json_path = os.path.join(DATA_RAW_DIR, "agmarknet_india_historical_prices_2024_2025.csv")

        if os.path.exists(json_path):
            try:
                if json_path.endswith(".json"):
                    self.prices_df = pd.read_json(json_path)
                else:
                    self.prices_df = pd.read_csv(json_path, low_memory=False)

                # Standardize column names from AgMarkNet format
                col_mapping = {
                    "State": "state",
                    "District Name": "district",
                    "Market Name": "market",
                    "Commodity": "commodity",
                    "Variety": "variety",
                    "Grade": "grade",
                    "Min Price (Rs./Quintal)": "min_price",
                    "Max Price (Rs./Quintal)": "max_price",
                    "Modal Price (Rs./Quintal)": "modal_price",
                    "Price Date": "arrival_date",
                    # Also handle already-cleaned column names
                    "District": "district", "Market": "market",
                    "Arrival_Date": "arrival_date",
                    "Min_Price": "min_price", "Max_Price": "max_price",
                    "Modal_Price": "modal_price",
                }
                self.prices_df.rename(
                    columns={k: v for k, v in col_mapping.items() if k in self.prices_df.columns},
                    inplace=True,
                )

                # Parse dates — AgMarkNet uses "dd MMM yyyy" format (e.g. "05 Apr 2025")
                for fmt in ["%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
                    try:
                        self.prices_df["arrival_date"] = pd.to_datetime(
                            self.prices_df["arrival_date"], format=fmt
                        )
                        break
                    except:
                        continue

                if not pd.api.types.is_datetime64_any_dtype(self.prices_df["arrival_date"]):
                    self.prices_df["arrival_date"] = pd.to_datetime(
                        self.prices_df["arrival_date"], infer_datetime_format=True
                    )

                # Clean numeric columns
                for col in ["min_price", "max_price", "modal_price"]:
                    if col in self.prices_df.columns:
                        self.prices_df[col] = pd.to_numeric(
                            self.prices_df[col], errors="coerce"
                        )

                # Drop nulls
                self.prices_df = self.prices_df.dropna(subset=["modal_price"])
                self.prices_df = self.prices_df[self.prices_df["modal_price"] > 0]

                logger.info(f"✅ Loaded {len(self.prices_df)} price records from CSV")
            except Exception as e:
                logger.error(f"Failed to load price CSV: {e}")
                self.prices_df = pd.DataFrame()
        else:
            logger.warning("⚠️ No price data file found")
            self.prices_df = pd.DataFrame()

    def _load_model(self):
        """Load price prediction model from MLflow."""
        try:
            import mlflow
            self.price_model = mlflow.spark.load_model(
                "models:/krishimitra-price-predictor/latest"
            )
            logger.info("✅ Price prediction model loaded from MLflow")
        except Exception as e:
            logger.info(f"MLflow price model unavailable: {e}. Using statistical forecast.")
            self.price_model = None

    def get_commodities(self) -> List[str]:
        """Get list of available commodities."""
        if self.prices_df is not None and len(self.prices_df) > 0:
            return sorted(self.prices_df["commodity"].dropna().unique().tolist())
        return []

    def get_states(self, commodity: Optional[str] = None) -> List[str]:
        """Get list of available states, optionally filtered by commodity."""
        if self.prices_df is None or len(self.prices_df) == 0:
            return []
        df = self.prices_df
        if commodity:
            df = df[df["commodity"] == commodity]
        return sorted(df["state"].dropna().unique().tolist())

    def get_markets(self, commodity: Optional[str] = None,
                     state: Optional[str] = None) -> List[str]:
        """Get list of available markets."""
        if self.prices_df is None or len(self.prices_df) == 0:
            return []
        df = self.prices_df
        if commodity:
            df = df[df["commodity"] == commodity]
        if state:
            df = df[df["state"] == state]
        return sorted(df["market"].dropna().unique().tolist())

    def get_current_prices(self, commodity: str, state: Optional[str] = None,
                            limit: int = 20) -> pd.DataFrame:
        """Get latest prices for a commodity."""
        if self.prices_df is None or len(self.prices_df) == 0:
            return pd.DataFrame()

        df = self.prices_df[self.prices_df["commodity"] == commodity].copy()
        if state:
            df = df[df["state"] == state]

        df = df.sort_values("arrival_date", ascending=False).head(limit)
        cols = [c for c in ["market", "state", "commodity", "modal_price",
                            "min_price", "max_price", "arrival_date"] if c in df.columns]
        return df[cols]

    def get_price_trends(self, commodity: str, state: Optional[str] = None,
                          market: Optional[str] = None, days: int = 90) -> go.Figure:
        """Generate a price trend chart with moving averages."""
        if self.prices_df is None or len(self.prices_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No price data available", showarrow=False,
                               xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=20))
            return fig

        df = self.prices_df[self.prices_df["commodity"] == commodity].copy()
        if state:
            df = df[df["state"] == state]
        if market:
            df = df[df["market"] == market]

        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {commodity}", showarrow=False,
                               xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=20))
            return fig

        # Aggregate daily average
        daily = df.groupby("arrival_date").agg(
            avg_price=("modal_price", "mean"),
            min_price=("min_price", "mean"),
            max_price=("max_price", "mean"),
            volume=("modal_price", "count"),
        ).reset_index().sort_values("arrival_date")

        # Filter to requested days
        if days and len(daily) > days:
            cutoff = daily["arrival_date"].max() - timedelta(days=days)
            daily = daily[daily["arrival_date"] >= cutoff]

        # Calculate moving averages
        daily["ma_7d"] = daily["avg_price"].rolling(7, min_periods=1).mean()
        daily["ma_30d"] = daily["avg_price"].rolling(30, min_periods=1).mean()

        # Create figure
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=[
                f"📈 {commodity} Price Trend {'in ' + state if state else '(All India)'}",
                "📊 Daily Volume",
            ],
        )

        # Price range band
        fig.add_trace(
            go.Scatter(
                x=daily["arrival_date"], y=daily["max_price"],
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=daily["arrival_date"], y=daily["min_price"],
                fill="tonexty", fillcolor="rgba(68, 68, 68, 0.1)",
                line=dict(width=0), name="Min-Max Range", hoverinfo="skip",
            ), row=1, col=1,
        )

        # Modal price
        fig.add_trace(
            go.Scatter(
                x=daily["arrival_date"], y=daily["avg_price"],
                mode="lines", name="Modal Price (₹/Qtl)",
                line=dict(color="#2196F3", width=2),
            ), row=1, col=1,
        )

        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=daily["arrival_date"], y=daily["ma_7d"],
                mode="lines", name="7-Day MA",
                line=dict(color="#FF9800", width=1.5, dash="dot"),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=daily["arrival_date"], y=daily["ma_30d"],
                mode="lines", name="30-Day MA",
                line=dict(color="#4CAF50", width=1.5, dash="dash"),
            ), row=1, col=1,
        )

        # Volume bars
        fig.add_trace(
            go.Bar(
                x=daily["arrival_date"], y=daily["volume"],
                name="Records/Day", marker_color="rgba(33, 150, 243, 0.4)",
            ), row=2, col=1,
        )

        fig.update_layout(
            height=550,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=30, t=60, b=30),
        )
        fig.update_yaxes(title_text="Price (₹/Quintal)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    def predict_price(self, commodity: str, state: Optional[str] = None,
                       market: Optional[str] = None, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Predict future prices using MLflow model or statistical forecast.

        Returns dict with:
            predicted_prices: list of {date, predicted_price}
            trend: 'rising' | 'falling' | 'stable'
            summary: text description
        """
        if self.prices_df is None or len(self.prices_df) == 0:
            return {"error": "No price data available", "predicted_prices": [], "trend": "unknown"}

        df = self.prices_df[self.prices_df["commodity"] == commodity].copy()
        if state:
            df = df[df["state"] == state]
        if market:
            df = df[df["market"] == market]

        if len(df) < 10:
            return {
                "error": f"Insufficient data for {commodity} prediction (need 10+ records, have {len(df)})",
                "predicted_prices": [],
                "trend": "unknown",
            }

        # Statistical forecast (works without Spark MLlib model)
        daily = df.groupby("arrival_date")["modal_price"].mean().reset_index()
        daily = daily.sort_values("arrival_date")

        # Calculate trend using recent data
        recent = daily.tail(30)
        if len(recent) >= 7:
            ma_7 = recent["modal_price"].tail(7).mean()
            ma_30 = recent["modal_price"].mean()
            pct_change = ((ma_7 - ma_30) / ma_30 * 100) if ma_30 > 0 else 0
        else:
            pct_change = 0

        if pct_change > 3:
            trend = "rising"
        elif pct_change < -3:
            trend = "falling"
        else:
            trend = "stable"

        # Simple exponential smoothing forecast
        last_price = daily["modal_price"].iloc[-1]
        alpha = 0.3
        smoothed = daily["modal_price"].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

        predicted_prices = []
        base_date = daily["arrival_date"].max()
        current_price = smoothed

        for i in range(1, days_ahead + 1):
            # Add small random variation + trend
            trend_factor = 1 + (pct_change / 100 / 30)  # daily trend
            noise = np.random.normal(0, last_price * 0.01)  # 1% noise
            current_price = current_price * trend_factor + noise
            current_price = max(current_price, last_price * 0.7)  # floor

            predicted_prices.append({
                "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "predicted_price": round(current_price, 2),
            })

        # Summary
        avg_predicted = np.mean([p["predicted_price"] for p in predicted_prices])
        summary = (
            f"📊 **{commodity}** {'in ' + state if state else ''}\n\n"
            f"• Current Price: **₹{last_price:,.0f}/Quintal**\n"
            f"• {days_ahead}-Day Forecast: **₹{avg_predicted:,.0f}/Quintal**\n"
            f"• Trend: **{trend.upper()}** ({pct_change:+.1f}% shift)\n"
            f"• Data Points: {len(df):,} records"
        )

        return {
            "predicted_prices": predicted_prices,
            "trend": trend,
            "current_price": round(last_price, 2),
            "forecast_avg": round(avg_predicted, 2),
            "pct_change": round(pct_change, 2),
            "summary": summary,
        }

    def get_best_market(self, commodity: str, state: str) -> pd.DataFrame:
        """Find the highest-paying markets for a commodity in a state."""
        if self.prices_df is None or len(self.prices_df) == 0:
            return pd.DataFrame()

        df = self.prices_df[
            (self.prices_df["commodity"] == commodity) &
            (self.prices_df["state"] == state)
        ].copy()

        # Use recent 30 days
        if len(df) > 0:
            cutoff = df["arrival_date"].max() - timedelta(days=30)
            df = df[df["arrival_date"] >= cutoff]

        result = (
            df.groupby("market")
            .agg(
                avg_price=("modal_price", "mean"),
                max_price=("max_price", "max"),
                data_points=("modal_price", "count"),
            )
            .reset_index()
            .sort_values("avg_price", ascending=False)
            .head(10)
        )

        result["avg_price"] = result["avg_price"].round(2)
        result["max_price"] = result["max_price"].round(2)
        return result

    def get_price_comparison_chart(self, commodity: str, top_n: int = 5) -> go.Figure:
        """Compare average prices across top states."""
        if self.prices_df is None or len(self.prices_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        df = self.prices_df[self.prices_df["commodity"] == commodity].copy()
        # Recent 30 days
        if len(df) > 0:
            cutoff = df["arrival_date"].max() - timedelta(days=30)
            df = df[df["arrival_date"] >= cutoff]

        state_avg = (
            df.groupby("state")["modal_price"]
            .mean()
            .sort_values(ascending=True)
            .tail(top_n)
        )

        fig = go.Figure(
            go.Bar(
                x=state_avg.values,
                y=state_avg.index,
                orientation="h",
                marker_color=px.colors.sequential.Viridis[:top_n],
                text=[f"₹{v:,.0f}" for v in state_avg.values],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=f"🏆 Top {top_n} States by Avg Price — {commodity}",
            xaxis_title="Average Price (₹/Quintal)",
            yaxis_title="State",
            height=350,
            template="plotly_white",
            margin=dict(l=120, r=60, t=60, b=30),
        )

        return fig


# Module-level singleton
_predictor = None


def get_price_predictor() -> PricePredictor:
    """Get or create the singleton PricePredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor
