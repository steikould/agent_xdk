from typing import Dict, Any, List, Tuple, Optional, AsyncGenerator, ClassVar
import pandas as pd
import numpy as np
import asyncio # For potentially long-running CPU-bound tasks

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types

# For more advanced stats, might need: from scipy import stats


class StatisticalAnalysisAgent(BaseAgent):
    """
    Performs time-series analysis and statistical computations on power and operational data.
    """

    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.logger.info(f"StatisticalAnalysisAgent '{self.name}' initialized.")

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        self.logger.info(f"'{self.name}' starting statistical analysis.")

        power_df = ctx.session.state.get("power_consumption_data")

        if power_df is None:
            error_msg = "'power_consumption_data' not found in session state."
            self.logger.error(error_msg)
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        if not isinstance(power_df, pd.DataFrame) or power_df.empty:
            error_msg = "'power_consumption_data' must be a non-empty Pandas DataFrame."
            self.logger.warning(error_msg) # Non-critical if empty, but log warning
            # Set empty/default results
            ctx.session.state["statistical_summary"] = {}
            ctx.session.state["time_series_aggregations"] = {"hourly": {}, "daily": {}}
            ctx.session.state["correlation_analysis"] = {}
            ctx.session.state["efficiency_trends"] = {}
            ctx.session.state["visualization_hints"] = []
            ctx.session.state["status_stats_analysis"] = "success_no_data"
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(f"Statistical Analysis: {error_msg}")]))
            return

        try:
            if not pd.api.types.is_datetime64_any_dtype(power_df["timestamp"]):
                power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])
        except Exception as e:
            error_msg = f"Failed to convert 'timestamp' in power_consumption_data: {e}"
            self.logger.error(error_msg, exc_info=True)
            ctx.session.state["status_stats_analysis"] = "error"
            ctx.session.state["error_message_stats_analysis"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        if "electrical_power_kw" not in power_df.columns:
            error_msg = "Missing 'electrical_power_kw' in power_consumption_data."
            self.logger.error(error_msg)
            ctx.session.state["status_stats_analysis"] = "error"
            ctx.session.state["error_message_stats_analysis"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        try:
            power_df["electrical_power_kw"] = pd.to_numeric(power_df["electrical_power_kw"], errors="coerce")
            power_df.dropna(subset=["electrical_power_kw"], inplace=True) # Essential for stats
            if power_df.empty:
                error_msg = "No valid 'electrical_power_kw' data after numeric conversion and NaN drop."
                self.logger.warning(error_msg)
                # Set empty/default results
                ctx.session.state["statistical_summary"] = {}
                ctx.session.state["time_series_aggregations"] = {"hourly": {}, "daily": {}}
                ctx.session.state["correlation_analysis"] = {}
                ctx.session.state["efficiency_trends"] = {}
                ctx.session.state["visualization_hints"] = []
                ctx.session.state["status_stats_analysis"] = "success_no_valid_power_data"
                yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(f"Statistical Analysis: {error_msg}")]))
                return
        except Exception as e:
            error_msg = f"Failed to convert 'electrical_power_kw' to numeric: {e}"
            self.logger.error(error_msg, exc_info=True)
            ctx.session.state["status_stats_analysis"] = "error"
            ctx.session.state["error_message_stats_analysis"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        # Core statistical logic, can be wrapped in a helper if it becomes too large
        try:
            all_pump_stats, all_time_aggregations, all_correlations, all_efficiency_trends, viz_hints = self._perform_statistical_calculations(power_df.copy())

            ctx.session.state["statistical_summary"] = all_pump_stats
            ctx.session.state["time_series_aggregations"] = all_time_aggregations
            ctx.session.state["correlation_analysis"] = all_correlations
            ctx.session.state["efficiency_trends"] = all_efficiency_trends
            ctx.session.state["visualization_hints"] = viz_hints
            ctx.session.state["status_stats_analysis"] = "success"

            self.logger.info("Statistical analysis completed successfully.")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(f"Statistical analysis complete. Generated {len(viz_hints)} visualization hints.")])
            )
        except Exception as e:
            error_msg = f"Error during statistical calculations: {e}"
            self.logger.error(error_msg, exc_info=True)
            ctx.session.state["status_stats_analysis"] = "error"
            ctx.session.state["error_message_stats_analysis"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))

    def _perform_statistical_calculations(self, power_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict, List]:
        all_pump_stats: Dict[str, Any] = {}
        all_time_aggregations: Dict[str, Any] = {"hourly": {}, "daily": {}}
        all_correlations: Dict[str, Any] = {}
        all_efficiency_trends: Dict[str, Any] = {}
        viz_hints: List[Dict[str, Any]] = []

        # Overall system statistics
        if not power_df.empty:
            system_power_df = power_df.groupby("timestamp")["electrical_power_kw"].sum().reset_index()
            system_power_df.rename(columns={"electrical_power_kw": "total_system_electrical_power_kw"}, inplace=True)
            overall_desc_stats = self._calculate_descriptive_stats(system_power_df, "total_system_electrical_power_kw", "SystemTotal")
            all_pump_stats["SystemTotal"] = overall_desc_stats
            if not system_power_df.empty:
                viz_hints.append({"type": "time_series", "data_source_column": "total_system_electrical_power_kw", "title": "Total System Electrical Power Over Time", "y_label": "Power (kW)", "data_df_name": "system_total_power_ts"})
                viz_hints.append({"type": "histogram", "data_source_column": "total_system_electrical_power_kw", "title": "Distribution of Total System Electrical Power", "x_label": "Power (kW)", "data_df_name": "system_total_power_hist"})
                all_time_aggregations["system_total_power_ts"] = system_power_df[["timestamp", "total_system_electrical_power_kw"]]
                all_time_aggregations["system_total_power_hist"] = system_power_df[["total_system_electrical_power_kw"]]

        # Per-pump analysis
        if "pump_name" not in power_df.columns:
            self.logger.warning("'pump_name' column not found in power_df. Skipping per-pump analysis.")
        else:
            for pump_name, group_df in power_df.groupby("pump_name"):
                self.logger.info(f"Performing statistical analysis for pump: {pump_name}")
                if group_df.empty: continue
                desc_stats = self._calculate_descriptive_stats(group_df, "electrical_power_kw", pump_name)
                all_pump_stats[pump_name] = desc_stats
                viz_hints.append({"type": "time_series", "pump_name": pump_name, "data_source_column": "electrical_power_kw", "title": f"{pump_name} Electrical Power Over Time", "y_label": "Power (kW)", "data_df_name": f"{pump_name}_power_ts"})
                all_time_aggregations[f"{pump_name}_power_ts"] = group_df[["timestamp", "electrical_power_kw"]]
                hourly_agg = self._aggregate_time_series(group_df.set_index("timestamp"), "H", pump_name)
                daily_agg = self._aggregate_time_series(group_df.set_index("timestamp"), "D", pump_name)
                if hourly_agg is not None: all_time_aggregations["hourly"][pump_name] = hourly_agg
                if daily_agg is not None: all_time_aggregations["daily"][pump_name] = daily_agg

                flow_rate_col_name = next((col for col in group_df.columns if pump_name in col and "_FLOW_RATE_M3S" in col), None)
                correlation_cols = ["electrical_power_kw"]
                if "delta_pressure_pa" in group_df.columns: correlation_cols.append("delta_pressure_pa")
                if flow_rate_col_name: correlation_cols.append(flow_rate_col_name)

                valid_corr_cols = [col for col in correlation_cols if col in group_df.columns and pd.api.types.is_numeric_dtype(group_df[col])]
                if len(valid_corr_cols) > 1:
                    pump_corr_df = group_df[valid_corr_cols].corr()
                    all_correlations[pump_name] = pump_corr_df
                    viz_hints.append({"type": "correlation_matrix_heatmap", "pump_name": pump_name, "data_df_name": f"correlation_{pump_name}", "title": f"{pump_name} Correlation Matrix"})
                    all_correlations[f"correlation_{pump_name}_data"] = pump_corr_df

                if "hydraulic_power_kw" in group_df.columns and "shaft_power_kw" in group_df.columns:
                    eff_df = group_df[["timestamp", "hydraulic_power_kw", "shaft_power_kw"]].copy()
                    eff_df["calculated_pump_efficiency"] = np.where((eff_df["shaft_power_kw"] > 1e-3), eff_df["hydraulic_power_kw"] / eff_df["shaft_power_kw"], np.nan)
                    eff_df.dropna(subset=["calculated_pump_efficiency"], inplace=True)
                    eff_df["calculated_pump_efficiency"] = eff_df["calculated_pump_efficiency"].clip(0, 1.1)
                    all_efficiency_trends[pump_name] = eff_df[["timestamp", "calculated_pump_efficiency"]]
                    if not eff_df.empty:
                        viz_hints.append({"type": "time_series", "pump_name": pump_name, "y_col": "calculated_pump_efficiency", "title": f"{pump_name} Calculated Pump Efficiency Over Time", "y_label": "Efficiency", "data_df_name": f"{pump_name}_efficiency_trend"})
                        all_efficiency_trends[f"{pump_name}_efficiency_trend_data"] = eff_df[["timestamp", "calculated_pump_efficiency"]]

        return all_pump_stats, all_time_aggregations, all_correlations, all_efficiency_trends, viz_hints

    def _calculate_descriptive_stats(self, df: pd.DataFrame, column_name: str, group_name: str) -> Dict[str, Any]:
        if column_name not in df.columns or df[column_name].empty:
            self.logger.warning(
                f"Column {column_name} for group {group_name} is empty for stats."
            )
            return {}
        series = df[column_name].dropna()
        if series.empty:
            return {}
        stats = {
            "mean": series.mean(),
            "median": series.median(),
            "std_dev": series.std(),
            "min": series.min(),
            "max": series.max(),
            "q1_25th_percentile": series.quantile(0.25),
            "q3_75th_percentile": series.quantile(0.75),
            "count": series.count(),
            "sum": series.sum(),
        }
        return {
            k: round(v, 3) if isinstance(v, (float, np.floating)) else v
            for k, v in stats.items()
        }

    def _aggregate_time_series(
        self, df: pd.DataFrame, freq: str, group_name: str
    ) -> Optional[pd.DataFrame]:
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning(
                f"Cannot aggregate for {group_name}: index is not DatetimeIndex."
            )
            return None
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return None

        agg_map = {col: "mean" for col in numeric_df.columns}
        if "electrical_power_kw" in agg_map:
            agg_map["electrical_power_kw"] = ["mean", "sum", "min", "max"]
        if "energy_kwh" in agg_map:
            agg_map["energy_kwh"] = "sum"

        try:
            aggregated_df = numeric_df.resample(freq).agg(agg_map)
            if isinstance(aggregated_df.columns, pd.MultiIndex):
                aggregated_df.columns = [
                    "_".join(col).strip() for col in aggregated_df.columns.values
                ]
            return aggregated_df.dropna(how="all")
        except Exception as e:
            self.logger.error(
                f"Error aggregating for {group_name} at freq {freq}: {e}", exc_info=True
            )
            return None


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    stat_agent = StatisticalAnalysisAgent()

    ts_power = [
        pd.Timestamp("2023-01-01 10:00:00") + pd.Timedelta(minutes=i * 10)
        for i in range(20)
    ]
    pump_a_power_data = pd.DataFrame(
        {
            "timestamp": ts_power,
            "pump_name": "PUMP_A",
            "electrical_power_kw": np.random.uniform(50, 150, 20),
            "hydraulic_power_kw": np.random.uniform(40, 120, 20),
            "shaft_power_kw": np.random.uniform(45, 130, 20),
            "delta_pressure_pa": np.random.uniform(100000, 500000, 20),
            "STN01_LINE01_PUMP_A_FLOW_RATE_M3S": np.random.uniform(0.01, 0.03, 20),
        }
    )
    sample_power_df_stats = pump_a_power_data

    print("\n--- Test Case: Valid Power Data (Statistical Analysis) ---")
    result = stat_agent.execute(
        {"power_consumption_data": sample_power_df_stats.copy()}
    )

    if result["status"] == "success":
        print("Statistical analysis successful.")
        # print("Statistical Summary (PUMP_A):", result["statistical_summary"].get("PUMP_A"))
        # print("Visualization Hints count:", len(result.get("visualization_hints", [])))
        assert "PUMP_A" in result["statistical_summary"]
        assert "SystemTotal" in result["statistical_summary"]
        assert result["time_series_aggregations"]["hourly"].get("PUMP_A") is not None
        assert len(result.get("visualization_hints", [])) > 0
    else:
        print(f"Statistical analysis failed: {result['error_message']}")

    print("\nStatisticalAnalysisAgent tests completed.")

stat_analysis_agent = StatisticalAnalysisAgent(
    name="StatisticalAnalysisAgent",
    description="Performs time-series analysis and statistical computations."
)