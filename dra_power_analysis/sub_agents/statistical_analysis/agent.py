"""
Statistical Analysis Agent for DRA Power Analysis
Location: dra_power_analysis/sub_agents/statistical_analysis/agent.py
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel, Field

from google.adk.agents import BaseAgent
from google.adk.tools import FunctionTool

# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models
class StatisticalAnalysisInput(BaseModel):
    """Input model for statistical analysis."""
    power_data: List[Dict[str, Any]] = Field(..., description="Power consumption data")
    analysis_options: Optional[Dict[str, Any]] = Field(
        default={},
        description="Analysis configuration options"
    )


class StatisticalAnalysisOutput(BaseModel):
    """Output model for statistical analysis results."""
    success: bool
    statistical_summary: Optional[Dict[str, Any]] = None
    time_series_aggregations: Optional[Dict[str, Any]] = None
    correlation_analysis: Optional[Dict[str, Any]] = None
    efficiency_trends: Optional[Dict[str, Any]] = None
    visualization_hints: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None


class StatisticalAnalysisAgent(BaseAgent):
    """
    Performs time-series analysis and statistical computations on power and operational data.
    """

    def __init__(self, name: str = "StatisticalAnalysisAgent",
                 description: str = "Performs statistical analysis on power data", **kwargs):
        """Initialize the Statistical Analysis Agent."""
        super().__init__(name=name, description=description, **kwargs)
        logger.info(f"StatisticalAnalysisAgent '{self.name}' initialized.")
        
        # Set up tools
        self._setup_tools()

    def _setup_tools(self):
        """Set up agent tools."""
        
        async def analyze_power_statistics(input_data: StatisticalAnalysisInput) -> StatisticalAnalysisOutput:
            """
            Perform comprehensive statistical analysis on power data.
            
            Args:
                input_data: Power consumption data and analysis options
                
            Returns:
                Statistical analysis results
            """
            try:
                # Convert input to DataFrame
                power_df = pd.DataFrame(input_data.power_data)
                
                if power_df.empty:
                    return StatisticalAnalysisOutput(
                        success=True,
                        statistical_summary={},
                        time_series_aggregations={"hourly": {}, "daily": {}},
                        correlation_analysis={},
                        efficiency_trends={},
                        visualization_hints=[]
                    )
                
                # Process timestamp column
                try:
                    if not pd.api.types.is_datetime64_any_dtype(power_df["timestamp"]):
                        power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])
                except Exception as e:
                    return StatisticalAnalysisOutput(
                        success=False,
                        error_message=f"Failed to convert timestamp: {e}"
                    )
                
                # Validate required columns
                if "electrical_power_kw" not in power_df.columns:
                    return StatisticalAnalysisOutput(
                        success=False,
                        error_message="Missing 'electrical_power_kw' column"
                    )
                
                # Convert to numeric
                try:
                    power_df["electrical_power_kw"] = pd.to_numeric(
                        power_df["electrical_power_kw"], errors="coerce"
                    )
                    power_df.dropna(subset=["electrical_power_kw"], inplace=True)
                except Exception as e:
                    return StatisticalAnalysisOutput(
                        success=False,
                        error_message=f"Failed to process power data: {e}"
                    )
                
                # Perform statistical calculations
                (all_pump_stats, all_time_aggregations, all_correlations, 
                 all_efficiency_trends, viz_hints) = self._perform_statistical_calculations(power_df.copy())
                
                return StatisticalAnalysisOutput(
                    success=True,
                    statistical_summary=all_pump_stats,
                    time_series_aggregations=all_time_aggregations,
                    correlation_analysis=all_correlations,
                    efficiency_trends=self._serialize_efficiency_trends(all_efficiency_trends),
                    visualization_hints=viz_hints
                )
                
            except Exception as e:
                logger.error(f"Statistical analysis failed: {e}", exc_info=True)
                return StatisticalAnalysisOutput(
                    success=False,
                    error_message=f"Statistical analysis error: {str(e)}"
                )
        
        async def analyze_trends(input_data: StatisticalAnalysisInput) -> StatisticalAnalysisOutput:
            """
            Analyze trends in power consumption data.
            
            Args:
                input_data: Power data for trend analysis
                
            Returns:
                Trend analysis results
            """
            try:
                power_df = pd.DataFrame(input_data.power_data)
                
                if power_df.empty:
                    return StatisticalAnalysisOutput(
                        success=True,
                        efficiency_trends={"message": "No data for trend analysis"}
                    )
                
                # Convert timestamp
                power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])
                
                # Simple trend analysis
                trends = {}
                if "pump_name" in power_df.columns:
                    for pump_name in power_df["pump_name"].unique():
                        pump_data = power_df[power_df["pump_name"] == pump_name]
                        if len(pump_data) > 1:
                            # Calculate trend (simplified)
                            x = np.arange(len(pump_data))
                            y = pump_data["electrical_power_kw"].values
                            z = np.polyfit(x, y, 1)
                            trend_direction = "increasing" if z[0] > 0 else "decreasing"
                            
                            trends[pump_name] = {
                                "trend": trend_direction,
                                "slope": float(z[0]),
                                "average_power": float(y.mean())
                            }
                
                return StatisticalAnalysisOutput(
                    success=True,
                    efficiency_trends=trends
                )
                
            except Exception as e:
                return StatisticalAnalysisOutput(
                    success=False,
                    error_message=f"Trend analysis error: {str(e)}"
                )
        
        # Register tools
        self.tools = [
            FunctionTool(func=analyze_power_statistics),
            FunctionTool(func=analyze_trends)
        ]

    def _perform_statistical_calculations(self, power_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict, List]:
        """Perform comprehensive statistical calculations."""
        all_pump_stats: Dict[str, Any] = {}
        all_time_aggregations: Dict[str, Any] = {"hourly": {}, "daily": {}}
        all_correlations: Dict[str, Any] = {}
        all_efficiency_trends: Dict[str, Any] = {}
        viz_hints: List[Dict[str, Any]] = []

        # Overall system statistics
        if not power_df.empty:
            system_power_df = power_df.groupby("timestamp")["electrical_power_kw"].sum().reset_index()
            system_power_df.rename(columns={"electrical_power_kw": "total_system_electrical_power_kw"}, inplace=True)
            overall_desc_stats = self._calculate_descriptive_stats(
                system_power_df, "total_system_electrical_power_kw", "SystemTotal"
            )
            all_pump_stats["SystemTotal"] = overall_desc_stats
            
            if not system_power_df.empty:
                viz_hints.extend([
                    {
                        "type": "time_series",
                        "data_source_column": "total_system_electrical_power_kw",
                        "title": "Total System Electrical Power Over Time",
                        "y_label": "Power (kW)",
                        "data_df_name": "system_total_power_ts"
                    },
                    {
                        "type": "histogram",
                        "data_source_column": "total_system_electrical_power_kw",
                        "title": "Distribution of Total System Electrical Power",
                        "x_label": "Power (kW)",
                        "data_df_name": "system_total_power_hist"
                    }
                ])
                all_time_aggregations["system_total_power_ts"] = system_power_df.to_dict('records')

        # Per-pump analysis
        if "pump_name" in power_df.columns:
            for pump_name, group_df in power_df.groupby("pump_name"):
                logger.info(f"Performing statistical analysis for pump: {pump_name}")
                
                if group_df.empty:
                    continue
                
                # Descriptive statistics
                desc_stats = self._calculate_descriptive_stats(
                    group_df, "electrical_power_kw", pump_name
                )
                all_pump_stats[pump_name] = desc_stats
                
                # Visualization hints
                viz_hints.append({
                    "type": "time_series",
                    "pump_name": pump_name,
                    "data_source_column": "electrical_power_kw",
                    "title": f"{pump_name} Electrical Power Over Time",
                    "y_label": "Power (kW)",
                    "data_df_name": f"{pump_name}_power_ts"
                })
                all_time_aggregations[f"{pump_name}_power_ts"] = group_df[
                    ["timestamp", "electrical_power_kw"]
                ].to_dict('records')
                
                # Time series aggregations
                hourly_agg = self._aggregate_time_series(
                    group_df.set_index("timestamp"), "H", pump_name
                )
                daily_agg = self._aggregate_time_series(
                    group_df.set_index("timestamp"), "D", pump_name
                )
                
                if hourly_agg is not None:
                    all_time_aggregations["hourly"][pump_name] = hourly_agg.to_dict('records')
                if daily_agg is not None:
                    all_time_aggregations["daily"][pump_name] = daily_agg.to_dict('records')
                
                # Correlations
                correlation_cols = ["electrical_power_kw"]
                if "delta_pressure_pa" in group_df.columns:
                    correlation_cols.append("delta_pressure_pa")
                
                flow_rate_col = next(
                    (col for col in group_df.columns if pump_name in col and "_FLOW_RATE_M3S" in col),
                    None
                )
                if flow_rate_col:
                    correlation_cols.append(flow_rate_col)
                
                valid_corr_cols = [
                    col for col in correlation_cols 
                    if col in group_df.columns and pd.api.types.is_numeric_dtype(group_df[col])
                ]
                
                if len(valid_corr_cols) > 1:
                    pump_corr_df = group_df[valid_corr_cols].corr()
                    all_correlations[pump_name] = pump_corr_df.to_dict()
                    viz_hints.append({
                        "type": "correlation_matrix_heatmap",
                        "pump_name": pump_name,
                        "data_df_name": f"correlation_{pump_name}",
                        "title": f"{pump_name} Correlation Matrix"
                    })
                
                # Efficiency trends
                if "hydraulic_power_kw" in group_df.columns and "shaft_power_kw" in group_df.columns:
                    eff_df = group_df[["timestamp", "hydraulic_power_kw", "shaft_power_kw"]].copy()
                    eff_df["calculated_pump_efficiency"] = np.where(
                        eff_df["shaft_power_kw"] > 1e-3,
                        eff_df["hydraulic_power_kw"] / eff_df["shaft_power_kw"],
                        np.nan
                    )
                    eff_df.dropna(subset=["calculated_pump_efficiency"], inplace=True)
                    eff_df["calculated_pump_efficiency"] = eff_df["calculated_pump_efficiency"].clip(0, 1.1)
                    
                    all_efficiency_trends[pump_name] = eff_df[["timestamp", "calculated_pump_efficiency"]]
                    
                    if not eff_df.empty:
                        viz_hints.append({
                            "type": "time_series",
                            "pump_name": pump_name,
                            "y_col": "calculated_pump_efficiency",
                            "title": f"{pump_name} Calculated Pump Efficiency Over Time",
                            "y_label": "Efficiency",
                            "data_df_name": f"{pump_name}_efficiency_trend"
                        })

        return all_pump_stats, all_time_aggregations, all_correlations, all_efficiency_trends, viz_hints

    def _calculate_descriptive_stats(self, df: pd.DataFrame, column_name: str, 
                                   group_name: str) -> Dict[str, Any]:
        """Calculate descriptive statistics for a column."""
        if column_name not in df.columns or df[column_name].empty:
            logger.warning(f"Column {column_name} for group {group_name} is empty for stats.")
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

    def _aggregate_time_series(self, df: pd.DataFrame, freq: str, 
                             group_name: str) -> Optional[pd.DataFrame]:
        """Aggregate time series data."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Cannot aggregate for {group_name}: index is not DatetimeIndex.")
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
            logger.error(f"Error aggregating for {group_name} at freq {freq}: {e}", exc_info=True)
            return None

    def _serialize_efficiency_trends(self, efficiency_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize efficiency trends for output."""
        serialized = {}
        for pump_name, trend_df in efficiency_trends.items():
            if isinstance(trend_df, pd.DataFrame):
                serialized[pump_name] = trend_df.to_dict('records')
            else:
                serialized[pump_name] = trend_df
        return serialized


# Create singleton instance for backward compatibility
stat_analysis_agent = StatisticalAnalysisAgent(
    name="StatisticalAnalysisAgent",
    description="Performs time-series analysis and statistical computations."
)