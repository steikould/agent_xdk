"""
Data Quality Validation Agent for DRA Power Analysis
Location: dra_power_analysis/sub_agents/data_quality/agent.py
"""
from typing import Dict, Any, List, Tuple, Optional, ClassVar
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from google.adk.agents import BaseAgent
from google.adk.tools import FunctionTool

# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models for tool schemas
class DataQualityInput(BaseModel):
    """Input model for data quality validation."""
    sensor_data: Dict[str, Any] = Field(..., description="Sensor data to validate (serialized DataFrame)")
    validation_config: Optional[Dict[str, Any]] = Field(
        default={},
        description="Validation configuration parameters"
    )


class DataQualityOutput(BaseModel):
    """Output model for data quality results."""
    success: bool
    validated_data: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    error_message: Optional[str] = None


class DataQualityValidationAgent(BaseAgent):
    """
    Assesses data integrity of sensor data and provides recommendations.
    """

    # Class constants with proper type annotations
    MAX_CONSECUTIVE_MISSING_THRESHOLD_MULTIPLIER: ClassVar[int] = 3
    PUMP_STATUS_MIN_CYCLE_DURATION_SECONDS: ClassVar[int] = 60
    FLOWRATE_REALISTIC_MIN: ClassVar[float] = 0.0
    FLOWRATE_REALISTIC_MAX_FACTOR: ClassVar[float] = 2.5
    TYPICAL_MAX_FLOWRATES: ClassVar[dict[str, float]] = {
        "UPSTREAM_FLOW": 500,
        "DOWNSTREAM_FLOW": 500,
        "PUMP_A_FLOW_RATE_M3S": 0.05,
        "PUMP_B_FLOW_RATE_M3S": 0.04,
        "DRA_RATE_PUMP_A": 20,
    }
    TIMESTAMP_CONSISTENCY_TOLERANCE_SECONDS: ClassVar[int] = 5

    def __init__(self, **kwargs):
        """Initialize the Data Quality Validation Agent."""
        # Initialize parent
        super().__init__(
            name="DataQualityValidationAgent",
            description="Assesses data integrity of retrieved sensor data",
            **kwargs
        )
        logger.info(f"DataQualityValidationAgent '{self.name}' initialized.")

    def get_tools(self) -> List[FunctionTool]:
        """Get agent tools."""
        return self._create_tools()
    
    def _create_tools(self) -> List[FunctionTool]:
        """Create agent tools."""
        
        async def validate_sensor_data(input_data: DataQualityInput) -> DataQualityOutput:
            """
            Validate sensor data quality.
            
            Args:
                input_data: Sensor data and validation configuration
                
            Returns:
                Data quality validation results
            """
            try:
                # Deserialize DataFrame
                sensor_df = pd.DataFrame(input_data.sensor_data)
                
                if sensor_df.empty:
                    return DataQualityOutput(
                        success=True,
                        quality_report={
                            "summary": "No data to assess.",
                            "issues": [],
                            "issues_by_type": {}
                        },
                        recommendations=["Ensure data retrieval process is working if data was expected."],
                        validated_data={}
                    )
                
                # Validate required columns
                expected_cols = ["timestamp", "tag_name", "value"]
                if not all(col in sensor_df.columns for col in expected_cols):
                    return DataQualityOutput(
                        success=False,
                        error_message=f"Missing required columns. Expected: {expected_cols}"
                    )
                
                # Process data
                try:
                    if not pd.api.types.is_datetime64_any_dtype(sensor_df["timestamp"]):
                        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
                    sensor_df["value"] = pd.to_numeric(sensor_df["value"], errors="coerce")
                except Exception as e:
                    return DataQualityOutput(
                        success=False,
                        error_message=f"Failed to process columns: {e}"
                    )
                
                # Perform quality checks
                processed_df, report, recommendations = self._perform_dq_checks(sensor_df.copy())
                
                # Serialize DataFrame for output
                validated_data = processed_df.to_dict('records')
                
                return DataQualityOutput(
                    success=True,
                    validated_data=validated_data,
                    quality_report=report,
                    recommendations=recommendations
                )
                
            except Exception as e:
                logger.error(f"Data quality validation failed: {e}", exc_info=True)
                return DataQualityOutput(
                    success=False,
                    error_message=f"Validation error: {str(e)}"
                )
        
        async def check_data_completeness(input_data: DataQualityInput) -> DataQualityOutput:
            """
            Check data completeness and coverage.
            
            Args:
                input_data: Sensor data to check
                
            Returns:
                Completeness analysis results
            """
            try:
                sensor_df = pd.DataFrame(input_data.sensor_data)
                
                # Analyze completeness
                completeness_report = {
                    "total_records": len(sensor_df),
                    "unique_tags": sensor_df["tag_name"].nunique() if "tag_name" in sensor_df.columns else 0,
                    "time_coverage": None,
                    "missing_data_percentage": 0
                }
                
                if not sensor_df.empty and "timestamp" in sensor_df.columns:
                    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
                    completeness_report["time_coverage"] = {
                        "start": sensor_df["timestamp"].min().isoformat(),
                        "end": sensor_df["timestamp"].max().isoformat(),
                        "duration_hours": (sensor_df["timestamp"].max() - sensor_df["timestamp"].min()).total_seconds() / 3600
                    }
                
                # Check for missing values
                if "value" in sensor_df.columns:
                    missing_count = sensor_df["value"].isna().sum()
                    completeness_report["missing_data_percentage"] = (missing_count / len(sensor_df)) * 100
                
                return DataQualityOutput(
                    success=True,
                    quality_report=completeness_report,
                    recommendations=self._generate_completeness_recommendations(completeness_report)
                )
                
            except Exception as e:
                return DataQualityOutput(
                    success=False,
                    error_message=f"Completeness check failed: {str(e)}"
                )
        
        # Return tools list
        return [
            FunctionTool(func=validate_sensor_data),
            FunctionTool(func=check_data_completeness)
        ]

    def _perform_dq_checks(self, sensor_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """
        Private helper to encapsulate the synchronous DQ check logic.
        Returns the processed DataFrame, report dictionary, and recommendations list.
        """
        sensor_df = sensor_df.sort_values(by=["tag_name", "timestamp"]).reset_index(drop=True)
        all_issues: List[Dict[str, Any]] = []
        all_recommendations: List[str] = []

        for tag_name, group_df in sensor_df.groupby("tag_name"):
            logger.debug(f"Assessing quality for tag: {tag_name}")
            group_df_cleaned = group_df.dropna(subset=["value"])
            if group_df_cleaned.empty:
                continue

            missing_issues, missing_recs = self._check_missing_data(group_df_cleaned.copy(), str(tag_name))
            all_issues.extend(missing_issues)
            all_recommendations.extend(missing_recs)

            if "STATUS" in str(tag_name).upper() and "PUMP" in str(tag_name).upper():
                pump_issues, pump_recs = self._check_pump_status_transitions(group_df_cleaned.copy(), str(tag_name))
                all_issues.extend(pump_issues)
                all_recommendations.extend(pump_recs)

            if any(ft in str(tag_name).upper() for ft in ["FLOW", "RATE"]):
                flow_issues, flow_recs = self._check_unrealistic_flowrates(group_df_cleaned.copy(), str(tag_name))
                all_issues.extend(flow_issues)
                all_recommendations.extend(flow_recs)

            ts_issues, ts_recs = self._check_timestamp_consistency(group_df_cleaned.copy(), str(tag_name))
            all_issues.extend(ts_issues)
            all_recommendations.extend(ts_recs)

        report = {
            "summary": f"Found {len(all_issues)} potential data quality issues.",
            "issues_by_type": self._summarize_issues_by_type(all_issues),
            "detailed_issues_count": len(all_issues),
            "detailed_issues": all_issues,
        }

        critical_alerts = [issue for issue in all_issues if issue.get("severity") == "critical"]
        if critical_alerts:
            all_recommendations.append("CRITICAL ALERTS generated. Review detailed issues immediately.")

        return sensor_df, report, list(set(all_recommendations))

    def _summarize_issues_by_type(self, issues: List[Dict]) -> Dict[str, int]:
        """Summarize issues by type."""
        summary_by_type = {}
        for issue in issues:
            issue_type = issue.get("type", "Unknown")
            summary_by_type[issue_type] = summary_by_type.get(issue_type, 0) + 1
        return summary_by_type

    def _check_missing_data(self, df: pd.DataFrame, tag_name: str) -> Tuple[List[Dict], List[str]]:
        """Check for missing data gaps."""
        issues, recs = [], []
        if len(df) < 2:
            return issues, recs

        time_diffs_seconds = df["timestamp"].diff().dt.total_seconds()
        if time_diffs_seconds.empty or len(time_diffs_seconds) <= 1:
            return issues, recs

        median_interval = time_diffs_seconds.median()
        if pd.notna(median_interval) and median_interval > 0:
            missing_threshold_seconds = (
                self.MAX_CONSECUTIVE_MISSING_THRESHOLD_MULTIPLIER * median_interval
            )
            missing_threshold_seconds = max(
                missing_threshold_seconds,
                self.TIMESTAMP_CONSISTENCY_TOLERANCE_SECONDS * 2,
            )

            large_gaps_indices = time_diffs_seconds[
                time_diffs_seconds > missing_threshold_seconds
            ].index
            for idx in large_gaps_indices:
                if idx == df.index[0]:
                    continue
                actual_gap = time_diffs_seconds[idx]
                issues.append({
                    "type": "Missing Data Gap",
                    "tag_name": tag_name,
                    "timestamp": df.loc[idx, "timestamp"],
                    "message": f"Potential data gap of {actual_gap:.0f}s before this point (expected ~{median_interval:.0f}s).",
                    "severity": "warning",
                })
        if issues:
            recs.append(f"Review data gaps for {tag_name}.")
        return issues, recs

    def _check_pump_status_transitions(self, df: pd.DataFrame, tag_name: str) -> Tuple[List[Dict], List[str]]:
        """Check pump status transitions."""
        issues, recs = [], []
        if len(df) < 2:
            return issues, recs

        # Values should be 0 or 1 for status tags
        if not df["value"].isin([0, 1]).all():
            issues.append({
                "type": "Invalid Pump Status Value",
                "tag_name": tag_name,
                "timestamp": df.iloc[0]["timestamp"],
                "message": "Pump status values are not all 0 or 1.",
                "severity": "critical",
            })
            recs.append(f"Correct invalid pump status values for {tag_name}.")
            return issues, recs

        last_status, last_ts = None, None
        for _, row in df.iterrows():
            current_status, current_ts = row["value"], row["timestamp"]
            if last_status is not None and current_status != last_status:
                duration = (current_ts - last_ts).total_seconds()
                if duration < self.PUMP_STATUS_MIN_CYCLE_DURATION_SECONDS:
                    issues.append({
                        "type": "Anomalous Pump Transition",
                        "tag_name": tag_name,
                        "timestamp": current_ts,
                        "message": f"Pump status changed from {int(last_status)} to {int(current_status)} in only {duration:.0f}s.",
                        "severity": "warning",
                    })
            last_status, last_ts = current_status, current_ts
        if issues:
            recs.append(f"Investigate frequent pump status changes for {tag_name}.")
        return issues, recs

    def _check_unrealistic_flowrates(self, df: pd.DataFrame, tag_name: str) -> Tuple[List[Dict], List[str]]:
        """Check for unrealistic flow rates."""
        issues, recs = [], []
        negative_flows = df[df["value"] < self.FLOWRATE_REALISTIC_MIN]
        for _, row in negative_flows.iterrows():
            issues.append({
                "type": "Unrealistic Flowrate (Negative)",
                "tag_name": tag_name,
                "timestamp": row["timestamp"],
                "value": row["value"],
                "message": f"Negative flowrate: {row['value']:.2f}.",
                "severity": "critical",
            })

        # Use specific typical max if available
        typical_max = self.TYPICAL_MAX_FLOWRATES.get(
            tag_name, df["value"].quantile(0.99) if len(df) > 10 else np.inf
        )
        absolute_max_threshold = typical_max * self.FLOWRATE_REALISTIC_MAX_FACTOR

        extreme_spikes = df[df["value"] > absolute_max_threshold]
        for _, row in extreme_spikes.iterrows():
            issues.append({
                "type": "Unrealistic Flowrate (Extreme Spike)",
                "tag_name": tag_name,
                "timestamp": row["timestamp"],
                "value": row["value"],
                "message": f"Extreme flowrate: {row['value']:.2f} (threshold: {absolute_max_threshold:.2f}).",
                "severity": "warning",
            })

        if any(i["type"] == "Unrealistic Flowrate (Negative)" for i in issues):
            recs.append(f"Critical: Negative flowrates for {tag_name}.")
        if any(i["type"] == "Unrealistic Flowrate (Extreme Spike)" for i in issues):
            recs.append(f"Review extreme flowrate spikes for {tag_name}.")
        return issues, recs

    def _check_timestamp_consistency(self, df: pd.DataFrame, tag_name: str) -> Tuple[List[Dict], List[str]]:
        """Check timestamp consistency."""
        issues, recs = [], []
        if len(df) < 2:
            return issues, recs

        time_diffs_seconds = df["timestamp"].diff().dt.total_seconds()
        if (time_diffs_seconds < 0).any():
            issues.append({
                "type": "Timestamp Order",
                "tag_name": tag_name,
                "timestamp": df.iloc[0]["timestamp"],
                "message": "Timestamps out of chronological order.",
                "severity": "critical",
            })
            recs.append(f"Critical: Fix timestamp order for {tag_name}.")
            return issues, recs

        if len(time_diffs_seconds) > 1:
            valid_diffs = time_diffs_seconds.iloc[1:]
            if not valid_diffs.empty:
                median_interval = valid_diffs.median()
                irregular_indices = valid_diffs[
                    (valid_diffs < median_interval / 3) | (valid_diffs > median_interval * 3)
                ][valid_diffs > self.TIMESTAMP_CONSISTENCY_TOLERANCE_SECONDS].index

                for idx in irregular_indices:
                    issues.append({
                        "type": "Timestamp Irregularity",
                        "tag_name": tag_name,
                        "timestamp": df.loc[idx, "timestamp"],
                        "message": f"Irregular interval of {time_diffs_seconds[idx]:.0f}s (median ~{median_interval:.0f}s).",
                        "severity": "info",
                    })
        if any(i["type"] == "Timestamp Irregularity" for i in issues):
            recs.append(f"Review timestamp irregularities for {tag_name}.")
        return issues, recs

    def _generate_completeness_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on completeness report."""
        recommendations = []
        
        if report.get("missing_data_percentage", 0) > 5:
            recommendations.append(f"High missing data rate ({report['missing_data_percentage']:.1f}%). Investigate sensor reliability.")
        
        if report.get("unique_tags", 0) < 10:
            recommendations.append("Low number of unique tags. Verify all sensors are reporting.")
        
        if report.get("time_coverage", {}).get("duration_hours", 0) < 24:
            recommendations.append("Less than 24 hours of data. Consider longer analysis period for trends.")
        
        return recommendations


# Create singleton instance for backward compatibility
dq_agent = DataQualityValidationAgent()