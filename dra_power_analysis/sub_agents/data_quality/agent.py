from typing import Dict, Any, List, Tuple, AsyncGenerator, ClassVar
import pandas as pd
import numpy as np
import asyncio # For potentially long-running CPU-bound tasks if needed

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types


class DataQualityValidationAgent(BaseAgent):
    """
    Assesses data integrity of sensor data and provides recommendations.
    """

    MAX_CONSECUTIVE_MISSING_THRESHOLD_MULTIPLIER: ClassVar[int] = 3  # Times median interval
    PUMP_STATUS_MIN_CYCLE_DURATION_SECONDS: ClassVar[float] = 60
    FLOWRATE_REALISTIC_MIN: ClassVar[float] = 0.0
    FLOWRATE_REALISTIC_MAX_FACTOR: ClassVar[float] = 2.5  # Max flowrate can be X times its typical max
    TYPICAL_MAX_FLOWRATES: ClassVar[dict] = {
        "UPSTREAM_FLOW": 500,
        "DOWNSTREAM_FLOW": 500,
        "PUMP_A_FLOW_RATE_M3S": 0.05,
        "PUMP_B_FLOW_RATE_M3S": 0.04,
        "DRA_RATE_PUMP_A": 20,
    }
    TIMESTAMP_CONSISTENCY_TOLERANCE_SECONDS: ClassVar[float] = 5

    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.logger.info(f"DataQualityValidationAgent '{self.name}' initialized.")

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        self.logger.info(f"'{self.name}' starting data quality assessment.")

        sensor_df = ctx.session.state.get("sensor_data_df")

        if sensor_df is None:
            error_msg = "'sensor_data_df' not found in session state."
            self.logger.error(error_msg)
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        if not isinstance(sensor_df, pd.DataFrame):
            error_msg = "'sensor_data_df' must be a Pandas DataFrame."
            self.logger.error(error_msg)
            ctx.session.state["status_dq_validation"] = "error"
            ctx.session.state["error_message_dq_validation"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        if sensor_df.empty:
            self.logger.info("Received empty sensor_data_df. No quality checks needed.")
            report = {"summary": "No data to assess.", "issues": [], "issues_by_type": {}}
            recommendations = ["Ensure data retrieval process is working if data was expected."]
            ctx.session.state["data_quality_report"] = report
            ctx.session.state["data_quality_recommendations"] = recommendations
            # validated_data is the same empty df
            ctx.session.state["validated_sensor_data_df"] = sensor_df
            ctx.session.state["status_dq_validation"] = "success_no_data"
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text("Data quality: No data to assess.")]))
            return

        expected_cols = ["timestamp", "tag_name", "value"]
        if not all(col in sensor_df.columns for col in expected_cols):
            error_msg = f"sensor_data_df missing required columns: {expected_cols}"
            self.logger.error(error_msg)
            ctx.session.state["status_dq_validation"] = "error"
            ctx.session.state["error_message_dq_validation"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        try:
            if not pd.api.types.is_datetime64_any_dtype(sensor_df["timestamp"]):
                sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
            # Attempt to convert 'value' to numeric, coercing errors to NaN
            # Specific checks later will handle if conversion failed for critical tags
            sensor_df["value"] = pd.to_numeric(sensor_df["value"], errors="coerce")
        except Exception as e:
            error_msg = f"Failed to process 'timestamp' or 'value' columns in sensor_data_df: {e}"
            self.logger.error(error_msg, exc_info=True)
            ctx.session.state["status_dq_validation"] = "error"
            ctx.session.state["error_message_dq_validation"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        # The core DQ logic can be computationally intensive but is primarily synchronous pandas operations.
        # If it were extremely long, we might use asyncio.to_thread. For now, direct call.
        try:
            processed_sensor_df, report, recommendations = self._perform_dq_checks(sensor_df.copy())

            ctx.session.state["data_quality_report"] = report
            ctx.session.state["data_quality_recommendations"] = recommendations
            # Store the processed (sorted, types converted) DataFrame
            ctx.session.state["validated_sensor_data_df"] = processed_sensor_df
            ctx.session.state["status_dq_validation"] = "success"

            self.logger.info(f"Data quality assessment completed. Summary: {report.get('summary')}")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(f"Data quality assessment complete: {report.get('summary')}")])
            )
        except Exception as e:
            error_msg = f"Error during data quality checks execution: {e}"
            self.logger.error(error_msg, exc_info=True)
            ctx.session.state["status_dq_validation"] = "error"
            ctx.session.state["error_message_dq_validation"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))


    def _perform_dq_checks(self, sensor_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """
        Private helper to encapsulate the synchronous DQ check logic.
        Returns the processed DataFrame, report dictionary, and recommendations list.
        """
        sensor_df = sensor_df.sort_values(by=["tag_name", "timestamp"]).reset_index(drop=True)
        all_issues: List[Dict[str, Any]] = []
        all_recommendations: List[str] = []

        for tag_name, group_df in sensor_df.groupby("tag_name"):
            self.logger.debug(f"Assessing quality for tag: {tag_name}")
            group_df_cleaned = group_df.dropna(subset=["value"])
            if group_df_cleaned.empty:
                continue

            missing_issues, missing_recs = self._check_missing_data(group_df_cleaned.copy(), tag_name)
            all_issues.extend(missing_issues)
            all_recommendations.extend(missing_recs)

            if "STATUS" in tag_name.upper() and "PUMP" in tag_name.upper():
                pump_issues, pump_recs = self._check_pump_status_transitions(group_df_cleaned.copy(), tag_name)
                all_issues.extend(pump_issues)
                all_recommendations.extend(pump_recs)

            if any(ft in tag_name.upper() for ft in ["FLOW", "RATE"]):
                flow_issues, flow_recs = self._check_unrealistic_flowrates(group_df_cleaned.copy(), tag_name)
                all_issues.extend(flow_issues)
                all_recommendations.extend(flow_recs)

            ts_issues, ts_recs = self._check_timestamp_consistency(group_df_cleaned.copy(), tag_name)
            all_issues.extend(ts_issues)
            all_recommendations.extend(ts_recs)

        report = {
            "summary": f"Found {len(all_issues)} potential data quality issues.",
            "issues_by_type": self._summarize_issues_by_type(all_issues),
            "detailed_issues_count": len(all_issues), # Adding this for direct access
            "detailed_issues": all_issues,
        }

        critical_alerts = [issue for issue in all_issues if issue.get("severity") == "critical"]
        if critical_alerts:
            all_recommendations.append("CRITICAL ALERTS generated. Review detailed issues immediately.")

        return sensor_df, report, list(set(all_recommendations))

    def _summarize_issues_by_type(self, issues: List[Dict]) -> Dict[str, int]:
        summary_by_type = {}
        for issue in issues:
            issue_type = issue.get("type", "Unknown")
            summary_by_type[issue_type] = summary_by_type.get(issue_type, 0) + 1
        return summary_by_type

    def _check_missing_data(
        self, df: pd.DataFrame, tag_name: str
    ) -> Tuple[List[Dict], List[str]]:
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
            # Ensure threshold is reasonably large (e.g., > few seconds)
            missing_threshold_seconds = max(
                missing_threshold_seconds,
                self.TIMESTAMP_CONSISTENCY_TOLERANCE_SECONDS * 2,
            )

            large_gaps_indices = time_diffs_seconds[
                time_diffs_seconds > missing_threshold_seconds
            ].index
            for idx in large_gaps_indices:
                if idx == df.index[0]:
                    continue  # Skip first, as diff is NaN
                actual_gap = time_diffs_seconds[idx]
                issues.append(
                    {
                        "type": "Missing Data Gap",
                        "tag_name": tag_name,
                        "timestamp": df.loc[idx, "timestamp"],
                        "message": f"Potential data gap of {actual_gap:.0f}s before this point (expected ~{median_interval:.0f}s).",
                        "severity": "warning",
                    }
                )
        if issues:
            recs.append(f"Review data gaps for {tag_name}.")
        return issues, recs

    def _check_pump_status_transitions(
        self, df: pd.DataFrame, tag_name: str
    ) -> Tuple[List[Dict], List[str]]:
        issues, recs = [], []
        if len(df) < 2:
            return issues, recs

        # Values should be 0 or 1 for status tags
        if not df["value"].isin([0, 1]).all():
            issues.append(
                {
                    "type": "Invalid Pump Status Value",
                    "tag_name": tag_name,
                    "timestamp": df.iloc[0]["timestamp"],
                    "message": "Pump status values are not all 0 or 1.",
                    "severity": "critical",
                }
            )
            recs.append(f"Correct invalid pump status values for {tag_name}.")
            return issues, recs  # Stop further transition checks if values are invalid

        last_status, last_ts = None, None
        for _, row in df.iterrows():
            current_status, current_ts = row["value"], row["timestamp"]
            if last_status is not None and current_status != last_status:
                duration = (current_ts - last_ts).total_seconds()
                if duration < self.PUMP_STATUS_MIN_CYCLE_DURATION_SECONDS:
                    issues.append(
                        {
                            "type": "Anomalous Pump Transition",
                            "tag_name": tag_name,
                            "timestamp": current_ts,
                            "message": f"Pump status changed from {int(last_status)} to {int(current_status)} in only {duration:.0f}s.",
                            "severity": "warning",
                        }
                    )
            last_status, last_ts = current_status, current_ts
        if issues:
            recs.append(f"Investigate frequent pump status changes for {tag_name}.")
        return issues, recs

    def _check_unrealistic_flowrates(
        self, df: pd.DataFrame, tag_name: str
    ) -> Tuple[List[Dict], List[str]]:
        issues, recs = [], []
        negative_flows = df[df["value"] < self.FLOWRATE_REALISTIC_MIN]
        for _, row in negative_flows.iterrows():
            issues.append(
                {
                    "type": "Unrealistic Flowrate (Negative)",
                    "tag_name": tag_name,
                    "timestamp": row["timestamp"],
                    "value": row["value"],
                    "message": f"Negative flowrate: {row['value']:.2f}.",
                    "severity": "critical",
                }
            )

        # Use specific typical max if available, else try to infer (less reliable for spikes)
        typical_max = self.TYPICAL_MAX_FLOWRATES.get(
            tag_name, df["value"].quantile(0.99) if len(df) > 10 else np.inf
        )
        absolute_max_threshold = typical_max * self.FLOWRATE_REALISTIC_MAX_FACTOR

        extreme_spikes = df[df["value"] > absolute_max_threshold]
        for _, row in extreme_spikes.iterrows():
            issues.append(
                {
                    "type": "Unrealistic Flowrate (Extreme Spike)",
                    "tag_name": tag_name,
                    "timestamp": row["timestamp"],
                    "value": row["value"],
                    "message": f"Extreme flowrate: {row['value']:.2f} (threshold: {absolute_max_threshold:.2f}).",
                    "severity": "warning",
                }
            )

        if any(i["type"] == "Unrealistic Flowrate (Negative)" for i in issues):
            recs.append(f"Critical: Negative flowrates for {tag_name}.")
        if any(i["type"] == "Unrealistic Flowrate (Extreme Spike)" for i in issues):
            recs.append(f"Review extreme flowrate spikes for {tag_name}.")
        return issues, recs

    def _check_timestamp_consistency(
        self, df: pd.DataFrame, tag_name: str
    ) -> Tuple[List[Dict], List[str]]:
        issues, recs = [], []
        if len(df) < 2:
            return issues, recs

        time_diffs_seconds = df["timestamp"].diff().dt.total_seconds()
        if (time_diffs_seconds < 0).any():  # Should have been sorted, but double check
            issues.append(
                {
                    "type": "Timestamp Order",
                    "tag_name": tag_name,
                    "timestamp": df.iloc[0]["timestamp"],
                    "message": "Timestamps out of chronological order.",
                    "severity": "critical",
                }
            )
            recs.append(f"Critical: Fix timestamp order for {tag_name}.")
            return issues, recs

        if len(time_diffs_seconds) > 1:
            valid_diffs = time_diffs_seconds.iloc[1:]  # Skip first NaN
            if not valid_diffs.empty:
                median_interval = valid_diffs.median()
                # Check for significant deviations from median interval
                irregular_indices = valid_diffs[
                    (valid_diffs < median_interval / 3)
                    | (valid_diffs > median_interval * 3)
                ][valid_diffs > self.TIMESTAMP_CONSISTENCY_TOLERANCE_SECONDS].index

                for idx in irregular_indices:
                    issues.append(
                        {
                            "type": "Timestamp Irregularity",
                            "tag_name": tag_name,
                            "timestamp": df.loc[idx, "timestamp"],
                            "message": f"Irregular interval of {time_diffs_seconds[idx]:.0f}s (median ~{median_interval:.0f}s).",
                            "severity": "info",
                        }
                    )
        if any(i["type"] == "Timestamp Irregularity" for i in issues):
            recs.append(f"Review timestamp irregularities for {tag_name}.")
        return issues, recs


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    dq_agent = DataQualityValidationAgent()

    now = pd.Timestamp.now().round("s")  # Round to seconds for cleaner diffs in mock
    sample_data_issues_list = [
        # PUMP_X_STATUS: quick transitions
        {
            "timestamp": now - pd.Timedelta(minutes=10),
            "tag_name": "PUMP_X_STATUS",
            "value": 0,
        },
        {
            "timestamp": now - pd.Timedelta(minutes=9, seconds=30),
            "tag_name": "PUMP_X_STATUS",
            "value": 1,
        },  # On after 30s
        {
            "timestamp": now - pd.Timedelta(minutes=9, seconds=0),
            "tag_name": "PUMP_X_STATUS",
            "value": 0,
        },  # Off after 30s (too quick)
        {
            "timestamp": now - pd.Timedelta(minutes=8, seconds=0),
            "tag_name": "PUMP_X_STATUS",
            "value": 1,
        },  # On after 60s (ok)
        # FLOW_Y_MAIN: gap, negative, spike, irregular ts
        {
            "timestamp": now - pd.Timedelta(minutes=20),
            "tag_name": "FLOW_Y_MAIN",
            "value": 200,
        },
        {
            "timestamp": now - pd.Timedelta(minutes=15),
            "tag_name": "FLOW_Y_MAIN",
            "value": 205,
        },  # 5 min gap before this
        {
            "timestamp": now - pd.Timedelta(minutes=14),
            "tag_name": "FLOW_Y_MAIN",
            "value": -10,
        },  # Negative
        {
            "timestamp": now - pd.Timedelta(minutes=13),
            "tag_name": "FLOW_Y_MAIN",
            "value": 5000,
        },  # Spike (assume typical max for FLOW_Y_MAIN is ~250)
        {
            "timestamp": now - pd.Timedelta(minutes=12),
            "tag_name": "FLOW_Y_MAIN",
            "value": 210,
        },
        {
            "timestamp": now - pd.Timedelta(minutes=11, seconds=57),
            "tag_name": "FLOW_Y_MAIN",
            "value": 208,
        },  # Irregular interval (3s)
        {
            "timestamp": now - pd.Timedelta(minutes=5),
            "tag_name": "FLOW_Y_MAIN",
            "value": 209,
        },  # Another large gap
    ]
    sample_data_issues = pd.DataFrame(sample_data_issues_list)
    # Update typical max for the test tag
    DataQualityValidationAgent.TYPICAL_MAX_FLOWRATES["FLOW_Y_MAIN"] = 250

    print("\n--- Test Case: Data with Issues ---")
    result_issues = dq_agent.execute({"sensor_data": sample_data_issues.copy()})
    if result_issues["status"] == "success":
        print(f"Summary: {result_issues['data_quality_report']['summary']}")
        print("Issues by Type:", result_issues["data_quality_report"]["issues_by_type"])
        # print("Detailed Issues:", result_issues['data_quality_report']['detailed_issues'])
        # print(f"Recommendations: {result_issues['recommendations']}")
        assert len(result_issues["data_quality_report"]["detailed_issues"]) > 0
        assert (
            "Anomalous Pump Transition"
            in result_issues["data_quality_report"]["issues_by_type"]
        )
        assert (
            "Missing Data Gap" in result_issues["data_quality_report"]["issues_by_type"]
        )
        assert (
            "Unrealistic Flowrate (Negative)"
            in result_issues["data_quality_report"]["issues_by_type"]
        )
        assert (
            "Unrealistic Flowrate (Extreme Spike)"
            in result_issues["data_quality_report"]["issues_by_type"]
        )

    print("\nDataQualityValidationAgent tests completed.")

dq_agent = DataQualityValidationAgent(
    name="DataQualityValidationAgent",
    description="Assesses data integrity of retrieved sensor data."
)