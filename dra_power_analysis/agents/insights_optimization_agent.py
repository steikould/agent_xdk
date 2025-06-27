from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base_agent import Agent


class InsightsOptimizationAgent(Agent):
    """
    Generates actionable recommendations for operations based on analysis results.
    """

    EFFICIENCY_DEGRADATION_THRESHOLD = 0.05  # 5% drop from baseline/average
    HIGH_POWER_CONSUMPTION_FACTOR = 2.0  # Max power > X times median power
    LOW_EFFICIENCY_ABSOLUTE_THRESHOLD = 0.60

    def __init__(self):
        super().__init__("InsightsOptimizationAgent")

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): statistical_summary, efficiency_trends,
                                 time_series_aggregations (optional), data_quality_report (optional)
        """
        required_keys = ["statistical_summary", "efficiency_trends"]
        if not self._validate_input(data, required_keys=required_keys):
            return self._handle_error("Missing required data for insights generation.")

        stats_summary = data.get("statistical_summary", {})
        efficiency_trends_data = data.get("efficiency_trends", {})
        dq_report = data.get("data_quality_report", {})

        opportunities: List[Dict[str, Any]] = []
        summary_points: List[str] = []
        risk_notes: List[str] = []

        for pump_name, pump_stats in stats_summary.items():
            if pump_name == "SystemTotal" or not isinstance(pump_stats, dict):
                continue
            self.logger.info(f"Generating insights for pump: {pump_name}")

            pump_eff_trend_df = efficiency_trends_data.get(pump_name)
            if (
                pump_eff_trend_df is not None
                and not pump_eff_trend_df.empty
                and "calculated_pump_efficiency" in pump_eff_trend_df.columns
            ):
                # Ensure efficiency is numeric and drop NaNs for calculation
                pump_eff_trend_df["calculated_pump_efficiency"] = pd.to_numeric(
                    pump_eff_trend_df["calculated_pump_efficiency"], errors="coerce"
                )
                valid_efficiencies = pump_eff_trend_df[
                    "calculated_pump_efficiency"
                ].dropna()

                if not valid_efficiencies.empty:
                    avg_efficiency = valid_efficiencies.mean()
                    current_efficiency = valid_efficiencies.iloc[-1]

                    if (
                        pd.notna(current_efficiency)
                        and current_efficiency < self.LOW_EFFICIENCY_ABSOLUTE_THRESHOLD
                    ):
                        opportunities.append(
                            {
                                "pump_name": pump_name,
                                "type": "Low Efficiency Operation",
                                "description": f"Pump {pump_name} operating at low efficiency ({current_efficiency:.2%}).",
                                "recommendation": f"Investigate {pump_name} for mechanical/operational issues. Consider refurbishment.",
                                "priority": 1,
                                "estimated_savings_potential": "High",
                            }
                        )
                        summary_points.append(
                            f"{pump_name}: Critically low efficiency ({current_efficiency:.2%}). Investigation needed."
                        )
                        risk_notes.append(
                            f"Delaying {pump_name} low efficiency investigation risks failure/high costs."
                        )

                    elif (
                        pd.notna(avg_efficiency)
                        and pd.notna(current_efficiency)
                        and avg_efficiency > 1e-3
                        and (avg_efficiency - current_efficiency) / avg_efficiency
                        > self.EFFICIENCY_DEGRADATION_THRESHOLD
                    ):
                        opportunities.append(
                            {
                                "pump_name": pump_name,
                                "type": "Efficiency Degradation",
                                "description": f"{pump_name} efficiency degraded by {((avg_efficiency - current_efficiency) / avg_efficiency):.2%}.",
                                "recommendation": f"Schedule inspection for {pump_name} for wear/fouling.",
                                "priority": 2,
                                "estimated_savings_potential": "Medium",
                            }
                        )
                        summary_points.append(
                            f"{pump_name}: Efficiency degraded. Plan maintenance."
                        )
            else:
                self.logger.debug(f"No valid efficiency trend data for {pump_name}.")

            max_power = pump_stats.get("max", 0)
            median_power = pump_stats.get("median", 0)
            if (
                pd.notna(max_power)
                and pd.notna(median_power)
                and median_power > 1e-3
                and max_power > median_power * self.HIGH_POWER_CONSUMPTION_FACTOR
            ):
                opportunities.append(
                    {
                        "pump_name": pump_name,
                        "type": "Intermittent High Power Consumption",
                        "description": f"{pump_name} shows high power peaks (max: {max_power:.2f} kW vs median: {median_power:.2f} kW).",
                        "recommendation": f"Investigate {pump_name} operations during peak power. Check for blockages/control issues.",
                        "priority": 3,
                        "estimated_savings_potential": "Variable",
                    }
                )
                summary_points.append(
                    f"{pump_name}: Shows intermittent high power peaks."
                )

            if (
                dq_report
                and dq_report.get("detailed_issues")
                and pump_eff_trend_df is not None
            ):
                pump_dq_issues = [
                    iss
                    for iss in dq_report["detailed_issues"]
                    if isinstance(iss, dict)
                    and iss.get("tag_name")
                    and pump_name in iss.get("tag_name", "")
                ]
                is_low_eff = any(
                    opp.get("pump_name") == pump_name
                    and opp.get("type") == "Low Efficiency Operation"
                    for opp in opportunities
                )
                if pump_dq_issues and is_low_eff:
                    opportunities.append(
                        {
                            "pump_name": pump_name,
                            "type": "Combined Maintenance Needed",
                            "description": f"{pump_name} has low efficiency and {len(pump_dq_issues)} sensor data quality issues.",
                            "recommendation": f"Prioritize combined maintenance for {pump_name}: address efficiency and sensor issues.",
                            "priority": 1,
                            "estimated_savings_potential": "High",
                        }
                    )
                    summary_points.append(
                        f"{pump_name}: Low efficiency & sensor DQ issues. Priority maintenance."
                    )

        if (
            not opportunities
            and stats_summary
            and any(k != "SystemTotal" for k in stats_summary.keys())
        ):
            summary_points.append(
                "System components analyzed appear to be operating within expected parameters. Continuous monitoring advised."
            )
        elif not stats_summary or all(k == "SystemTotal" for k in stats_summary.keys()):
            summary_points.append(
                "No per-pump statistical summary available to generate specific insights."
            )

        ranked_opportunities = sorted(
            opportunities, key=lambda x: x.get("priority", 99)
        )
        self.logger.info(
            f"Generated {len(ranked_opportunities)} optimization opportunities."
        )
        return {
            "status": "success",
            "optimization_opportunities": ranked_opportunities,
            "executive_summary_points": list(set(summary_points)),
            "risk_assessment_notes": list(set(risk_notes)),
        }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    insights_agent = InsightsOptimizationAgent()

    pump_a_eff_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-03"]),
            "calculated_pump_efficiency": [0.75, 0.68],
        }
    )
    pump_b_eff_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-03"]),
            "calculated_pump_efficiency": [0.58, 0.52],
        }
    )

    sample_input = {
        "statistical_summary": {
            "PUMP_A": {"mean": 100, "median": 95, "max": 180},
            "PUMP_B": {"mean": 120, "median": 110, "max": 300},
            "SystemTotal": {"mean": 220, "median": 205, "max": 480},
        },
        "efficiency_trends": {"PUMP_A": pump_a_eff_data, "PUMP_B": pump_b_eff_data},
        "data_quality_report": {
            "detailed_issues": [{"tag_name": "PUMP_B_SENSOR", "severity": "critical"}]
        },
    }

    print("\n--- Test Case: Generating Insights ---")
    result = insights_agent.execute(sample_input)
    if result["status"] == "success":
        # print("Executive Summary Points:", result["executive_summary_points"])
        # print("Optimization Opportunities:", result["optimization_opportunities"])
        assert len(result["optimization_opportunities"]) > 0
        assert any(
            "PUMP_A" in opp.get("description", "")
            and "Degradation" in opp.get("type", "")
            for opp in result["optimization_opportunities"]
        )
        assert any(
            "PUMP_B" in opp.get("description", "")
            and "Low Efficiency Operation" in opp.get("type", "")
            for opp in result["optimization_opportunities"]
        )
    else:
        print(f"Insights generation failed: {result['error_message']}")

    print("\nInsightsOptimizationAgent tests completed.")
