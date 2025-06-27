from typing import Dict, Any, List, AsyncGenerator, Tuple
import pandas as pd
import numpy as np
import asyncio # For potentially long-running CPU-bound tasks

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event, EventActions, types


class InsightsOptimizationAgent(BaseAgent):
    """
    Generates actionable recommendations for operations based on analysis results.
    """

    EFFICIENCY_DEGRADATION_THRESHOLD = 0.05  # 5% drop from baseline/average
    HIGH_POWER_CONSUMPTION_FACTOR = 2.0  # Max power > X times median power
    LOW_EFFICIENCY_ABSOLUTE_THRESHOLD = 0.60

    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.logger.info(f"InsightsOptimizationAgent '{self.name}' initialized.")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        self.logger.info(f"'{self.name}' starting insights and optimization generation.")

        stats_summary = ctx.session.state.get("statistical_summary")
        efficiency_trends_data = ctx.session.state.get("efficiency_trends")
        dq_report = ctx.session.state.get("data_quality_report") # Optional based on previous agent

        if stats_summary is None or efficiency_trends_data is None:
            error_msg = "Missing 'statistical_summary' or 'efficiency_trends' in session state."
            self.logger.error(error_msg)
            # Potentially non-escalating if some insights can be generated without all data
            # However, for this agent, these are fairly critical.
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))
            return

        # Ensure dq_report is a dict, even if empty, to simplify access later
        if dq_report is None:
            dq_report = {}


        # Core insights logic - can be wrapped in a helper if it's very large
        try:
            opportunities, summary_points, risk_notes = self._generate_insights(
                stats_summary, efficiency_trends_data, dq_report
            )

            ranked_opportunities = sorted(opportunities, key=lambda x: x.get("priority", 99))

            ctx.session.state["optimization_opportunities"] = ranked_opportunities
            ctx.session.state["executive_summary_points"] = list(set(summary_points))
            ctx.session.state["risk_assessment_notes"] = list(set(risk_notes))
            ctx.session.state["status_insights"] = "success"

            # Consolidate final report elements here
            # This part was previously in the old orchestrator
            final_report_payload = {
                "user_inputs": ctx.session.state.get("user_params"),
                "data_quality_assessment": ctx.session.state.get("data_quality_report"),
                "data_quality_recommendations": ctx.session.state.get("data_quality_recommendations"),
                "power_consumption_summary": ctx.session.state.get("power_summary_metrics"),
                "statistical_analysis_highlights": ctx.session.state.get("statistical_summary"),
                "visualization_data_store": { # Store data for potential UI agent use
                    "time_series_aggregations": ctx.session.state.get("time_series_aggregations", {}),
                    "correlation_analysis": ctx.session.state.get("correlation_analysis", {}),
                    "efficiency_trends": ctx.session.state.get("efficiency_trends", {})
                },
                "visualization_hints": ctx.session.state.get("visualization_hints", []),
                "optimization_roadmap": {"opportunities": ranked_opportunities},
                "executive_summary_points": list(set(summary_points)),
                "risk_assessment_notes": list(set(risk_notes)),
                "status": "success" # Overall status
            }

            # Create executive summary string
            exec_summary_str = "Executive Summary:\n"
            if not final_report_payload["executive_summary_points"]:
                exec_summary_str += "- Analysis complete. No specific high-priority points generated. Review detailed sections."
            else:
                for point in final_report_payload["executive_summary_points"]:
                    exec_summary_str += f"- {point}\n"

            power_summary = final_report_payload.get("power_consumption_summary", {})
            if power_summary: # Check if power_summary exists and is not None
                 exec_summary_str += f"- Approx total energy: {power_summary.get('total_electrical_energy_kwh_approx','N/A')} kWh. Avg power: {power_summary.get('average_electrical_power_kw','N/A')} kW.\n"

            final_report_payload["executive_summary"] = exec_summary_str
            ctx.session.state["final_consolidated_report"] = final_report_payload


            self.logger.info(f"Generated {len(ranked_opportunities)} optimization opportunities.")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(f"Insights generated: {len(ranked_opportunities)} opportunities.")])
            )

        except Exception as e:
            error_msg = f"Error during insights generation: {e}"
            self.logger.error(error_msg, exc_info=True)
            ctx.session.state["status_insights"] = "error"
            ctx.session.state["error_message_insights"] = error_msg
            yield Event(author=self.name, content=types.Content(parts=[types.Part.from_text(error_msg)]), actions=EventActions(escalate=True))

    def _generate_insights(self, stats_summary: Dict, efficiency_trends_data: Dict, dq_report: Dict) -> Tuple[List[Dict], List[str], List[str]]:
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
                and isinstance(pump_eff_trend_df, pd.DataFrame) # Check if it's a DataFrame
                and not pump_eff_trend_df.empty
                and "calculated_pump_efficiency" in pump_eff_trend_df.columns
            ):
                pump_eff_trend_df["calculated_pump_efficiency"] = pd.to_numeric(
                    pump_eff_trend_df["calculated_pump_efficiency"], errors="coerce"
                )
                valid_efficiencies = pump_eff_trend_df["calculated_pump_efficiency"].dropna()

                if not valid_efficiencies.empty:
                    avg_efficiency = valid_efficiencies.mean()
                    # Ensure there's at least one entry before trying iloc[-1]
                    current_efficiency = valid_efficiencies.iloc[-1] if len(valid_efficiencies) > 0 else np.nan


                    if pd.notna(current_efficiency) and current_efficiency < self.LOW_EFFICIENCY_ABSOLUTE_THRESHOLD:
                        opportunities.append({"pump_name": pump_name, "type": "Low Efficiency Operation", "description": f"Pump {pump_name} operating at low efficiency ({current_efficiency:.2%}).", "recommendation": f"Investigate {pump_name} for mechanical/operational issues. Consider refurbishment.", "priority": 1, "estimated_savings_potential": "High"})
                        summary_points.append(f"{pump_name}: Critically low efficiency ({current_efficiency:.2%}). Investigation needed.")
                        risk_notes.append(f"Delaying {pump_name} low efficiency investigation risks failure/high costs.")
                    elif pd.notna(avg_efficiency) and pd.notna(current_efficiency) and avg_efficiency > 1e-3 and (avg_efficiency - current_efficiency) / avg_efficiency > self.EFFICIENCY_DEGRADATION_THRESHOLD:
                        opportunities.append({"pump_name": pump_name, "type": "Efficiency Degradation", "description": f"{pump_name} efficiency degraded by {((avg_efficiency - current_efficiency) / avg_efficiency):.2%}.", "recommendation": f"Schedule inspection for {pump_name} for wear/fouling.", "priority": 2, "estimated_savings_potential": "Medium"})
                        summary_points.append(f"{pump_name}: Efficiency degraded. Plan maintenance.")
            else:
                self.logger.debug(f"No valid efficiency trend data or DataFrame for {pump_name}.")

            max_power = pump_stats.get("max", 0)
            median_power = pump_stats.get("median", 0)
            if pd.notna(max_power) and pd.notna(median_power) and median_power > 1e-3 and max_power > median_power * self.HIGH_POWER_CONSUMPTION_FACTOR:
                opportunities.append({"pump_name": pump_name, "type": "Intermittent High Power Consumption", "description": f"{pump_name} shows high power peaks (max: {max_power:.2f} kW vs median: {median_power:.2f} kW).", "recommendation": f"Investigate {pump_name} operations during peak power. Check for blockages/control issues.", "priority": 3, "estimated_savings_potential": "Variable"})
                summary_points.append(f"{pump_name}: Shows intermittent high power peaks.")

            if dq_report and dq_report.get("detailed_issues") and pump_eff_trend_df is not None and isinstance(pump_eff_trend_df, pd.DataFrame):
                pump_dq_issues = [iss for iss in dq_report["detailed_issues"] if isinstance(iss, dict) and iss.get("tag_name") and pump_name in iss.get("tag_name", "")]
                is_low_eff = any(opp.get("pump_name") == pump_name and opp.get("type") == "Low Efficiency Operation" for opp in opportunities)
                if pump_dq_issues and is_low_eff:
                    opportunities.append({"pump_name": pump_name, "type": "Combined Maintenance Needed", "description": f"{pump_name} has low efficiency and {len(pump_dq_issues)} sensor data quality issues.", "recommendation": f"Prioritize combined maintenance for {pump_name}: address efficiency and sensor issues.", "priority": 1, "estimated_savings_potential": "High"})
                    summary_points.append(f"{pump_name}: Low efficiency & sensor DQ issues. Priority maintenance.")

        if not opportunities and stats_summary and any(k != "SystemTotal" for k in stats_summary.keys()):
            summary_points.append("System components analyzed appear to be operating within expected parameters. Continuous monitoring advised.")
        elif not stats_summary or all(k == "SystemTotal" for k in stats_summary.keys()):
            summary_points.append("No per-pump statistical summary available to generate specific insights.")

        return opportunities, summary_points, risk_notes


if __name__ == "__main__":
    import logging
    import asyncio

    async def test_insights_agent():
        logging.basicConfig(level=logging.DEBUG)
        agent_description = "Test Insights Agent - Generates actionable recommendations."
        insights_agent = InsightsOptimizationAgent(name="TestInsightsAgent", description=agent_description)

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

    # Mock InvocationContext and Session
    mock_session_state = {
        "statistical_summary": sample_input["statistical_summary"],
        "efficiency_trends": sample_input["efficiency_trends"],
        "data_quality_report": sample_input["data_quality_report"],
        # Add other data that might be used for the final consolidated report
        "user_params": {"location_id": "TestLoc", "pipeline_line_number": "TestPipe", "start_date": "2023-01-01", "end_date": "2023-01-03"},
        "data_quality_recommendations": ["Review sensor X"],
        "power_summary_metrics": {"total_electrical_energy_kwh_approx": 1000, "average_electrical_power_kw": 50},
        "time_series_aggregations": {}, "correlation_analysis": {}, "visualization_hints": []
    }
    mock_session = type('Session', (), {'state': mock_session_state})()
    mock_ctx = type('InvocationContext', (), {'session': mock_session})()

    print("\n--- Test Case: Generating Insights (ADK Agent) ---")
    async for event in insights_agent._run_async_impl(mock_ctx):
        print(f"Event from {event.author}: {event.content.parts[0].text if event.content and event.content.parts else 'No event content'}")

    if mock_ctx.session.state.get("status_insights") == "success":
        opportunities = mock_ctx.session.state.get("optimization_opportunities", [])
        # print("Executive Summary Points:", mock_ctx.session.state.get("executive_summary_points"))
        # print("Optimization Opportunities:", opportunities)
        # print("Final Consolidated Report:", mock_ctx.session.state.get("final_consolidated_report"))
        assert len(opportunities) > 0
        assert any("PUMP_A" in opp.get("description", "") and "Degradation" in opp.get("type", "") for opp in opportunities)
        assert any("PUMP_B" in opp.get("description", "") and "Low Efficiency Operation" in opp.get("type", "") for opp in opportunities)
        assert mock_ctx.session.state.get("final_consolidated_report") is not None
        assert "Executive Summary:" in mock_ctx.session.state.get("final_consolidated_report", {}).get("executive_summary","")
    else:
        print(f"Insights generation failed: {mock_ctx.session.state.get('error_message_insights')}")

    print("\nInsightsOptimizationAgent ADK tests completed.")

    asyncio.run(test_insights_agent())
