from typing import Dict, Any
import logging # Ensure logging is available for all modules
from .base_agent import Agent
from .user_interface_agent import UserInterfaceAgent
from .bigquery_data_retrieval_agent import BigQueryDataRetrievalAgent 
from .data_quality_agent import DataQualityValidationAgent
from .power_consumption_agent import PowerConsumptionCalculationAgent
from .statistical_analysis_agent import StatisticalAnalysisAgent
from .insights_optimization_agent import InsightsOptimizationAgent

class RootOrchestratorAgent(Agent):
    """
    Coordinates all sub-agents and manages the overall workflow execution
    for power consumption analysis.
    """

    def __init__(self, use_mock_bq: bool = True):
        super().__init__("RootOrchestratorAgent")
        self.ui_agent = UserInterfaceAgent()
        self.bq_retrieval_agent = BigQueryDataRetrievalAgent(use_mock=use_mock_bq)
        self.dq_agent = DataQualityValidationAgent()
        self.power_calc_agent = PowerConsumptionCalculationAgent()
        self.stat_analysis_agent = StatisticalAnalysisAgent()
        self.insights_agent = InsightsOptimizationAgent()
        self.logger.info("All sub-agents initialized.")

    def execute(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs the full analysis workflow.
        """
        self.logger.info("Starting multi-agent power consumption analysis workflow...")
        
        final_report: Dict[str, Any] = {"status": "initiated", "error_message": None}

        try:
            ui_result = self.ui_agent.execute()
            if ui_result.get("status") == "error":
                raise Exception(f"User Interface Agent failed: {ui_result.get('error_message')}")
            user_params = ui_result.get("user_input", {})
            final_report["user_inputs"] = user_params
            self.logger.info(f"User input: {user_params}")

            bq_params = {
                "location_id": user_params.get("location_id"),
                "pipeline_line": user_params.get("pipeline_line_number"),
                "start_date": user_params.get("start_date"),
                "end_date": user_params.get("end_date"),
                "aggregation_interval": "raw",
                "tags_to_retrieve": ["pump_status", "pressure", "flowrate", "dra_injection_rate", "tank_level"]
            }
            retrieval_result = self.bq_retrieval_agent.execute(bq_params)
            if retrieval_result.get("status") == "error":
                raise Exception(f"BQ Retrieval Agent failed: {retrieval_result.get('error_message')}")
            sensor_data_df = retrieval_result.get("sensor_data")
            if sensor_data_df is None or sensor_data_df.empty:
                raise Exception("No sensor data retrieved. Cannot proceed.")
            self.logger.info(f"Retrieved {len(sensor_data_df)} sensor records.")

            dq_result = self.dq_agent.execute({"sensor_data": sensor_data_df})
            if dq_result.get("status") == "error": # Log but don't necessarily stop
                self.logger.error(f"Data Quality Agent encountered issues: {dq_result.get('error_message')}")
            final_report["data_quality_assessment"] = dq_result.get("data_quality_report", {})
            final_report["data_quality_recommendations"] = dq_result.get("recommendations", [])
            self.logger.info(f"DQ summary: {final_report['data_quality_assessment'].get('summary')}")

            power_calc_result = self.power_calc_agent.execute({"sensor_data": sensor_data_df.copy()})
            if power_calc_result.get("status") == "error":
                raise Exception(f"Power Calc Agent failed: {power_calc_result.get('error_message')}")
            power_consumption_df = power_calc_result.get("power_consumption_data")
            final_report["power_consumption_summary"] = power_calc_result.get("summary_metrics")
            if power_consumption_df is None or power_consumption_df.empty:
                raise Exception("Power consumption could not be calculated.")
            self.logger.info(f"Power calc summary: {final_report['power_consumption_summary']}")

            stat_input = {"power_consumption_data": power_consumption_df.copy()}
            stat_analysis_result = self.stat_analysis_agent.execute(stat_input)
            if stat_analysis_result.get("status") == "error": # Log but don't necessarily stop
                 self.logger.error(f"Stats Agent failed: {stat_analysis_result.get('error_message')}")
            final_report["statistical_analysis_highlights"] = stat_analysis_result.get("statistical_summary", {})
            # Store data needed for visualization hints if UI agent were to use them directly
            final_report["visualization_data_store"] = {
                "time_series_aggregations": stat_analysis_result.get("time_series_aggregations", {}),
                "correlation_analysis": stat_analysis_result.get("correlation_analysis", {}),
                "efficiency_trends": stat_analysis_result.get("efficiency_trends", {})
            }
            final_report["visualization_hints"] = stat_analysis_result.get("visualization_hints", [])
            self.logger.info(f"Stats analysis generated for {len(final_report['statistical_analysis_highlights'])} groups.")

            insights_input = {
                "statistical_summary": final_report["statistical_analysis_highlights"],
                "efficiency_trends": stat_analysis_result.get("efficiency_trends", {}), # Pass the actual data, not just summary
                "data_quality_report": final_report["data_quality_assessment"],
            }
            insights_result = self.insights_agent.execute(insights_input)
            if insights_result.get("status") == "error": # Log but don't necessarily stop
                self.logger.error(f"Insights Agent failed: {insights_result.get('error_message')}")
            final_report["optimization_roadmap"] = {"opportunities": insights_result.get("optimization_opportunities", [])}
            final_report["executive_summary_points"] = insights_result.get("executive_summary_points", [])
            final_report["risk_assessment_notes"] = insights_result.get("risk_assessment_notes", [])
            self.logger.info(f"Insights generated {len(final_report['optimization_roadmap']['opportunities'])} opportunities.")

            # Consolidate executive summary
            exec_summary_str = "Executive Summary:\n"
            if not final_report["executive_summary_points"]:
                exec_summary_str += "- Analysis complete. No specific high-priority points generated. Review detailed sections."

            for point in final_report["executive_summary_points"]:
                exec_summary_str += f"- {point}\n"
            if final_report.get("power_consumption_summary"):
                pcs = final_report["power_consumption_summary"]
                exec_summary_str += f"- Approx total energy: {pcs.get('total_electrical_energy_kwh_approx','N/A')} kWh. Avg power: {pcs.get('average_electrical_power_kw','N/A')} kW.
"
            final_report["executive_summary"] = exec_summary_str
            final_report["status"] = "success"

        except Exception as e:
            self.logger.critical(f"Workflow terminated with error: {e}", exc_info=True)
            final_report["status"] = "error"
            final_report["error_message"] = str(e)
        
        self.logger.info(f"Workflow finished with status: {final_report['status']}")
        self.ui_agent.display_results(final_report) # Display final consolidated report or error
        return final_report

if __name__ == '__main__':
    # To run the orchestrator:
    # python -m dra_power_analysis.agents.root_orchestrator_agent
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    orchestrator = RootOrchestratorAgent(use_mock_bq=True)
    
    print("Attempting to run Root Orchestrator Agent workflow...")
    print("If non-interactive, UI agent might default or fail input.")
    print("Mock Inputs: Location: STN_A001, Pipeline: PL123, Dates: 2023-01-01 to 2023-01-01")

    final_workflow_result = orchestrator.execute()

    # Summary already printed by ui_agent.display_results() via orchestrator
    if final_workflow_result.get("status") == "success":
        print("\n--- Orchestrator Test: Workflow COMPLETED SUCCESSFULLY (mock run) ---")
    else:
        print("\n--- Orchestrator Test: Workflow FAILED (mock run) ---")
        # Error details would have been printed by ui_agent or logged.
