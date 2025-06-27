import asyncio
from typing import Dict, Any, List, AsyncGenerator
import logging

# ADK Imports
from google.adk.agents import SequentialAgent, InvocationContext # BaseAgent removed as it's not directly used here
from google.adk.events import Event # EventActions removed as it's not directly used here
from google.adk.runtime import Session

# Local Agent Class Imports (Now ADK BaseAgent subclasses)
from dra_power_analysis.agents.user_interface_agent import UserInterfaceAgent
from dra_power_analysis.agents.bigquery_data_retrieval_agent import BigQueryDataRetrievalAgent
from dra_power_analysis.agents.data_quality_agent import DataQualityValidationAgent
from dra_power_analysis.agents.power_consumption_agent import PowerConsumptionCalculationAgent
from dra_power_analysis.agents.statistical_analysis_agent import StatisticalAnalysisAgent
from dra_power_analysis.agents.insights_optimization_agent import InsightsOptimizationAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

class ADKRootOrchestratorAgent(SequentialAgent):
    """
    Orchestrates the power consumption analysis workflow using ADK's SequentialAgent.
    """

    def __init__(self, use_mock_bq: bool = True, **kwargs):
        
        # Instantiate all sub-agents with their names and descriptions
        # Descriptions are simplified from AGENT.md
        ui_agent_input = UserInterfaceAgent(
            name="UserInputAgent",
            description="Handles user input gathering and validation.",
            role="input"
        )
        bq_retrieval_agent = BigQueryDataRetrievalAgent(
            name="BigQueryDataRetrievalAgent",
            description="Executes optimized queries against BigQuery.",
            use_mock=use_mock_bq
        )
        dq_agent = DataQualityValidationAgent(
            name="DataQualityValidationAgent",
            description="Assesses data integrity of retrieved sensor data."
        )
        power_calc_agent = PowerConsumptionCalculationAgent(
            name="PowerConsumptionCalculationAgent",
            description="Converts sensor data to power consumption metrics."
        )
        stat_analysis_agent = StatisticalAnalysisAgent(
            name="StatisticalAnalysisAgent",
            description="Performs time-series analysis and statistical computations."
        )
        insights_agent = InsightsOptimizationAgent(
            name="InsightsOptimizationAgent",
            description="Generates actionable recommendations for operations."
        )
        ui_agent_output = UserInterfaceAgent(
            name="UserOutputAgent",
            description="Handles presentation of final results to the user.",
            role="output"
        )

        sub_agents_list = [
            ui_agent_input,
            bq_retrieval_agent,
            dq_agent,
            power_calc_agent,
            stat_analysis_agent,
            insights_agent,
            ui_agent_output
        ]

        super().__init__(
            name="RootOrchestratorSequentialAgent",
            description="Orchestrates the entire power consumption analysis pipeline sequentially.",
            sub_agents=sub_agents_list,
            **kwargs
        )
        self.logger.info(f"'{self.name}' initialized with {len(self.sub_agents)} sub-agents.")

    # The _run_async_impl for SequentialAgent is provided by ADK.
    # It will execute its sub_agents in order.

async def main_async(use_mock_bq: bool = True):
    """Asynchronous main function to run the ADK orchestrator."""
    logger.info("Attempting to run ADK Root Orchestrator Agent workflow...")

    orchestrator = ADKRootOrchestratorAgent(use_mock_bq=use_mock_bq)

    # Initial state could include global configurations if needed by multiple agents,
    # but in this flow, the first agent (UserInputAgent) populates the necessary initial state.
    session = Session(state={})

    final_workflow_result = {}
    workflow_error = None

    try:
        # The 'request' to a SequentialAgent can be a simple trigger string or empty
        # if the flow doesn't depend on an initial LLM interpretation by the orchestrator itself.
        # Here, the UserInputAgent handles the initial interaction.
        async for event in orchestrator.run(request="Start power consumption analysis pipeline", session=session):
            logger.info(f"Event from '{event.author}': {event.content.parts[0].text if event.content and event.content.parts else 'Event has no textual content.'}")
            if event.actions and event.actions.escalate:
                logger.error(f"Workflow escalation by '{event.author}'. Halting.")
                workflow_error = f"Escalation by {event.author}: {event.content.parts[0].text if event.content and event.content.parts else 'No specific error message.'}"
                break

        if workflow_error:
            final_workflow_result = session.state
            final_workflow_result["status"] = "error"
            # Attempt to pull a more specific error if set by an agent
            final_workflow_result["error_message"] = workflow_error
            # If the UI output agent ran before escalation, it might have already printed.
            # If escalation happened before UI output, we might want to print the error here.
            # Check if final_consolidated_report exists and has an error status
            consolidated_report = session.state.get("final_consolidated_report", {})
            if consolidated_report.get("status") == "error" and "error_message" in consolidated_report:
                 final_workflow_result["error_message"] = consolidated_report["error_message"]
            elif "error_message_user_input" in session.state: # Check for early UI error
                 final_workflow_result["error_message"] = session.state["error_message_user_input"]


        else:
            final_workflow_result = session.state.get("final_consolidated_report", session.state)
            # Ensure status is success if no error/escalation occurred and report exists
            if "status" not in final_workflow_result or final_workflow_result.get("status") != "error":
                final_workflow_result["status"] = "success"


    except Exception as e:
        logger.critical(f"Critical error during ADK orchestrator execution: {e}", exc_info=True)
        final_workflow_result = session.state # Get whatever state existed
        final_workflow_result["status"] = "error"
        final_workflow_result["error_message"] = f"Orchestrator critical error: {str(e)}"

    # Final outcome logging
    if final_workflow_result.get("status") == "success":
        logger.info("\n--- ADK Orchestrator: Workflow COMPLETED SUCCESSFULLY ---")
        # Results are displayed by the UserOutputAgent if the workflow reached it.
    else:
        logger.error(f"\n--- ADK Orchestrator: Workflow FAILED or HALTED ---")
        logger.error(f"Error: {final_workflow_result.get('error_message', 'Unknown error or early halt.')}")
        # If the UserOutputAgent didn't run or if it's an orchestrator level error,
        # we might need to explicitly print the error details if not already handled.
        # The UserOutputAgent is designed to display final_consolidated_report which includes error status.

if __name__ == '__main__':
    # To run the orchestrator:
    # python -m main
    # (if main.py is in the root of a package 'dra_power_analysis',
    #  you might run `python -m dra_power_analysis.main` from one level up,
    #  or adjust PYTHONPATH if running main.py directly from its directory)
    
    # For simplicity, assuming main.py can be run directly due to relative imports of agents.
    # If running as `python main.py` from `dra_power_analysis/agents/` context, imports might need adjustment.
    # The current structure with `from dra_power_analysis.agents...` implies `main.py` is outside `agents` dir,
    # or `dra_power_analysis` is in PYTHONPATH.

    # We'll assume `python main.py` is run from the directory containing `dra_power_analysis` package.
    # If `main.py` is inside `dra_power_analysis` and is the main entry point:
    # One way to handle imports if main.py is inside dra_power_analysis:
    # Add project root to sys.path if needed, or use `python -m dra_power_analysis.main`

    # The current setup implies main.py is at the root, alongside the dra_power_analysis directory.
    # Example:
    # project_root/
    #  main.py
    #  dra_power_analysis/
    #    __init__.py
    #    agents/
    #      __init__.py
    #      user_interface_agent.py
    #      ...
    # In this case, `from dra_power_analysis.agents...` should work when running `python main.py` from `project_root`.

    asyncio.run(main_async(use_mock_bq=True)) # Set use_mock_bq as needed
    logger.info("ADK Workflow execution finished.")
