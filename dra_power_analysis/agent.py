import logging
from google.adk.agents import Agent
from dra_power_analysis import prompt

from dra_power_analysis.sub_agents.bigquery_data_retrieval.agent import BigQueryDataAgent
# from dra_power_analysis.sub_agents.data_quality.agent import dq_agent
# from dra_power_analysis.sub_agents.insights_optimization.agent import insights_agent
# from dra_power_analysis.sub_agents.power_consumption.agent import power_calc_agent
# from dra_power_analysis.sub_agents.statistical_analysis.agent import stat_analysis_agent
# from dra_power_analysis.sub_agents.user_interface.agent import ui_agent_output

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

model_name = "gemini-2.0-flash"

data_agent_instance = BigQueryDataAgent(
    name="DataQueryAgent",
    description="Agent capable of querying sensor data from BigQuery.",
    project_id="my-gcp-project", # Replace with your actual project ID for real BQ
    dataset_id="sensor_data_prod",
    table_id="pipeline_metrics",
    use_mock=False # Set to True for local development without GCP connection
)

root_agent = Agent(
    name="PowerAnalysisInitiator",
    model=model_name,
    description="Guides the energy pipeline analyst in initiating power consumption analysis and optimization tasks.",
    instruction=prompt.ROOT_AGENT_INSTR,
    # generate_content_config=types.GenerateContentConfig(
    #     temperature=0,
    # # ),
    # tools=[append_to_state],
    sub_agents=[
        data_agent_instance
        # dq_agent,
        # insights_agent,
        # power_calc_agent,
        # stat_analysis_agent,
        # ui_agent_output
        ]
    # before_agent_callback=_load_some_object,
)
