# Multi-Agent Pipeline Power Consumption Analysis System

This project implements a multi-agent system for analyzing power consumption at fuel pipeline operations, specifically focusing on Drag Reducing Agent (DRA) injection stations.

## Overview

The system utilizes a series of specialized agents, coordinated by a Root Orchestrator, to perform tasks ranging from data retrieval and validation to power consumption calculation, statistical analysis, and optimization recommendations. It is designed to integrate with Google Cloud services, particularly BigQuery for data storage and retrieval.

## System Architecture

The system is composed of the following key agents:

*   **Root Orchestrator Agent**: Manages the overall workflow and coordinates sub-agents.
*   **User Interface Agent**: Handles all user interactions, input gathering, and results presentation.
*   **BigQuery Data Retrieval Agent**: Fetches sensor data from BigQuery (currently uses a mock for standalone operation).
*   **Data Quality and Validation Agent**: Assesses the integrity of the retrieved data.
*   **Power Consumption Calculation Agent**: Calculates power usage based on sensor data and engineering formulas.
*   **Statistical Analysis Agent**: Performs time-series analysis and computes statistical metrics.
*   **Insights and Optimization Agent**: Generates actionable recommendations for operational improvements.

## Project Structure

dra_power_analysis/ ├── agents/ │ ├── init.py │ ├── base_agent.py │ ├── bigquery_data_retrieval_agent.py │ ├── data_quality_agent.py │ ├── insights_optimization_agent.py │ ├── power_consumption_agent.py │ ├── root_orchestrator_agent.py │ ├── statistical_analysis_agent.py │ └── user_interface_agent.py ├── data/ (placeholder for mock data, lookup tables) ├── utils/ (placeholder for shared utilities) ├── init.py └── main.py

AGENTS.md README.md requirements.txt .gitignore LICENSE


## Setup and Usage

1.  **Prerequisites**:
    *   Python 3.8+
    *   Ensure all dependencies from `requirements.txt` are installed:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Running the System**:
    *   Navigate to the parent directory of the `dra_power_analysis` folder.
    *   Run the main application module:
        ```bash
        python -m dra_power_analysis.main
        ```
    *   The system will prompt for inputs (Location ID, Pipeline Line Number, Start Date, End Date).
    *   For testing with the current mock data, suggested inputs are:
        *   Location ID: `STN_A001`
        *   Pipeline Line Number: `PL123`
        *   Start Date: `2023-01-01`
        *   End Date: `2023-01-02`

3.  **Google Cloud Integration (Future)**:
    *   To use with a real BigQuery instance:
        *   Set `use_mock_bq=False` in `dra_power_analysis/agents/root_orchestrator_agent.py` (or make this configurable).
        *   Ensure your Google Cloud SDK is authenticated (e.g., via `gcloud auth application-default login` or a service account JSON key file pointed to by `GOOGLE_APPLICATION_CREDENTIALS`).
        *   Update `PROJECT_ID`, `DATASET_ID`, and `GOLDEN_TABLE_ID` in `bigquery_data_retrieval_agent.py` to match your BigQuery setup.
        *   The structure of your BigQuery table should align with the expectations of the `_build_master_query` method in `bigquery_data_retrieval_agent.py`.

## Agent Details

For detailed responsibilities and interaction protocols of each agent, refer to `AGENTS.md`.

## Testing

Each agent module (`*.py` files within `dra_power_analysis/agents/`) contains an `if __name__ == '__main__':` block with basic unit-like tests for its core functionality. These can be run directly, e.g.:

```bash
python -m dra_power_analysis.agents.user_interface_agent
(Ensure that when running individual agent tests, the Python path allows imports from .base_agent etc. Running with -m from the project root's parent directory helps manage this.)

Further Development
Implement actual BigQuery client logic and robust query building.
Develop more sophisticated data validation and cleaning rules.
Enhance engineering calculations in the Power Consumption Agent.
Add advanced statistical models (e.g., forecasting, anomaly detection).
Integrate visualization libraries for graphical output.
Deploy agents using cloud-native services (e.g., Cloud Functions, Pub/Sub).
Develop a more comprehensive testing suite (unit and integration tests).
