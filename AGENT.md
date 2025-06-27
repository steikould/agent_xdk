# Agent Interaction Protocols and Responsibilities (AGENTS.md)

This document outlines the guidelines, responsibilities, and interaction protocols for the AI agents working on the Multi-Agent Pipeline Power Consumption Analysis System. Adherence to these protocols is crucial for effective collaboration and system integrity.

## General Principles for All Agents

1.  **Modularity**: Each agent should be self-contained and responsible for a specific set of tasks.
2.  **Clear Interfaces**: Agents must communicate through well-defined interfaces, primarily by passing structured data (e.g., dictionaries or data objects).
3.  **Error Handling**: Each agent is responsible for handling its internal errors gracefully and reporting them to the orchestrator or the calling agent.
4.  **Logging**: All agents should implement detailed logging for actions, decisions, errors, and data transformations. This is crucial for debugging and auditing.
5.  **Idempotency**: Where possible, agent actions should be idempotent, meaning that performing an action multiple times should have the same effect as performing it once.
6.  **Configuration**: Agents should be configurable (e.g., through parameters passed during initialization or method calls) rather than hardcoding values.
7.  **Adherence to `AGENTS.md`**: All agents must operate within the guidelines set forth in this document. Any deviation required by specific tasks must be explicitly approved by the user.

## Agent-Specific Responsibilities and Protocols

### 1. Root Orchestrator Agent
    *   **Primary Role**: Coordinate all sub-agents and manage workflow execution.
    *   **Responsibilities**:
        *   Parse and validate initial user inputs (delegating detailed validation to the User Interface Agent).
        *   Orchestrate the sequence of operations by invoking other agents in the correct order.
        *   Manage the flow of data between agents.
        *   Implement top-level error management and retry logic for agent execution.
        *   Consolidate results from various agents into a comprehensive report.
        *   Communicate final results or critical errors back to the user (via the User Interface Agent).
    *   **Interaction**:
        *   Receives initial trigger to start the workflow.
        *   Calls the `UserInterfaceAgent` to get user input.
        *   Calls data management and analysis agents sequentially, passing necessary data.
        *   Receives processed data or error statuses from sub-agents.

### 2. User Interface Agent
    *   **Primary Role**: Handle all user interactions and input validation.
    *   **Responsibilities**:
        *   Accept user inputs: location ID, pipeline line number, date range (start/end dates).
        *   Perform comprehensive validation of these inputs:
            *   Location existence (e.g., against a known list or configuration).
            *   Pipeline line number format and existence.
            *   Date range validity (not future dates, not exceeding data retention policies, start date before end date).
        *   Provide clear, user-friendly feedback on input validation errors.
        *   Format and present final results, summaries, and errors to the user.
    *   **Interaction**:
        *   Called by the `RootOrchestratorAgent` to obtain input.
        *   Returns validated input data or error messages to the orchestrator.
        *   Called by the `RootOrchestratorAgent` to display final outputs or critical system errors.

### 3. BigQuery Data Retrieval Agent
    *   **Primary Role**: Execute optimized queries against the golden IoT sensor data table in BigQuery.
    *   **Responsibilities**:
        *   Construct and execute SQL queries based on parameters (location, pipeline, date range, aggregation interval).
        *   Retrieve specific data fields: pump status, flowrates (upstream/downstream), tag interface, and unit metadata.
        *   Implement query optimizations (e.g., use of partitioned fields, clustering, appropriate `WHERE` clauses).
        *   Handle BigQuery API errors (e.g., authentication, query failures, network issues).
        *   Implement mechanisms for result caching if specified by requirements.
        *   Ensure proper handling of time zones if relevant to the dataset.
    *   **Interaction**:
        *   Called by the `RootOrchestratorAgent` with validated parameters.
        *   Returns a structured dataset (e.g., list of dictionaries, Pandas DataFrame) or error information.

### 4. Data Quality and Validation Agent
    *   **Primary Role**: Assess data integrity of retrieved sensor data and provide recommendations.
    *   **Responsibilities**:
        *   Perform validation checks:
            *   Identify missing sensor readings or data gaps.
            *   Detect anomalous pump status transitions.
            *   Flag unrealistic flowrate values.
            *   Check for sensor calibration drift indicators (if logic is defined).
            *   Validate timestamp consistency and chronological order.
        *   Generate a data quality report.
        *   Provide recommendations for data cleaning, sensor maintenance, or operational adjustments.
    *   **Interaction**:
        *   Called by the `RootOrchestratorAgent` with the raw data from `BigQueryDataRetrievalAgent`.
        *   Returns a data quality assessment report and the (potentially cleaned or annotated) data.

### 5. Power Consumption Calculation Agent
    *   **Primary Role**: Convert sensor data to power consumption metrics.
    *   **Responsibilities**:
        *   Apply pump efficiency factors from static lookup tables or configurations.
        *   Calculate instantaneous and aggregated power consumption using relevant engineering formulas (e.g., hydraulic power, motor power).
        *   Account for VSD efficiency, pipe diameter changes, pressure drops, DRA injection rates, and mixing energy if data and formulas are available.
        *   Maintain transparency in calculations (e.g., log formulas used and intermediate values).
    *   **Interaction**:
        *   Called by the `RootOrchestratorAgent` with validated sensor data and necessary configuration (e.g., pump curves, efficiency tables).
        *   Returns calculated power consumption data.

### 6. Statistical Analysis Agent
    *   **Primary Role**: Perform time-series analysis and statistical computations on power and operational data.
    *   **Responsibilities**:
        *   Perform time-series aggregation (hourly, daily, weekly averages).
        *   Calculate statistical measures (mean, median, std deviation, percentiles).
        *   Conduct trend analysis and seasonal decomposition (if applicable).
        *   Analyze correlations (e.g., flowrates vs. power consumption).
        *   Track pump efficiency trends over time.
        *   Identify energy consumption patterns and peak demand.
        *   Prepare data suitable for generating visualizations (e.g., time-series plots, histograms, scatter plots).
    *   **Interaction**:
        *   Called by the `RootOrchestratorAgent` with power consumption data and relevant operational data.
        *   Returns statistical analysis results and data formatted for visualization.

### 7. Insights and Optimization Agent
    *   **Primary Role**: Generate actionable recommendations for operational improvements.
    *   **Responsibilities**:
        *   Analyze data to identify optimal DRA injection timing and rates.
        *   Detect patterns indicating pump performance degradation.
        *   Recommend maintenance windows based on efficiency trends.
        *   Suggest operational parameter adjustments for energy savings.
        *   Provide a ranked list of optimization opportunities with estimated savings and risk assessment.
    *   **Interaction**:
        *   Called by the `RootOrchestratorAgent` with results from the `StatisticalAnalysisAgent` and `PowerConsumptionCalculationAgent`.
        *   Returns an optimization roadmap and actionable insights.

## Data Exchange Format

*   Unless specified otherwise, agents should exchange data using Python dictionaries.
*   Keys in dictionaries should be descriptive and consistently named.
*   Timestamps should be in a consistent format, preferably ISO 8601 with UTC timezone.

## Error Reporting and Flow

1.  If an agent encounters an error it cannot handle, it should log the error in detail and propagate an error status or exception to its calling agent (typically the `RootOrchestratorAgent`).
2.  The `RootOrchestratorAgent` is responsible for deciding whether to retry an operation, halt the workflow, or proceed with partial results (if applicable and safe).
3.  Critical errors that halt the workflow should be communicated to the user via the `UserInterfaceAgent` with as much context as possible.

## Code Development Guidelines for Agents

*   **Clarity and Readability**: Code should be well-commented, with clear variable and function names.
*   **Unit Tests**: Each agent should have corresponding unit tests for its core logic.
*   **Dependencies**: Minimize external dependencies. If new dependencies are required, they must be added to `requirements.txt`.
*   **Configuration Management**: Avoid hardcoding values. Use configuration files or pass parameters.
*   **Google Cloud SDK Usage**:
    *   Use official Google Cloud client libraries.
    *   Implement proper authentication (e.g., service accounts, Application Default Credentials).
    *   Follow best practices for API usage (e.g., error handling, retries for transient errors).

## Future Agent Integration

If new agents are added to the system, this `AGENTS.md` file must be updated to include their roles, responsibilities, and interaction protocols. The `RootOrchestratorAgent` will likely need modification to integrate new agents into the workflow.