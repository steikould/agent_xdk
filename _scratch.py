ui_agent_input = Agent(
    name="UserInputAgent",
    model=model_name,
    description="Handles user input gathering and validation.",
    instruction="""
    INSTRUCTIONS:
    - Prompt the user for required parameters.
    - Validate the input format and completeness.
    - Use 'input_collector_tool' to store validated input into the shared state.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="input_collector_tool",
            description="Stores validated user input into shared state.",
            # tool=,
        )
    ],
)

bq_retrieval_agent = Agent(
    name="BigQueryDataRetrievalAgent",
    model=model_name,
    description="Executes optimized queries against BigQuery.",
    instruction="""
    INSTRUCTIONS:
    - Retrieve query parameters from shared state.
    - Execute the BigQuery query using 'bigquery_tool'.
    - Store the result in shared state under 'RAW_DATA'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="bigquery_tool",
            description="Executes BigQuery queries and returns results.",
            # tool=,
        )
    ],
)

dq_agent = Agent(
    name="DataQualityValidationAgent",
    model=model_name,
    description="Assesses data integrity of retrieved sensor data.",
    instruction="""
    INSTRUCTIONS:
    - Retrieve 'RAW_DATA' from shared state.
    - Validate schema, nulls, and expected ranges using 'dq_tool'.
    - Store results in 'VALIDATED_DATA' and log any issues.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="dq_tool",
            description="Performs data quality checks on structured data.",
            # tool=,
        )
    ],
)

power_calc_agent = Agent(
    name="PowerConsumptionCalculationAgent",
    model=model_name,
    description="Converts sensor data to power consumption metrics.",
    instruction="""
    INSTRUCTIONS:
    - Retrieve 'VALIDATED_DATA' from shared state.
    - Apply power conversion formulas using 'power_calc_tool'.
    - Store results in 'POWER_METRICS'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="power_calc_tool",
            description="Calculates power consumption from sensor data.",
            # tool=DummyTool(),
        )
    ],
)

stat_analysis_agent = Agent(
    name="StatisticalAnalysisAgent",
    model=model_name,
    description="Performs time-series analysis and statistical computations.",
    instruction="""
    INSTRUCTIONS:
    - Retrieve 'POWER_METRICS' from shared state.
    - Perform statistical analysis using 'stats_tool'.
    - Store insights in 'ANALYSIS_SUMMARY'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="stats_tool",
            description="Performs statistical analysis on time-series data.",
            # tool=DummyTool(),
        )
    ],
)

insights_agent = Agent(
    name="InsightsOptimizationAgent",
    model=model_name,
    description="Generates actionable recommendations for operations.",
    instruction="""
    INSTRUCTIONS:
    - Retrieve 'ANALYSIS_SUMMARY' from shared state.
    - Generate operational insights using 'insight_tool'.
    - Store recommendations in 'RECOMMENDATIONS'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="insight_tool",
            description="Generates operational recommendations from analysis.",
            # tool=DummyTool(),
        )
    ],
)

ui_agent_output = Agent(
    name="UserOutputAgent",
    model=model_name,
    description="Handles presentation of final results to the user.",
    instruction="""
    INSTRUCTIONS:
    - Retrieve 'RECOMMENDATIONS' from shared state.
    - Format and present results using 'output_tool'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        CrewaiTool(
            name="output_tool",
            description="Formats and presents final output to the user.",
            # tool=DummyTool(),
        )
    ],
)

