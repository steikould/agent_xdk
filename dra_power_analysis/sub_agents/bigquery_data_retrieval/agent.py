from typing import Dict, Any, List, Optional, ClassVar
import logging
import datetime
import asyncio
import functools

# Pydantic imports for defining schemas and configurations
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict, model_validator # Import model_validator

# Google ADK imports for agent and tool definitions
from google.adk.agents import BaseAgent
from google.adk.tools import FunctionTool, ToolContext 

# Google Cloud BigQuery import
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None # Ensure bigquery is None if import available

# --- Pydantic Models for Tool Input and Output Schemas (unchanged) ---
class QueryBigQueryDataInput(BaseModel):
    start_time: datetime.datetime = Field(..., description="The UTC start timestamp for data retrieval (inclusive). Example: '2023-01-01T00:00:00Z'")
    end_time: datetime.datetime = Field(..., description="The UTC end timestamp for data retrieval (inclusive). Example: '2023-01-02T23:59:59Z'")
    location_id: str = Field(..., description="The ID of the physical location (e.g., 'STN_A001', 'LOC_NYC').")
    line_id: str = Field(..., description="The identifier for the specific line or pipeline (e.g., 'PL123', 'Line_B').")

    @field_validator('end_time')
    @classmethod
    def validate_end_time_after_start_time(cls, v, info):
        if 'start_time' in info.data and v < info.data['start_time']:
            raise ValueError("End time cannot be before start time.")
        return v

class QueryResultRow(BaseModel):
    timestamp: datetime.datetime
    tag_name: str
    value: float
    unit: str
    quality: str
    location_id: str
    pipeline_id: str

class QueryBigQueryDataOutput(BaseModel):
    status: str = Field(..., description="Status of the data retrieval ('success' or 'error').")
    data: Optional[List[QueryResultRow]] = Field(None, description="List of queried data rows if retrieval was successful.")
    error_message: Optional[str] = Field(None, description="Error message if retrieval failed.")
    query_executed: Optional[str] = Field(None, description="The SQL query that was executed (for debugging/transparency).")

# --- The actual function that will be wrapped as a FunctionTool (unchanged) ---
async def _query_bigquery_tool_func(
    tool_input: QueryBigQueryDataInput,
    bq_client: bigquery.Client,
    logger: logging.Logger,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> QueryBigQueryDataOutput:
    """
    Queries historical sensor data from a specified BigQuery table using parameterized SQL,
    filtered by start_time, end_time, location_id, and line_id. Returns structured data.
    """
    if not BIGQUERY_AVAILABLE:
        return QueryBigQueryDataOutput(
            status="error",
            error_message="Google Cloud BigQuery library not found. Cannot execute real BigQuery queries.",
            query_executed="N/A"
        )

    start_time = tool_input.start_time
    end_time = tool_input.end_time
    location_id = tool_input.location_id
    line_id = tool_input.line_id

    table_fqn = f"`{project_id}.{dataset_id}.{table_id}`"

    sql_query = f"""
    SELECT
        *
    FROM
        {table_fqn}
    WHERE
        timestamp BETWEEN @start_time AND @end_time
        AND location_id = @location_id
        AND pipeline_id = @line_id
    ORDER BY
        timestamp, tag_name
    """

    query_params = [
        bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time),
        bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time),
        bigquery.ScalarQueryParameter("location_id", "STRING", location_id),
        bigquery.ScalarQueryParameter("line_id", "STRING", line_id),
    ]

    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    
    logger.debug(f"Constructed SQL Query:\n{sql_query.strip()}")
    logger.debug(f"Constructed Query Parameters: {[(p.name, p.value, p.type_) for p in query_params]}")

    try:
        logger.info(f"Initiating BigQuery query for location '{location_id}', line '{line_id}' from {project_id}.{dataset_id}.{table_id}...")
        
        loop = asyncio.get_running_loop()
        query_job = await loop.run_in_executor(
            None,
            lambda: bq_client.query(sql_query, job_config=job_config)
        )
        
        results = await loop.run_in_executor(
            None,
            lambda: list(query_job.result())
        )

        if query_job.error_result:
            error_info = query_job.error_result
            return QueryBigQueryDataOutput(
                status="error",
                error_message=f"BigQuery query failed: {error_info.get('message', 'Unknown BQ error')}",
                query_executed=sql_query
            )

        logger.info(f"Successfully retrieved {len(results)} rows from BigQuery.")

        queried_data_list = []
        for row in results:
            record = row.to_api_repr()
            try:
                if 'timestamp' in record and isinstance(record['timestamp'], str):
                    record['timestamp'] = datetime.datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                queried_data_list.append(QueryResultRow(**record))
            except ValidationError as e:
                logger.warning(f"Skipping invalid data record due to validation error: {record} - {e}")
            except Exception as e:
                logger.warning(f"Skipping record due to unexpected error during conversion: {record} - {e}")

        return QueryBigQueryDataOutput(
            status="success",
            data=queried_data_list,
            query_executed=sql_query,
        )
    except Exception as e:
        logger.exception("An unexpected error occurred during BigQuery data retrieval.")
        return QueryBigQueryDataOutput(
            status="error",
            error_message=f"An unexpected error occurred during BigQuery data retrieval: {type(e).__name__} - {e}",
            query_executed=sql_query
        )

# --- Attach schemas directly to the function, as FunctionTool might infer them ---
_query_bigquery_tool_func.input_schema = QueryBigQueryDataInput
_query_bigquery_tool_func.output_schema = QueryBigQueryDataOutput


# --- BigQueryDataAgent Class (FIXED with 'tools' field) ---
class BigQueryDataAgent(BaseAgent):
    """
    A google-adk agent that provides a tool to query sensor data from BigQuery
    using parameterized SQL queries.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Make these required or use env vars
    project_id: str = Field(..., description="The GCP Project ID for BigQuery.")
    dataset_id: str = Field(..., description="The BigQuery Dataset ID.")
    table_id: str = Field(..., description="The BigQuery Table ID.")
    use_mock: bool = Field(default=False, description="Set to True to use a mock BigQuery client for testing.")

    # Private fields
    _client: Optional[bigquery.Client] = None
    _logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(__name__), exclude=True)
    
    # Public tools field - don't exclude if it needs to be accessible
    tools: List[FunctionTool] = Field(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._logger = logging.getLogger(self.name or __name__)

    @model_validator(mode='after')
    def setup_client_and_tools(self) -> "BigQueryDataAgent":
        """Initialize BigQuery client and create tools after model validation."""
        
        if self.use_mock:
            raise ValueError("Mock client is not supported in this version of BigQueryDataAgent.")
        
        if not BIGQUERY_AVAILABLE:
            raise ImportError("Required 'google-cloud-bigquery' library not found. Please install it.")
        
        # Initialize BigQuery client
        try:
            self._client = bigquery.Client(project=self.project_id)
            self._logger.info(f"Initialized Google BigQuery Client for project: {self.project_id}")
        except Exception as e:
            self._logger.exception(f"Failed to initialize BigQuery client: {e}")
            raise RuntimeError(f"BigQuery client initialization failed: {e}")
        
        # Create the tool
        query_tool = FunctionTool(
            func=functools.partial(
                _query_bigquery_tool_func,
                bq_client=self._client,
                logger=self._logger,
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                table_id=self.table_id,
            )
        )
        
        self.tools = [query_tool]
        return self

# --- Local Testing Block (unchanged) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    YOUR_GCP_PROJECT_ID = "your-gcp-project-id"
    YOUR_DATASET_ID = "your_dataset_id"
    YOUR_TABLE_ID = "your_table_id"

    if YOUR_GCP_PROJECT_ID == "your-gcp-project-id":
        print("\n--- WARNING: Please update 'YOUR_GCP_PROJECT_ID', 'YOUR_DATASET_ID', and 'YOUR_TABLE_ID' in the testing block with your actual BigQuery details. ---")
        print("Skipping live BigQuery test due to unconfigured project details.")
    else:
        print(f"\nInitializing BigQueryDataAgent for real BigQuery: {YOUR_GCP_PROJECT_ID}.{YOUR_DATASET_ID}.{YOUR_TABLE_ID}")
        try:
            data_agent = BigQueryDataAgent(
                project_id=YOUR_GCP_PROJECT_ID,
                dataset_id=YOUR_DATASET_ID,
                table_id=YOUR_TABLE_ID,
                use_mock=False
            )
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"Agent initialization failed: {e}. Cannot proceed with live BigQuery tests.")
            exit()

        async def run_live_test_cases():
            print(f"\n--- Test Case 1: Live Data Retrieval (Real BigQuery) ---")
            test_input_live = QueryBigQueryDataInput(
                start_time=datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                end_time=datetime.datetime(2024, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc),
                location_id="STN_A001",
                line_id="PL123",
            )

            tool_func_with_deps = data_agent.query_bigquery_data_tool.func
            
            print(f"Attempting to query BigQuery for location='{test_input_live.location_id}', line='{test_input_live.line_id}' from {test_input_live.start_time} to {test_input_live.end_time}...")
            result: QueryBigQueryDataOutput = await tool_func_with_deps(test_input_live)

            if result.status == "success":
                print("Test Case 1: Live Data retrieval successful.")
                data_rows = result.data
                print(f"Retrieved {len(data_rows) if data_rows else 0} data points.")
                if data_rows:
                    print("Sample data (first 5 rows):")
                    for i, row in enumerate(data_rows[:5]):
                        print(f"  {i+1}: {row.model_dump_json()}")
                else:
                    print("No data retrieved for Test Case 1. This might be expected if no data matches the criteria.")
                print(f"Query Executed:\n{result.query_executed}")
            else:
                print(f"Test Case 1: Live Data retrieval failed with error: {result.error_message}")
                print(f"Query Executed:\n{result.query_executed}")

            print(f"\n--- Test Case 2: Invalid Date Range (Pydantic validation) ---")
            try:
                invalid_time_input = QueryBigQueryDataInput(
                    start_time=datetime.datetime(2024, 6, 10, 10, 0, 0, tzinfo=datetime.timezone.utc),
                    end_time=datetime.datetime(2024, 6, 10, 8, 0, 0, tzinfo=datetime.timezone.utc),
                    location_id="STN_C003",
                    line_id="Line_X",
                )
                result_invalid = await tool_func_with_deps(invalid_time_input)
                print(f"Test Case 2: Unexpected success with invalid date range: {result_invalid.status}")
            except ValidationError as e:
                print(f"Test Case 2: Successfully caught Pydantic Validation Error for invalid date range:\n{e.errors()}")
            except Exception as e:
                print(f"Test Case 2: Caught unexpected error: {e}")

        asyncio.run(run_live_test_cases())
        print("\nAll BigQueryDataAgent tests completed.")