from typing import Dict, Any, List, Optional, AsyncGenerator
import logging
import pandas as pd
import numpy as np # Required for mock data generation
import datetime
import time # For simulating query time and caching
import asyncio # For running blocking IO in thread

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event, EventActions, types

# Attempt to import google.cloud.bigquery, but don't fail if not available (for mock usage)
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None # Placeholder if not available


# Mock BigQuery Client for now
class MockBigQueryClient:
    def __init__(self, project: Optional[str] = None):
        self.project = project
        self.logger = logging.getLogger(f"{__name__}.MockBigQueryClient") # Use module-based logger
        self.logger.info(
            f"MockBigQueryClient initialized for project: {project or 'default'}."
        )
        self.query_cache = (
            {}
        )
        self.cache_ttl_seconds = 300

    async def query_async(self, query_string: str, job_config=None) -> "MockQueryJob":
        # Simulate async query execution
        await asyncio.sleep(0.05) # Simulate small network latency and query processing

        self.logger.info(
            f"Executing mock query (first 100 chars): {query_string[:100]}..."
        )

        if query_string in self.query_cache:
            timestamp, cached_result_df = self.query_cache[query_string]
            if (time.time() - timestamp) < self.cache_ttl_seconds:
                self.logger.info("Returning cached result for mock query.")
                return MockQueryJob(
                    cached_result_df.copy(), "COMPLETED_FROM_CACHE"
                )

        data = []
        location_id_mock = "STN_A001"
        pipeline_line_mock = "PL123"
        try:
            if "Location:" in query_string:
                location_id_mock = (
                    query_string.split("Location: ")[1].split(",")[0].strip()
                )
            if "Pipeline:" in query_string:
                pipeline_line_mock = (
                    query_string.split("Pipeline: ")[1].split("\n")[0].strip()
                )
        except IndexError:
            pass

        start_dt_mock = datetime.datetime.strptime(
            query_string.split("Dates: ")[1].split(" to ")[0], "%Y-%m-%d"
        )
        num_days_mock = (
            datetime.datetime.strptime(
                query_string.split(" to ")[1].split("\n")[0], "%Y-%m-%d"
            )
            - start_dt_mock
        ).days + 1
        num_entries_per_day = 2

        current_ts = start_dt_mock
        for _ in range(num_days_mock * num_entries_per_day):
            if "pump_status" in query_string:
                data.append({"timestamp": current_ts, "tag_name": f"{location_id_mock}_{pipeline_line_mock}_PUMP_A_STATUS", "value": np.random.choice([0, 1]), "unit": "state", "quality": "good"})
                data.append({"timestamp": current_ts, "tag_name": f"{location_id_mock}_{pipeline_line_mock}_PUMP_B_STATUS", "value": np.random.choice([0, 1]), "unit": "state", "quality": "good"})
            if "pressure" in query_string:
                data.append({"timestamp": current_ts, "tag_name": f"{location_id_mock}_{pipeline_line_mock}_PUMP_A_UPSTREAM_PRESSURE_PA", "value": 100000 + np.random.rand() * 10000, "unit": "Pa", "quality": "good"})
                data.append({"timestamp": current_ts, "tag_name": f"{location_id_mock}_{pipeline_line_mock}_PUMP_A_DOWNSTREAM_PRESSURE_PA", "value": 500000 + np.random.rand() * 50000, "unit": "Pa", "quality": "good"})
            if "flowrate" in query_string:
                 data.append({"timestamp": current_ts, "tag_name": f"{location_id_mock}_{pipeline_line_mock}_PUMP_A_FLOW_RATE_M3S", "value": 0.02 + np.random.rand() * 0.005, "unit": "m3/s", "quality": "good"})
            current_ts += datetime.timedelta(hours=12)

        if not data:
            self.logger.warning(f"Mock query did not match specific tag groups for: {query_string[:100]}")

        result_df = pd.DataFrame(data)
        self.query_cache[query_string] = (time.time(), result_df.copy())
        return MockQueryJob(result_df.copy(), "COMPLETED")


class MockQueryJob:
    def __init__(self, result_df: pd.DataFrame, state: str):
        self._result_df = result_df
        self._state = state
        self.job_id = "mock_job_" + str(int(time.time() * 1000))
        self.error_result = None # Simulate no error for mock success

    async defresult(self, timeout: Optional[int] = None) -> pd.DataFrame:
        # Simulate async result fetching
        await asyncio.sleep(0.01)
        return self._result_df

    def state(self) -> str: # This can remain synchronous
        return self._state


class BigQueryDataRetrievalAgent(BaseAgent):
    """
    Executes optimized queries against sensor data table in BigQuery.
    (Currently uses a MockBigQueryClient for development without live BQ).
    """

    PROJECT_ID = "your-gcp-project"
    DATASET_ID = "iot_sensor_data"
    GOLDEN_TABLE_ID = "pi_system_golden_data"

    AGGREGATION_MAPPING = {
        "1min": "MINUTE",
        "5min": "MINUTE",
        "15min": "MINUTE",
        "hourly": "HOUR",
        "raw": None,
    }

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        use_mock: bool = True,
    ):
        super().__init__("BigQueryDataRetrievalAgent")
        self.project_id = project_id or self.PROJECT_ID
        self.dataset_id = dataset_id or self.DATASET_ID

        if use_mock:
            self.client = MockBigQueryClient(project=self.project_id)
            self.logger.info("Using MockBigQueryClient.")
        else:
            try:
                from google.cloud import bigquery  # Import only when needed

                self.client = bigquery.Client(project=self.project_id)
                self.logger.info(
                    f"Initialized Google BigQuery Client for project: {self.project_id}"
                )
            except ImportError:
                self.logger.error(
                    "google-cloud-bigquery library not found. Falling back to MockBigQueryClient."
                )
                self.client = MockBigQueryClient(project=self.project_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize real BigQuery client: {e}. Falling back to MockBigQueryClient."
                )
                self.client = MockBigQueryClient(project=self.project_id)

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves data from BigQuery based on the provided parameters.
        Args:
            data (Dict[str, Any]): location_id, pipeline_line, start_date, end_date,
                                 aggregation_interval, tags_to_retrieve (List[str])
        """
        required_keys = ["location_id", "pipeline_line", "start_date", "end_date"]
        if not self._validate_input(data, required_keys=required_keys):
            return self._handle_error("Missing required parameters for data retrieval.")

        location_id = data["location_id"]
        pipeline_line = data["pipeline_line"]
        start_date_str = data["start_date"]
        end_date_str = data["end_date"]
        aggregation_interval = data.get("aggregation_interval", "raw")
        # Default tags if not specified, covering common needs for the analysis pipeline
        tags_to_retrieve = data.get(
            "tags_to_retrieve",
            ["pump_status", "pressure", "flowrate", "dra_injection_rate", "tank_level"],
        )

        if aggregation_interval not in self.AGGREGATION_MAPPING:
            return self._handle_error(
                f"Invalid aggregation interval: {aggregation_interval}."
            )

        try:
            start_datetime = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_datetime = (
                datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
                + datetime.timedelta(days=1)
                - datetime.timedelta(microseconds=1)
            )
        except ValueError:
            return self._handle_error("Invalid date format. Please use YYYY-MM-DD.")

        query = self._build_master_query(
            location_id,
            pipeline_line,
            start_datetime,
            end_datetime,
            tags_to_retrieve,
            aggregation_interval,
        )

        try:
            self.logger.info(
                f"Executing query for location '{location_id}', pipeline '{pipeline_line}'."
            )
            query_job = self.client.query(query)
            results_df = query_job.result()

            if query_job.error_result:
                error_info = query_job.error_result
                return self._handle_error(
                    f"BigQuery query failed: {error_info.get('message', 'Unknown BQ error')}"
                )

            self.logger.info(
                f"Successfully retrieved {len(results_df)} rows from BigQuery."
            )

            return {
                "status": "success",
                "sensor_data": results_df,
                "metadata": {
                    "query_executed": query,
                    "aggregation_used": aggregation_interval,
                },
            }
        except Exception as e:
            # For real BQ: from google.cloud.exceptions import GoogleCloudError
            # if isinstance(e, GoogleCloudError):
            #    return self._handle_error("BigQuery API error occurred.", exception=e)
            return self._handle_error(
                "An unexpected error occurred during BigQuery data retrieval.",
                exception=e,
            )

    def _build_master_query(
        self,
        location_id: str,
        pipeline_line: str,
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
        tag_groups_to_retrieve: List[str],
        aggregation_interval: str,
    ) -> str:
        """
        Constructs a BigQuery SQL query. Simplified for mock interaction.
        For a real system, this would generate complex SQL tailored to the schema.
        `tag_groups_to_retrieve` helps the mock client generate relevant data.
        """
        table_fqn = f"`{self.project_id}.{self.dataset_id}.{self.GOLDEN_TABLE_ID}`"

        # This query string is primarily for the MOCK client to interpret.
        # A real query would be structured SQL.
        simulated_content_marker = (
            ", ".join(tag_groups_to_retrieve)
            if tag_groups_to_retrieve
            else "generic_data"
        )

        # Actual SQL parts (conceptual for real BQ)
        select_clause = (
            "SELECT timestamp, tag_name, value, unit, quality"  # Base selection
        )
        time_trunc_sql = self.AGGREGATION_MAPPING.get(aggregation_interval)
        group_by_clause = ""
        if time_trunc_sql:
            # Example of real aggregation, adjust based on actual needs and value types
            select_clause = (
                f"SELECT TIMESTAMP_TRUNC(timestamp, {time_trunc_sql}) as truncated_timestamp, "
                f"tag_name, AVG(SAFE_CAST(value AS FLOAT64)) as avg_value, "
                f"ANY_VALUE(unit) as unit, STRING_AGG(DISTINCT SAFE_CAST(quality AS STRING), ';') as qualities"
            )
            group_by_clause = "GROUP BY truncated_timestamp, tag_name ORDER BY truncated_timestamp, tag_name"
        else:
            select_clause += " ORDER BY timestamp, tag_name"  # Order raw data

        # Where clause for actual BigQuery
        # Ensure proper SQL formatting and parameterization if using real BQ client libraries with query parameters
        actual_where_clause = (
            f"WHERE location_id = '{location_id}'\n"
            f"  AND pipeline_id = '{pipeline_line}' \n"
            f"  AND timestamp >= TIMESTAMP('{start_dt.isoformat()}')\n"
            f"  AND timestamp <= TIMESTAMP('{end_dt.isoformat()}')"
        )
        # Tag filtering would be added here, e.g. AND tag_name IN ('tag1', 'tag2', ...)
        # or based on mapping `tag_groups_to_retrieve` to specific tag patterns or names.

        # For the mock, the query string is more of a descriptor to guide mock data generation.
        query = f"""
        MOCK QUERY for: {simulated_content_marker}
        Location: {location_id}, Pipeline: {pipeline_line}
        Dates: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}
        Aggregation: {aggregation_interval}
        FROM {table_fqn}
        (Actual SQL conceptualized as: {select_clause} {actual_where_clause} {group_by_clause})
        """

        self.logger.debug(f"Constructed query (mock-oriented): {query.strip()}")
        return query.strip()


if __name__ == "__main__":
    import logging  # Ensure logging is configured for standalone test

    logging.basicConfig(level=logging.DEBUG)  # See debug logs from agent

    retrieval_agent = BigQueryDataRetrievalAgent(use_mock=True)

    test_params_success = {
        "location_id": "STN_A001",
        "pipeline_line": "PL123",
        "start_date": "2023-01-01",
        "end_date": "2023-01-01",  # Short range for mock
        "aggregation_interval": "raw",
        "tags_to_retrieve": ["pump_status", "pressure", "flowrate"],
    }

    print(f"\n--- Test Case: Successful Retrieval (Mocked) ---")
    result = retrieval_agent.execute(test_params_success)
    if result.get("status") == "success":
        print("Data retrieval successful.")
        sensor_df = result.get("sensor_data")
        print(
            f"Retrieved {len(sensor_df) if sensor_df is not None else 'None'} data points."
        )
        if sensor_df is not None and not sensor_df.empty:
            print("Sample data (first 5 rows):\n", sensor_df.head())
            # Verify some expected tags based on mock generation logic
            assert any(
                f"STN_A001_PL123_PUMP_A_STATUS" in tag
                for tag in sensor_df["tag_name"].unique()
            ), "Missing PUMP_A_STATUS"
            assert any(
                f"STN_A001_PL123_PUMP_A_FLOW_RATE_M3S" in tag
                for tag in sensor_df["tag_name"].unique()
            ), "Missing PUMP_A_FLOW_RATE_M3S"

    else:
        print(f"Data retrieval failed: {result.get('error_message')}")

    print(f"\n--- Test Case: Cached Retrieval (Mocked) ---")
    # Should use cache if query is identical and within TTL
    time.sleep(0.1)  # Ensure timestamp is different if test runs too fast
    result_cached = retrieval_agent.execute(test_params_success)
    if result_cached.get("status") == "success":
        print(
            "Cached data retrieval successful (check logs for 'Returning cached result')."
        )
        # Add specific check if MockQueryJob indicates source from cache, if implemented
    else:
        print(f"Cached data retrieval failed: {result_cached.get('error_message')}")

    print("\nBigQueryDataRetrievalAgent tests completed.")
