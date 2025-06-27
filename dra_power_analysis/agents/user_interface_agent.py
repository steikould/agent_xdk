import datetime
from typing import Dict, Any, Optional, Tuple, AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event, EventActions, types


class UserInterfaceAgent(BaseAgent):
    """
    Handles all user interactions, input gathering, validation, and results presentation.
    Can operate in 'input' or 'output' role.
    """

    # Configuration for validation
    VALID_LOCATIONS = ["STN_A001", "STN_B002", "STN_C003"]
    PIPELINE_LINE_NUMBER_FORMAT_PREFIX = "PL"
    MAX_DATE_RANGE_DAYS = 365
    MAX_PAST_DAYS_DATA_RETENTION = 5 * 365

    def __init__(self, name: str, description: str, role: str = "input", **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        if role not in ["input", "output"]:
            raise ValueError("Role must be 'input' or 'output'")
        self.role = role
        self.logger.info(f"UserInterfaceAgent '{self.name}' initialized with role: {self.role}.")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        self.logger.info(f"UserInterfaceAgent '{self.name}' (role: {self.role}) starting execution.")

        if self.role == "input":
            try:
                user_input = await asyncio.to_thread(self._get_user_input_interactive)

                if not user_input:
                    self.logger.warning("User input process aborted or cancelled by user.")
                    # Escalate to stop the sequence if input is aborted
                    yield Event(
                        author=self.name,
                        content=types.Content(parts=[types.Part.from_text("User input aborted. Halting workflow.")]),
                        actions=EventActions(escalate=True)
                    )
                    return

                is_valid, validation_errors = self._validate_all_inputs(user_input)

                if not is_valid:
                    error_message = "Input validation failed. Errors:\n" + "\n".join(validation_errors)
                    self.display_error(error_message) # Display error to console
                    self.logger.error(error_message)
                     # Escalate to stop the sequence if input is invalid
                    yield Event(
                        author=self.name,
                        content=types.Content(parts=[types.Part.from_text(f"Input validation failed: {error_message}. Halting workflow.")]),
                        actions=EventActions(escalate=True)
                    )
                    return

                self.logger.info(f"User input successfully validated: {user_input}")
                ctx.session.state["user_params"] = user_input
                ctx.session.state["status_user_input"] = "success"
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(f"User input collected and validated: {user_input}")])
                )

            except Exception as e:
                self.logger.error(f"Error during user input phase: {e}", exc_info=True)
                ctx.session.state["status_user_input"] = "error"
                ctx.session.state["error_message_user_input"] = str(e)
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(f"Error in user input agent: {e}")]),
                    actions=EventActions(escalate=True) # Escalate on unexpected error
                )

        elif self.role == "output":
            final_report = ctx.session.state.get("final_consolidated_report")
            if final_report:
                await asyncio.to_thread(self.display_results, final_report)
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text("Final results displayed to user.")])
                )
            else:
                error_msg = "Final report not found in session state for display."
                self.logger.error(error_msg)
                await asyncio.to_thread(self.display_error, error_msg)
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(error_msg)])
                )
        else:
            # Should not happen due to __init__ validation
            self.logger.error(f"Unknown role for UserInterfaceAgent: {self.role}")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(f"Internal error: Unknown role {self.role}")])
            )
        self.logger.info(f"UserInterfaceAgent '{self.name}' (role: {self.role}) finished execution.")

    def _get_user_input_interactive(self) -> Optional[Dict[str, str]]:
        # This method involves blocking I/O (input()), so it's run in a separate thread by asyncio.to_thread
        """
        Interactively prompts the user for necessary inputs.
        Returns a dictionary of inputs or None if the user wishes to abort.
        """
        self.logger.info("Prompting user for inputs.")
        try:
            location_id = input(
                f"Enter Location ID (e.g., {', '.join(self.VALID_LOCATIONS)}): "
            ).strip()
            if not location_id:
                return None  # Allow abort

            pipeline_line_number = input(
                f"Enter Pipeline Line Number (e.g., {self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX}XXX): "
            ).strip()
            if not pipeline_line_number:
                return None  # Allow abort

            start_date_str = input("Enter Start Date (YYYY-MM-DD): ").strip()
            if not start_date_str:
                return None  # Allow abort

            end_date_str = input("Enter End Date (YYYY-MM-DD): ").strip()
            if not end_date_str:
                return None  # Allow abort

            return {
                "location_id": location_id,
                "pipeline_line_number": pipeline_line_number,
                "start_date": start_date_str,
                "end_date": end_date_str,
            }
        except (
            EOFError
        ):  # Handle cases where input stream is closed (e.g., in non-interactive environments)
            self.logger.warning(
                "EOFError encountered during input. Assuming no input provided."
            )
            return None
        except KeyboardInterrupt:
            self.logger.info("User interrupted input process.")
            return None

    def _validate_all_inputs(
        self, user_input: Dict[str, str]
    ) -> Tuple[bool, list[str]]:
        """
        Validates all collected user inputs using specific validation methods.
        """
        errors = []

        # Location ID
        if not self._validate_location(user_input.get("location_id")):
            errors.append(
                f"Invalid Location ID '{user_input.get('location_id')}'. Valid locations: {self.VALID_LOCATIONS}"
            )

        # Pipeline Line Number
        if not self._validate_pipeline_line_number(
            user_input.get("pipeline_line_number")
        ):
            errors.append(
                f"Invalid Pipeline Line Number '{user_input.get('pipeline_line_number')}'. Must start with '{self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX}' and be followed by digits."
            )

        # Date Range (requires parsing first)
        start_date, end_date = None, None
        try:
            start_date_str = user_input.get("start_date")
            end_date_str = user_input.get("end_date")

            if not start_date_str or not end_date_str:
                errors.append("Start date and end date cannot be empty.")
            else:
                start_date = datetime.datetime.strptime(
                    start_date_str, "%Y-%m-%d"
                ).date()
                end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

                date_range_errors = self._validate_date_range(start_date, end_date)
                if date_range_errors:
                    errors.extend(date_range_errors)

        except ValueError:
            errors.append("Invalid date format. Please use YYYY-MM-DD.")

        return not errors, errors

    def _validate_location(self, location_id: Optional[str]) -> bool:
        """Validates if the location ID exists in the system (mocked)."""
        if not location_id:
            return False
        is_valid = location_id in self.VALID_LOCATIONS
        if not is_valid:
            self.logger.warning(f"Validation failed for Location ID: '{location_id}'.")
        return is_valid

    def _validate_pipeline_line_number(self, line_number: Optional[str]) -> bool:
        """Validates the pipeline line number format (mocked)."""
        if not line_number:
            return False
        # Example: Must start with "PL" and be followed by 3 digits, e.g., PL001
        is_valid = (
            line_number.startswith(self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX)
            and len(line_number) > len(self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX)
            and line_number[len(self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX) :].isdigit()
        )
        if not is_valid:
            self.logger.warning(
                f"Validation failed for Pipeline Line Number: '{line_number}'."
            )
        return is_valid

    def _validate_date_range(
        self, start_date: Optional[datetime.date], end_date: Optional[datetime.date]
    ) -> list[str]:
        """
        Validates the date range:
        - Not future dates.
        - Start date is not after end date.
        - Range does not exceed maximum allowed days.
        - Start date is not beyond data retention period.
        """
        errors = []
        if not start_date or not end_date:
            errors.append("Start date or end date is missing for range validation.")
            return errors

        today = datetime.date.today()

        if start_date > today:
            errors.append(f"Start date {start_date} cannot be in the future.")
        if end_date > today:
            errors.append(f"End date {end_date} cannot be in the future.")
        if start_date > end_date:
            errors.append(
                f"Start date {start_date} cannot be after end date {end_date}."
            )

        if not errors:  # Only proceed if basic date logic is fine
            if (end_date - start_date).days > self.MAX_DATE_RANGE_DAYS:
                errors.append(
                    f"Date range exceeds maximum of {self.MAX_DATE_RANGE_DAYS} days."
                )

            oldest_allowed_date = today - datetime.timedelta(
                days=self.MAX_PAST_DAYS_DATA_RETENTION
            )
            if start_date < oldest_allowed_date:
                errors.append(
                    f"Start date {start_date} is beyond the data retention period of {self.MAX_PAST_DAYS_DATA_RETENTION} days (oldest allowed: {oldest_allowed_date})."
                )

        if errors:
            for error in errors:
                self.logger.warning(f"Date range validation error: {error}")
        return errors

    def display_results(self, results_data: Dict[str, Any]):
        """
        Formats and presents the final results to the user.
        This is a placeholder and should be expanded based on the structure of `results_data`.
        """
        self.logger.info("Displaying results to the user.")
        print("\n--- Analysis Results ---")
        if results_data.get("status") == "success":
            summary = results_data.get("executive_summary", "No summary available.")
            print("\nExecutive Summary:")
            print(summary)

            # User Inputs (from orchestrator's final report)
            user_inputs = results_data.get("user_inputs")
            if user_inputs:
                print("\nAnalysis Parameters:")
                print(f"  Location: {user_inputs.get('location_id')}")
                print(f"  Pipeline Line: {user_inputs.get('pipeline_line_number')}")
                print(
                    f"  Date Range: {user_inputs.get('start_date')} to {user_inputs.get('end_date')}"
                )

            # Data Quality Assessment (from orchestrator's final report)
            quality_assessment = results_data.get("data_quality_assessment")
            if quality_assessment:
                print("\nData Quality Assessment:")
                print(
                    f"  Summary: {quality_assessment.get('issues_found_summary', 'N/A')}"
                )
                if quality_assessment.get("detailed_issues_count", 0) > 0:
                    print(
                        f"  Detailed Issues Count: {quality_assessment.get('detailed_issues_count')}"
                    )
                if quality_assessment.get("recommendations"):
                    print("  Recommendations:")
                    for rec in quality_assessment.get("recommendations"):
                        print(f"    - {rec}")

            # Power Consumption Summary (from orchestrator's final report)
            power_summary = results_data.get("power_consumption_summary")
            if power_summary:
                print("\nPower Consumption Summary:")
                for key, value in power_summary.items():
                    # Simple formatting: replace underscores, title case
                    formatted_key = key.replace("_", " ").title()
                    print(f"  {formatted_key}: {value}")

            # Statistical Analysis Highlights (from orchestrator's final report)
            stats_highlights = results_data.get("statistical_analysis_highlights")
            if stats_highlights:
                print("\nStatistical Analysis Highlights:")
                if stats_highlights.get("pumps_analyzed"):
                    print(
                        f"  Pumps Analyzed: {', '.join(stats_highlights.get('pumps_analyzed'))}"
                    )
                if stats_highlights.get("system_avg_power") is not None:
                    print(
                        f"  System Average Power: {stats_highlights.get('system_avg_power'):.2f} kW"
                    )

            # Optimization Roadmap (from orchestrator's final report)
            optimization_roadmap = results_data.get("optimization_roadmap")
            if optimization_roadmap and optimization_roadmap.get("opportunities"):
                print("\nOptimization Roadmap:")
                for item in optimization_roadmap.get("opportunities", []):
                    print(
                        f"  - Priority {item.get('priority', 'N/A')}: {item.get('type', 'N/A')} for {item.get('pump_name', 'N/A')}"
                    )
                    print(f"    Description: {item.get('description')}")
                    print(f"    Recommendation: {item.get('recommendation')}")
                    print(
                        f"    Estimated Savings: {item.get('estimated_savings_potential', 'N/A')}"
                    )
            elif optimization_roadmap:
                print("\nOptimization Roadmap:")
                print(
                    "  No specific optimization opportunities identified at this time."
                )

        elif results_data.get("status") == "error":
            self.display_error(
                f"Analysis process failed or was aborted: {results_data.get('error_message', 'Unknown error')}"
            )
        else:
            print("No results to display or unknown result status.")
        print("--- End of Results ---")

    def display_error(self, message: str):
        """
        Displays an error message to the user.
        """
        self.logger.error(f"Displaying error to user: {message}")
        print(f"\n--- ERROR ---")
        print(message)
        print("--- END ERROR ---")


if __name__ == "__main__":
    ui_agent = UserInterfaceAgent()

    # To test interactively, you would just run:
    # validated_input_package = ui_agent.execute()
    # if validated_input_package.get("status") == "success":
    #     print("\nValidated Input Package:")
    #     print(validated_input_package)
    # else:
    #     print("\nInput process failed or was invalid:")
    #     print(validated_input_package)

    # For non-interactive testing of validation logic:
    print("\n--- Testing Validation Logic ---")

    # Test case 1: Valid input
    valid_test_input = {
        "location_id": "STN_A001",
        "pipeline_line_number": "PL123",
        "start_date": (datetime.date.today() - datetime.timedelta(days=10)).strftime(
            "%Y-%m-%d"
        ),
        "end_date": (datetime.date.today() - datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),
    }
    is_valid, errors = ui_agent._validate_all_inputs(valid_test_input)
    print(f"Test Case 1 (Valid): Valid = {is_valid}, Errors = {errors}")
    assert is_valid

    # Test case 2: Invalid location
    invalid_loc_input = {**valid_test_input, "location_id": "INVALID_LOC"}
    is_valid, errors = ui_agent._validate_all_inputs(invalid_loc_input)
    print(f"Test Case 2 (Invalid Location): Valid = {is_valid}, Errors = {errors}")
    assert not is_valid and any("Invalid Location ID" in e for e in errors)

    # Test display_results (mocked orchestrator output)
    print("\n--- Testing Display Logic (Simulated Orchestrator Output) ---")
    mock_orchestrator_success_results = {
        "status": "success",
        "user_inputs": {
            "location_id": "STN_A001",
            "pipeline_line_number": "PL123",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
        },
        "executive_summary": "Executive Summary:\n- PUMP_X: Efficiency has degraded. Maintenance planning advised.\n- Total approximate electrical energy consumed: 1234.56 kWh.\n",
        "data_quality_assessment": {
            "issues_found_summary": "Found 2 potential data quality issues.",
            "detailed_issues_count": 2,
            "recommendations": ["Review data gaps for SENSOR_Y."],
        },
        "power_consumption_summary": {
            "total_pumps_processed": 1,
            "total_electrical_energy_kwh_approx": 1234.56,
            "average_electrical_power_kw": 51.44,
            "peak_electrical_power_kw": 150.22,
        },
        "statistical_analysis_highlights": {
            "pumps_analyzed": ["PUMP_X"],
            "system_avg_power": 51.44,
        },
        "optimization_roadmap": {
            "opportunities": [
                {
                    "pump_name": "PUMP_X",
                    "type": "Efficiency Degradation",
                    "priority": 2,
                    "description": "Pump PUMP_X shows an efficiency degradation...",
                    "recommendation": "Schedule inspection for PUMP_X...",
                    "estimated_savings_potential": "Medium (2-8%)",
                }
            ]
        },
    }
    ui_agent.display_results(mock_orchestrator_success_results)

    mock_orchestrator_error_results = {
        "status": "error",
        "error_message": "Failed to retrieve data from BigQuery due to authentication failure.",
    }
    ui_agent.display_results(mock_orchestrator_error_results)

    print(
        "\nTo test interactive input, uncomment the `ui_agent.execute()` call in the main block and run the script."
    )
