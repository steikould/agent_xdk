"""
User Interface Agent for DRA Power Analysis
Location: dra_power_analysis/sub_agents/user_interface/agent.py
"""
import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from pydantic import BaseModel, Field

from google.adk.agents import BaseAgent
from google.adk.tools import FunctionTool

# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models
class UserInputModel(BaseModel):
    """Model for user input parameters."""
    location_id: str = Field(..., description="Location identifier (e.g., STN_A001)")
    pipeline_line_number: str = Field(..., description="Pipeline line number (e.g., PL123)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class UserInterfaceInput(BaseModel):
    """Input model for UI operations."""
    operation: str = Field(..., description="Operation type: collect_input, display_results")
    data: Optional[Dict[str, Any]] = Field(default={}, description="Operation data")
    user_input: Optional[UserInputModel] = Field(default=None, description="User input for validation")


class UserInterfaceOutput(BaseModel):
    """Output model for UI operations."""
    success: bool
    validated_input: Optional[Dict[str, str]] = None
    formatted_output: Optional[str] = None
    validation_errors: Optional[List[str]] = None
    error_message: Optional[str] = None


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

    def __init__(self, name: str = "UserInterfaceAgent",
                 description: str = "Handles user interactions",
                 role: str = "input", **kwargs):
        """Initialize the User Interface Agent."""
        super().__init__(name=name, description=description, **kwargs)
        
        if role not in ["input", "output"]:
            raise ValueError("Role must be 'input' or 'output'")
        
        self.role = role
        logger.info(f"UserInterfaceAgent '{self.name}' initialized with role: {self.role}.")
        
        # Set up tools
        self._setup_tools()

    def _setup_tools(self):
        """Set up agent tools."""
        
        async def validate_user_input(input_data: UserInterfaceInput) -> UserInterfaceOutput:
            """
            Validate user input parameters.
            
            Args:
                input_data: User input to validate
                
            Returns:
                Validation results
            """
            try:
                if not input_data.user_input:
                    return UserInterfaceOutput(
                        success=False,
                        error_message="No user input provided"
                    )
                
                user_input = input_data.user_input.model_dump()
                
                # Validate all inputs
                is_valid, validation_errors = self._validate_all_inputs(user_input)
                
                if is_valid:
                    return UserInterfaceOutput(
                        success=True,
                        validated_input=user_input
                    )
                else:
                    return UserInterfaceOutput(
                        success=False,
                        validation_errors=validation_errors
                    )
                    
            except Exception as e:
                logger.error(f"Input validation failed: {e}", exc_info=True)
                return UserInterfaceOutput(
                    success=False,
                    error_message=f"Validation error: {str(e)}"
                )
        
        async def format_analysis_results(input_data: UserInterfaceInput) -> UserInterfaceOutput:
            """
            Format analysis results for display.
            
            Args:
                input_data: Analysis results to format
                
            Returns:
                Formatted output
            """
            try:
                results_data = input_data.data
                if not results_data:
                    return UserInterfaceOutput(
                        success=False,
                        error_message="No results data provided"
                    )
                
                # Format the results
                formatted_output = self._format_results(results_data)
                
                return UserInterfaceOutput(
                    success=True,
                    formatted_output=formatted_output
                )
                
            except Exception as e:
                logger.error(f"Results formatting failed: {e}", exc_info=True)
                return UserInterfaceOutput(
                    success=False,
                    error_message=f"Formatting error: {str(e)}"
                )
        
        async def create_summary_report(input_data: UserInterfaceInput) -> UserInterfaceOutput:
            """
            Create a summary report from analysis results.
            
            Args:
                input_data: Complete analysis results
                
            Returns:
                Summary report
            """
            try:
                results = input_data.data
                
                # Create executive summary
                summary = self._create_executive_summary(results)
                
                return UserInterfaceOutput(
                    success=True,
                    formatted_output=summary
                )
                
            except Exception as e:
                return UserInterfaceOutput(
                    success=False,
                    error_message=f"Summary creation error: {str(e)}"
                )
        
        # Register tools based on role
        if self.role == "input":
            self.tools = [FunctionTool(func=validate_user_input)]
        else:  # output role
            self.tools = [
                FunctionTool(func=format_analysis_results),
                FunctionTool(func=create_summary_report)
            ]

    def _validate_all_inputs(self, user_input: Dict[str, str]) -> Tuple[bool, List[str]]:
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
        if not self._validate_pipeline_line_number(user_input.get("pipeline_line_number")):
            errors.append(
                f"Invalid Pipeline Line Number '{user_input.get('pipeline_line_number')}'. "
                f"Must start with '{self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX}' and be followed by digits."
            )

        # Date Range
        start_date, end_date = None, None
        try:
            start_date_str = user_input.get("start_date")
            end_date_str = user_input.get("end_date")

            if not start_date_str or not end_date_str:
                errors.append("Start date and end date cannot be empty.")
            else:
                start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

                date_range_errors = self._validate_date_range(start_date, end_date)
                if date_range_errors:
                    errors.extend(date_range_errors)

        except ValueError:
            errors.append("Invalid date format. Please use YYYY-MM-DD.")

        return not errors, errors

    def _validate_location(self, location_id: Optional[str]) -> bool:
        """Validates if the location ID exists in the system."""
        if not location_id:
            return False
        is_valid = location_id in self.VALID_LOCATIONS
        if not is_valid:
            logger.warning(f"Validation failed for Location ID: '{location_id}'.")
        return is_valid

    def _validate_pipeline_line_number(self, line_number: Optional[str]) -> bool:
        """Validates the pipeline line number format."""
        if not line_number:
            return False
        is_valid = (
            line_number.startswith(self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX)
            and len(line_number) > len(self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX)
            and line_number[len(self.PIPELINE_LINE_NUMBER_FORMAT_PREFIX):].isdigit()
        )
        if not is_valid:
            logger.warning(f"Validation failed for Pipeline Line Number: '{line_number}'.")
        return is_valid

    def _validate_date_range(self, start_date: Optional[datetime.date], 
                           end_date: Optional[datetime.date]) -> List[str]:
        """Validates the date range."""
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
            errors.append(f"Start date {start_date} cannot be after end date {end_date}.")

        if not errors:
            if (end_date - start_date).days > self.MAX_DATE_RANGE_DAYS:
                errors.append(f"Date range exceeds maximum of {self.MAX_DATE_RANGE_DAYS} days.")

            oldest_allowed_date = today - datetime.timedelta(days=self.MAX_PAST_DAYS_DATA_RETENTION)
            if start_date < oldest_allowed_date:
                errors.append(
                    f"Start date {start_date} is beyond the data retention period of "
                    f"{self.MAX_PAST_DAYS_DATA_RETENTION} days (oldest allowed: {oldest_allowed_date})."
                )

        if errors:
            for error in errors:
                logger.warning(f"Date range validation error: {error}")
        return errors

    def _format_results(self, results_data: Dict[str, Any]) -> str:
        """Format analysis results for display."""
        output = "\n--- Analysis Results ---\n"
        
        if results_data.get("status") == "success":
            # Executive Summary
            summary = results_data.get("executive_summary", "No summary available.")
            output += f"\nExecutive Summary:\n{summary}\n"

            # User Inputs
            user_inputs = results_data.get("user_inputs")
            if user_inputs:
                output += "\nAnalysis Parameters:\n"
                output += f"  Location: {user_inputs.get('location_id')}\n"
                output += f"  Pipeline Line: {user_inputs.get('pipeline_line_number')}\n"
                output += f"  Date Range: {user_inputs.get('start_date')} to {user_inputs.get('end_date')}\n"

            # Data Quality Assessment
            quality_assessment = results_data.get("data_quality_assessment")
            if quality_assessment:
                output += "\nData Quality Assessment:\n"
                output += f"  Summary: {quality_assessment.get('summary', 'N/A')}\n"
                issues_count = quality_assessment.get("detailed_issues_count", 0)
                if issues_count > 0:
                    output += f"  Detailed Issues Count: {issues_count}\n"

            # Power Consumption Summary
            power_summary = results_data.get("power_consumption_summary")
            if power_summary:
                output += "\nPower Consumption Summary:\n"
                for key, value in power_summary.items():
                    formatted_key = key.replace("_", " ").title()
                    output += f"  {formatted_key}: {value}\n"

            # Optimization Roadmap
            optimization = results_data.get("optimization_roadmap")
            if optimization and optimization.get("opportunities"):
                output += "\nOptimization Roadmap:\n"
                for item in optimization.get("opportunities", []):
                    output += f"  - Priority {item.get('priority', 'N/A')}: "
                    output += f"{item.get('type', 'N/A')} for {item.get('pump_name', 'N/A')}\n"
                    output += f"    Description: {item.get('description')}\n"
                    output += f"    Recommendation: {item.get('recommendation')}\n"
                    output += f"    Estimated Savings: {item.get('estimated_savings_potential', 'N/A')}\n"

        elif results_data.get("status") == "error":
            output += f"\nAnalysis process failed: {results_data.get('error_message', 'Unknown error')}\n"
        else:
            output += "\nNo results to display or unknown result status.\n"
        
        output += "\n--- End of Results ---"
        return output

    def _create_executive_summary(self, results: Dict[str, Any]) -> str:
        """Create an executive summary from results."""
        summary = "=== EXECUTIVE SUMMARY ===\n\n"
        
        # Key findings
        summary += "KEY FINDINGS:\n"
        exec_points = results.get("executive_summary_points", [])
        if exec_points:
            for point in exec_points:
                summary += f"• {point}\n"
        else:
            summary += "• Analysis completed successfully\n"
        
        # Metrics
        power_summary = results.get("power_summary_metrics", {})
        if power_summary:
            summary += f"\nKEY METRICS:\n"
            summary += f"• Total Energy: {power_summary.get('total_electrical_energy_kwh_approx', 'N/A')} kWh\n"
            summary += f"• Average Power: {power_summary.get('average_electrical_power_kw', 'N/A')} kW\n"
            summary += f"• Peak Power: {power_summary.get('peak_electrical_power_kw', 'N/A')} kW\n"
        
        # Recommendations
        opportunities = results.get("optimization_opportunities", [])
        if opportunities:
            summary += f"\nTOP RECOMMENDATIONS:\n"
            for opp in opportunities[:3]:  # Top 3
                summary += f"• {opp.get('recommendation', 'N/A')}\n"
        
        return summary


# Create singleton instances for backward compatibility
ui_agent_input = UserInterfaceAgent(
    name="UserInterfaceInputAgent",
    description="Handles user input collection and validation",
    role="input"
)

ui_agent_output = UserInterfaceAgent(
    name="UserInterfaceOutputAgent",
    description="Handles presentation of final results to the user",
    role="output"
)