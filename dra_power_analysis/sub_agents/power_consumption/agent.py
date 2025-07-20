"""
Power Consumption Calculation Agent for DRA Power Analysis
Location: dra_power_analysis/sub_agents/power_consumption/agent.py
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel, Field

from google.adk.agents import BaseAgent
from google.adk.tools import FunctionTool

# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models
class PowerCalculationInput(BaseModel):
    """Input model for power calculation."""
    sensor_data: List[Dict[str, Any]] = Field(..., description="Validated sensor data")
    pump_specifications: Optional[Dict[str, Dict[str, float]]] = Field(
        default={},
        description="Pump-specific efficiency parameters"
    )


class PowerCalculationOutput(BaseModel):
    """Output model for power calculation results."""
    success: bool
    power_data: Optional[List[Dict[str, Any]]] = None
    summary_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class PowerConsumptionCalculationAgent(BaseAgent):
    """
    Calculates power consumption metrics from sensor data using engineering formulas
    and lookup tables for efficiencies.
    """

    # Class constants
    DEFAULT_MOTOR_EFFICIENCY = 0.90
    DEFAULT_PUMP_EFFICIENCY = 0.75
    FLUID_DENSITY_KG_M3 = 900
    GRAVITY_ACCEL_M_S2 = 9.81

    PUMP_SPECIFIC_DATA = {
        "PUMP_A": {
            "motor_efficiency": 0.92,
            "pump_hydraulic_efficiency": 0.78,
            "vsd_efficiency": 0.95,
        },
        "PUMP_B": {
            "motor_efficiency": 0.88,
            "pump_hydraulic_efficiency": 0.70,
            "vsd_efficiency": 0.90,
        },
    }

    def __init__(self, name: str = "PowerCalculationAgent",
                 description: str = "Calculates power consumption metrics", **kwargs):
        """Initialize the Power Consumption Calculation Agent."""
        super().__init__(name=name, description=description, **kwargs)
        logger.info(f"PowerConsumptionCalculationAgent '{self.name}' initialized.")
        
        # Set up tools
        self._setup_tools()

    def _setup_tools(self):
        """Set up agent tools."""
        
        async def calculate_power_consumption(input_data: PowerCalculationInput) -> PowerCalculationOutput:
            """
            Calculate power consumption from sensor data.
            
            Args:
                input_data: Sensor data and pump specifications
                
            Returns:
                Power calculation results
            """
            try:
                # Convert input to DataFrame
                sensor_df = pd.DataFrame(input_data.sensor_data)
                
                if sensor_df.empty:
                    return PowerCalculationOutput(
                        success=True,
                        power_data=[],
                        summary_metrics={
                            "status": "no_data",
                            "message": "No sensor data provided for power calculation."
                        }
                    )
                
                # Validate required columns
                if not all(col in sensor_df.columns for col in ["timestamp", "tag_name", "value"]):
                    return PowerCalculationOutput(
                        success=False,
                        error_message="Missing required columns: timestamp, tag_name, value"
                    )
                
                # Process data
                try:
                    if not pd.api.types.is_datetime64_any_dtype(sensor_df["timestamp"]):
                        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
                    sensor_df["value"] = pd.to_numeric(sensor_df["value"], errors="coerce")
                    sensor_df.dropna(subset=["value"], inplace=True)
                except Exception as e:
                    return PowerCalculationOutput(
                        success=False,
                        error_message=f"Error processing data columns: {e}"
                    )
                
                # Merge pump specifications if provided
                pump_specs = {**self.PUMP_SPECIFIC_DATA}
                if input_data.pump_specifications:
                    pump_specs.update(input_data.pump_specifications)
                
                # Perform power calculations
                power_df, summary = self._perform_power_calculations(sensor_df.copy(), pump_specs)
                
                # Convert results to serializable format
                power_data = power_df.to_dict('records') if not power_df.empty else []
                
                return PowerCalculationOutput(
                    success=True,
                    power_data=power_data,
                    summary_metrics=summary
                )
                
            except Exception as e:
                logger.error(f"Power calculation failed: {e}", exc_info=True)
                return PowerCalculationOutput(
                    success=False,
                    error_message=f"Power calculation error: {str(e)}"
                )
        
        async def calculate_pump_efficiency(input_data: PowerCalculationInput) -> PowerCalculationOutput:
            """
            Calculate pump efficiency from power data.
            
            Args:
                input_data: Power consumption data
                
            Returns:
                Efficiency calculation results
            """
            try:
                # This would calculate pump efficiency based on hydraulic vs electrical power
                sensor_df = pd.DataFrame(input_data.sensor_data)
                
                efficiency_results = {}
                for pump_name in self.PUMP_SPECIFIC_DATA.keys():
                    pump_data = sensor_df[sensor_df["tag_name"].str.contains(pump_name)]
                    if not pump_data.empty:
                        # Simplified efficiency calculation
                        efficiency_results[pump_name] = {
                            "average_efficiency": 0.75,  # Placeholder
                            "efficiency_trend": "stable",
                            "recommendation": "Monitor for changes"
                        }
                
                return PowerCalculationOutput(
                    success=True,
                    summary_metrics={"efficiency_analysis": efficiency_results}
                )
                
            except Exception as e:
                return PowerCalculationOutput(
                    success=False,
                    error_message=f"Efficiency calculation error: {str(e)}"
                )
        
        # Register tools
        self.tools = [
            FunctionTool(func=calculate_power_consumption),
            FunctionTool(func=calculate_pump_efficiency)
        ]

    def _perform_power_calculations(self, sensor_df: pd.DataFrame, 
                                   pump_specs: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Private helper to perform power calculations.
        
        Args:
            sensor_df: Sensor data DataFrame
            pump_specs: Pump specifications
            
        Returns:
            Tuple of (power data DataFrame, summary metrics)
        """
        identified_pumps_for_processing = []
        all_tags = sensor_df["tag_name"].unique()
        
        # Identify pumps to process
        for configured_pump_name in pump_specs.keys():
            if any(configured_pump_name in tag and "_STATUS" in tag for tag in all_tags):
                identified_pumps_for_processing.append(configured_pump_name)

        if not identified_pumps_for_processing:
            logger.warning("No pump status tags found to identify any pumps.")
            return pd.DataFrame(), {"status": "no_pumps_identified"}

        logger.info(f"Identified pumps for power calculation: {identified_pumps_for_processing}")

        all_power_data_frames = []

        try:
            # Pivot sensor data
            sensor_df_no_duplicates = sensor_df.drop_duplicates(
                subset=["timestamp", "tag_name"], keep="last"
            )
            pivoted_df = sensor_df_no_duplicates.pivot(
                index="timestamp", columns="tag_name", values="value"
            )
        except Exception as e:
            logger.error(f"Failed to pivot sensor data: {e}", exc_info=True)
            raise ValueError(f"Failed to pivot sensor data: {e}")

        for pump_name in identified_pumps_for_processing:
            logger.debug(f"Calculating power for pump: {pump_name}")

            # Find relevant tags
            status_tag = next((tag for tag in pivoted_df.columns 
                             if pump_name in tag and "_STATUS" in tag), None)
            up_pressure_tag = next((tag for tag in pivoted_df.columns 
                                  if pump_name in tag and "_UPSTREAM_PRESSURE_PA" in tag), None)
            down_pressure_tag = next((tag for tag in pivoted_df.columns 
                                    if pump_name in tag and "_DOWNSTREAM_PRESSURE_PA" in tag), None)
            flow_rate_tag = next((tag for tag in pivoted_df.columns 
                                if pump_name in tag and "_FLOW_RATE_M3S" in tag), None)

            required_tags = [status_tag, up_pressure_tag, down_pressure_tag, flow_rate_tag]
            if not all(required_tags):
                logger.warning(f"Pump {pump_name}: Missing required tags. Skipping.")
                continue

            # Extract pump data
            pump_df = pivoted_df[[t for t in required_tags if t is not None]].copy()
            pump_df.dropna(inplace=True)

            if pump_df.empty:
                logger.warning(f"Pump {pump_name}: No complete data rows. Skipping.")
                continue

            # Get pump specifications
            specs = pump_specs.get(pump_name, {})
            motor_eff = specs.get("motor_efficiency", self.DEFAULT_MOTOR_EFFICIENCY)
            pump_hyd_eff = specs.get("pump_hydraulic_efficiency", self.DEFAULT_PUMP_EFFICIENCY)
            vsd_eff = specs.get("vsd_efficiency", 1.0)

            # Calculate power metrics
            pump_df["delta_pressure_pa"] = pump_df[down_pressure_tag] - pump_df[up_pressure_tag]
            pump_df["hydraulic_power_w"] = pump_df[flow_rate_tag] * pump_df["delta_pressure_pa"]
            pump_df.loc[pump_df["hydraulic_power_w"] < 0, "hydraulic_power_w"] = 0

            pump_df["shaft_power_w"] = np.where(
                pump_hyd_eff > 0, pump_df["hydraulic_power_w"] / pump_hyd_eff, np.nan
            )
            total_drive_eff = motor_eff * vsd_eff
            pump_df["electrical_power_w"] = np.where(
                total_drive_eff > 0, pump_df["shaft_power_w"] / total_drive_eff, np.nan
            )

            # Set power to 0 when pump is off
            pump_df.loc[pump_df[status_tag] == 0,
                       ["hydraulic_power_w", "shaft_power_w", "electrical_power_w"]] = 0

            # Convert to kW
            for col_w in ["hydraulic_power_w", "shaft_power_w", "electrical_power_w"]:
                pump_df[col_w.replace("_w", "_kw")] = pump_df[col_w] / 1000

            pump_df["pump_name"] = pump_name
            pump_df = pump_df.reset_index()

            # Select output columns
            output_cols = [
                "timestamp", "pump_name", "delta_pressure_pa",
                "hydraulic_power_kw", "shaft_power_kw", "electrical_power_kw",
                status_tag, flow_rate_tag
            ]
            output_cols = [col for col in output_cols if col in pump_df.columns]
            all_power_data_frames.append(pump_df[output_cols])

        if not all_power_data_frames:
            return pd.DataFrame(), {"status": "no_data_processed"}

        # Combine results
        final_power_df = pd.concat(all_power_data_frames, ignore_index=True)
        final_power_df.sort_values(by=["timestamp", "pump_name"], inplace=True)

        # Calculate summary metrics
        total_energy_kwh = 0
        if "electrical_power_kw" in final_power_df.columns and not final_power_df.empty:
            final_power_df["time_delta_h"] = (
                final_power_df.groupby("pump_name")["timestamp"]
                .diff()
                .dt.total_seconds()
                .fillna(0) / 3600.0
            )
            final_power_df["energy_kwh"] = (
                final_power_df["electrical_power_kw"] * final_power_df["time_delta_h"]
            )
            total_energy_kwh = final_power_df["energy_kwh"].sum()

        summary = {
            "total_pumps_calculated_for": len(all_power_data_frames),
            "total_electrical_energy_kwh_approx": round(total_energy_kwh, 2),
            "average_electrical_power_kw": (
                round(final_power_df["electrical_power_kw"].mean(), 2)
                if not final_power_df.empty else 0
            ),
            "peak_electrical_power_kw": (
                round(final_power_df["electrical_power_kw"].max(), 2)
                if not final_power_df.empty else 0
            ),
        }
        
        return final_power_df, summary


# Create singleton instance for backward compatibility
power_calc_agent = PowerConsumptionCalculationAgent(
    name="PowerConsumptionCalculationAgent",
    description="Converts sensor data to power consumption metrics."
)