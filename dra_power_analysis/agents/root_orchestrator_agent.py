from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base_agent import Agent


class PowerConsumptionCalculationAgent(Agent):
    """
    Calculates power consumption metrics from sensor data using engineering formulas
    and lookup tables for efficiencies.
    """

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
        },
    }

    def __init__(self):
        super().__init__("PowerConsumptionCalculationAgent")

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): sensor_data (pd.DataFrame), pump_config (optional),
                                 pipe_characteristics (optional).
        """
        required_keys = ["sensor_data"]
        if not self._validate_input(data, required_keys=required_keys):
            return self._handle_error("Missing 'sensor_data' for power calculation.")

        sensor_df = data.get("sensor_data")
        if not isinstance(sensor_df, pd.DataFrame) or sensor_df.empty:
            return self._handle_error(
                "'sensor_data' must be a non-empty Pandas DataFrame."
            )

        if not all(
            col in sensor_df.columns for col in ["timestamp", "tag_name", "value"]
        ):
            return self._handle_error(
                "sensor_data missing basic columns: timestamp, tag_name, value."
            )

        try:
            if not pd.api.types.is_datetime64_any_dtype(sensor_df["timestamp"]):
                sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
            sensor_df["value"] = pd.to_numeric(sensor_df["value"], errors="coerce")
            sensor_df.dropna(
                subset=["value"], inplace=True
            )  # Critical for calculations
        except Exception as e:
            return self._handle_error(
                "Error processing sensor_data columns for power calc.", exception=e
            )

        identified_pumps_for_processing = []
        all_tags = sensor_df["tag_name"].unique()
        for configured_pump_name in self.PUMP_SPECIFIC_DATA.keys():
            # Check if any tag contains this configured pump name AND "_STATUS"
            if any(
                configured_pump_name in tag and "_STATUS" in tag for tag in all_tags
            ):
                identified_pumps_for_processing.append(configured_pump_name)

        if not identified_pumps_for_processing:
            # Fallback: try to infer any _PUMP_X_STATUS tag if no configured pumps match
            # This part might be too broad if tags are not well-defined, use with caution or refine.
            self.logger.warning(
                "No pump status tags matching configured pumps. Attempting to infer from generic tags like '..._PUMP_X_STATUS'."
            )
            inferred_pump_names = list(
                set(
                    [
                        tag.split("_STATUS")[0]
                        for tag in all_tags
                        if "_PUMP_" in tag and "_STATUS" in tag
                    ]
                )
            )
            # A more specific inference might look for a pattern like `PREFIX_PUMPNAME_SUFFIX`
            # For now, this is a simple inference.
            if inferred_pump_names:
                identified_pumps_for_processing.extend(inferred_pump_names)
                identified_pumps_for_processing = list(
                    set(identified_pumps_for_processing)
                )  # unique

            if not identified_pumps_for_processing:
                return self._handle_error(
                    "No pump status tags found to identify any pumps (configured or inferred)."
                )

        self.logger.info(
            f"Identified pumps for power calculation: {identified_pumps_for_processing}"
        )

        all_power_data_frames = []

        try:
            sensor_df_no_duplicates = sensor_df.drop_duplicates(
                subset=["timestamp", "tag_name"], keep="last"
            )
            pivoted_df = sensor_df_no_duplicates.pivot(
                index="timestamp", columns="tag_name", values="value"
            )
        except Exception as e:
            return self._handle_error(
                "Failed to pivot sensor data for power calc. Check for duplicate timestamp/tag_name pairs.",
                exception=e,
            )

        for (
            pump_name_key
        ) in identified_pumps_for_processing:  # This is the key like 'PUMP_A', 'PUMP_B'
            self.logger.debug(f"Calculating power for pump: {pump_name_key}")

            # Find full tag names based on pump_name_key (e.g., 'PUMP_A')
            # This dynamic matching assumes tags will contain the pump_name_key string.
            status_tag = next(
                (
                    tag
                    for tag in pivoted_df.columns
                    if pump_name_key in tag and "_STATUS" in tag
                ),
                None,
            )
            up_pressure_tag = next(
                (
                    tag
                    for tag in pivoted_df.columns
                    if pump_name_key in tag and "_UPSTREAM_PRESSURE_PA" in tag
                ),
                None,
            )
            down_pressure_tag = next(
                (
                    tag
                    for tag in pivoted_df.columns
                    if pump_name_key in tag and "_DOWNSTREAM_PRESSURE_PA" in tag
                ),
                None,
            )
            flow_rate_tag = next(
                (
                    tag
                    for tag in pivoted_df.columns
                    if pump_name_key in tag and "_FLOW_RATE_M3S" in tag
                ),
                None,
            )

            required_pump_tags_found = [
                status_tag,
                up_pressure_tag,
                down_pressure_tag,
                flow_rate_tag,
            ]
            if not all(required_pump_tags_found):
                self.logger.warning(
                    f"Pump {pump_name_key}: Missing one or more required full tag names (status, pressures, flow). Skipping."
                )
                continue

            current_pump_tags_to_select = [
                t for t in required_pump_tags_found if t is not None
            ]
            pump_df = pivoted_df[current_pump_tags_to_select].copy()
            pump_df.dropna(inplace=True)

            if pump_df.empty:
                self.logger.warning(
                    f"Pump {pump_name_key}: No complete data rows after selecting and NaN drop. Skipping."
                )
                continue

            # Use pump_name_key to get specific data, fallback to defaults if key not in PUMP_SPECIFIC_DATA (for inferred pumps)
            pump_specs = self.PUMP_SPECIFIC_DATA.get(pump_name_key, {})
            motor_eff = pump_specs.get(
                "motor_efficiency", self.DEFAULT_MOTOR_EFFICIENCY
            )
            pump_hyd_eff = pump_specs.get(
                "pump_hydraulic_efficiency", self.DEFAULT_PUMP_EFFICIENCY
            )
            vsd_eff = pump_specs.get(
                "vsd_efficiency", 1.0
            )  # Assume 1.0 (no VSD / perfect VSD) if not specified

            pump_df["delta_pressure_pa"] = (
                pump_df[down_pressure_tag] - pump_df[up_pressure_tag]
            )
            pump_df["hydraulic_power_w"] = (
                pump_df[flow_rate_tag] * pump_df["delta_pressure_pa"]
            )
            pump_df.loc[pump_df["hydraulic_power_w"] < 0, "hydraulic_power_w"] = (
                0  # No negative hydraulic power
            )

            pump_df["shaft_power_w"] = np.where(
                pump_hyd_eff > 0, pump_df["hydraulic_power_w"] / pump_hyd_eff, np.nan
            )
            total_drive_eff = motor_eff * vsd_eff
            pump_df["electrical_power_w"] = np.where(
                total_drive_eff > 0, pump_df["shaft_power_w"] / total_drive_eff, np.nan
            )

            # If status_tag (actual full tag name) value is 0, set powers to 0
            pump_df.loc[
                pump_df[status_tag] == 0,
                ["hydraulic_power_w", "shaft_power_w", "electrical_power_w"],
            ] = 0

            for col_w in ["hydraulic_power_w", "shaft_power_w", "electrical_power_w"]:
                pump_df[col_w.replace("_w", "_kw")] = pump_df[col_w] / 1000

            pump_df["pump_name"] = (
                pump_name_key  # Store the base name (PUMP_A, PUMP_B etc.)
            )
            pump_df = pump_df.reset_index()  # timestamp becomes a column

            output_cols_base = [
                "timestamp",
                "pump_name",
                "delta_pressure_pa",
                "hydraulic_power_kw",
                "shaft_power_kw",
                "electrical_power_kw",
            ]
            # Add the actual dynamic tags used for status and flow to the output for clarity/traceability
            final_output_cols = output_cols_base + [status_tag, flow_rate_tag]
            final_output_cols = [
                col for col in final_output_cols if col in pump_df.columns
            ]  # Ensure all exist
            all_power_data_frames.append(pump_df[final_output_cols])

        if not all_power_data_frames:
            return self._handle_error(
                "No pump data could be processed for power calculation based on identified pumps."
            )

        final_power_df = pd.concat(all_power_data_frames, ignore_index=True)
        final_power_df.sort_values(by=["timestamp", "pump_name"], inplace=True)

        total_energy_kwh = 0
        if "electrical_power_kw" in final_power_df.columns and not final_power_df.empty:
            # Ensure time_delta_h is calculated per pump correctly after concat
            final_power_df["time_delta_h"] = (
                final_power_df.groupby("pump_name")["timestamp"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
                / 3600.0
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
                if not final_power_df.empty
                else 0
            ),
            "peak_electrical_power_kw": (
                round(final_power_df["electrical_power_kw"].max(), 2)
                if not final_power_df.empty
                else 0
            ),
        }

        self.logger.info(f"Power consumption calculation completed. Summary: {summary}")
        return {
            "status": "success",
            "power_consumption_data": final_power_df,
            "summary_metrics": summary,
        }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)  # To see debug logs from agent
    pc_agent = PowerConsumptionCalculationAgent()

    ts = [
        pd.Timestamp("2023-01-01 10:00:00") + pd.Timedelta(minutes=i) for i in range(3)
    ]
    mock_sensor_data_list = []
    # PUMP_A data (matches PUMP_SPECIFIC_DATA)
    # Using more descriptive tag names as the mock BQ agent might produce
    pump_a_id = "PUMP_A"
    for t_idx, t_val in enumerate(ts):
        mock_sensor_data_list.extend(
            [
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_a_id}_STATUS",
                    "value": 1 if t_idx < 2 else 0,
                },
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_a_id}_UPSTREAM_PRESSURE_PA",
                    "value": 100000 + t_idx * 1000,
                },
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_a_id}_DOWNSTREAM_PRESSURE_PA",
                    "value": 500000 + t_idx * 2000,
                },
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_a_id}_FLOW_RATE_M3S",
                    "value": 0.02 + t_idx * 0.001,
                },
            ]
        )
    # PUMP_C data (will use default efficiencies as PUMP_C is not in PUMP_SPECIFIC_DATA by default)
    pump_c_id = "PUMP_C"  # This pump is not in PUMP_SPECIFIC_DATA, so should use defaults if inferred
    for t_idx, t_val in enumerate(ts):
        mock_sensor_data_list.extend(
            [
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_c_id}_STATUS",
                    "value": 1,
                },
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_c_id}_UPSTREAM_PRESSURE_PA",
                    "value": 90000,
                },
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_c_id}_DOWNSTREAM_PRESSURE_PA",
                    "value": 400000,
                },
                {
                    "timestamp": t_val,
                    "tag_name": f"STN01_LINE01_{pump_c_id}_FLOW_RATE_M3S",
                    "value": 0.018,
                },
            ]
        )
    mock_sensor_df = pd.DataFrame(mock_sensor_data_list)

    print(
        "\n--- Test Case: Valid Sensor Data (PUMP_A configured, PUMP_C inferred/default) ---"
    )
    # PUMP_C is not in PUMP_SPECIFIC_DATA, so it tests the inference and default efficiency path.
    # The agent's `identified_pumps_for_processing` logic will try to find it via `_PUMP_` and `_STATUS` pattern.
    result = pc_agent.execute({"sensor_data": mock_sensor_df.copy()})

    if result["status"] == "success":
        power_df = result["power_consumption_data"]
        summary = result["summary_metrics"]
        print("Summary Metrics:", summary)
        # print("Calculated Power Data:\n", power_df)
        assert "PUMP_A" in power_df["pump_name"].unique(), "PUMP_A should be processed"
        # Check if PUMP_C was inferred and processed (it should be due to the fallback logic)
        assert (
            "STN01_LINE01_PUMP_C" in power_df["pump_name"].unique()
        ), "PUMP_C (full inferred name) should be processed"

        pump_a_status_col = "STN01_LINE01_PUMP_A_STATUS"
        assert (
            power_df[
                (power_df["pump_name"] == "PUMP_A") & (power_df[pump_a_status_col] == 0)
            ]["electrical_power_kw"].sum()
            == 0
        )
    else:
        print(f"Power calculation failed: {result['error_message']}")

    print("\nPowerConsumptionCalculationAgent tests completed.")
