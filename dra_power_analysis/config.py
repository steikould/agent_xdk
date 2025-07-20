"""
Configuration settings for DRA Power Analysis System
Location: dra_power_analysis/config.py
"""
import os
from typing import Dict, Any


class Config:
    """Configuration class for DRA Power Analysis System."""
    
    # Google Cloud Configuration
    GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "my-gcp-project")
    BQ_DATASET_ID = os.environ.get("BQ_DATASET_ID", "sensor_data_prod")
    BQ_TABLE_ID = os.environ.get("BQ_TABLE_ID", "pipeline_metrics")
    
    # Model Configuration
    MODEL_NAME = os.environ.get("DRA_MODEL_NAME", "gemini-2.0-flash")
    
    # System Configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
    SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))
    
    # Analysis Configuration
    DEFAULT_ANALYSIS_FOCUS = "comprehensive"
    MAX_DATE_RANGE_DAYS = 365
    CACHE_TTL_SECONDS = 300
    
    # Valid Locations
    VALID_LOCATIONS = ["STN_A001", "STN_B002", "STN_C003"]
    
    # Pump Specifications
    PUMP_SPECS: Dict[str, Dict[str, float]] = {
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
        "PUMP_C": {
            "motor_efficiency": 0.85,
            "pump_hydraulic_efficiency": 0.68,
            "vsd_efficiency": 0.88,
        }
    }
    
    # Data Quality Thresholds
    DQ_CONFIG = {
        "max_consecutive_missing_multiplier": 3,
        "pump_status_min_cycle_seconds": 60,
        "flowrate_realistic_min": 0.0,
        "flowrate_realistic_max_factor": 2.5,
        "timestamp_consistency_tolerance_seconds": 5
    }
    
    # Statistical Analysis Configuration
    STATS_CONFIG = {
        "aggregation_intervals": ["1min", "5min", "15min", "hourly", "daily"],
        "efficiency_degradation_threshold": 0.05,
        "high_power_consumption_factor": 2.0,
        "low_efficiency_absolute_threshold": 0.60
    }
    
    @classmethod
    def get_bigquery_config(cls) -> Dict[str, str]:
        """Get BigQuery configuration."""
        return {
            "project_id": cls.GCP_PROJECT_ID,
            "dataset_id": cls.BQ_DATASET_ID,
            "table_id": cls.BQ_TABLE_ID
        }
    
    @classmethod
    def get_pump_specs(cls, pump_name: str) -> Dict[str, float]:
        """Get specifications for a specific pump."""
        return cls.PUMP_SPECS.get(pump_name, {
            "motor_efficiency": 0.90,
            "pump_hydraulic_efficiency": 0.75,
            "vsd_efficiency": 0.95
        })
    
    @classmethod
    def validate_location(cls, location_id: str) -> bool:
        """Validate if location ID is valid."""
        return location_id in cls.VALID_LOCATIONS
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "gcp": cls.get_bigquery_config(),
            "model": cls.MODEL_NAME,
            "system": {
                "log_level": cls.LOG_LEVEL,
                "max_workers": cls.MAX_WORKERS,
                "session_timeout": cls.SESSION_TIMEOUT
            },
            "analysis": {
                "default_focus": cls.DEFAULT_ANALYSIS_FOCUS,
                "max_date_range_days": cls.MAX_DATE_RANGE_DAYS,
                "cache_ttl": cls.CACHE_TTL_SECONDS
            },
            "valid_locations": cls.VALID_LOCATIONS,
            "pump_specs": cls.PUMP_SPECS,
            "dq_config": cls.DQ_CONFIG,
            "stats_config": cls.STATS_CONFIG
        }


# Create a singleton instance
config = Config()