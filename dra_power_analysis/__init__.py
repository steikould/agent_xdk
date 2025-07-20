# dra_power_analysis/__init__.py

"""
DRA Power Analysis System

A multi-agent system for analyzing pump power consumption in energy pipelines.
"""

__version__ = "0.1.0"
APP_NAME = "DRA Power Analysis System"

# Import the main root agent for easy access
from .agent import root_agent

# Make key components available at package level
__all__ = [
    "root_agent",
    "APP_NAME",
    "__version__"
]