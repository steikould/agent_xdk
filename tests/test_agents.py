# File: tests/test_agents.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Import the base test framework
from test_framework import (
    BaseAgentTest, TestResult, TestStatus,
    AgentTestOrchestrator, AgentEnhancementIterator
)

# Import your actual agents
from dra_power_analysis.sub_agents.data_quality_agent import DataQualityAgent
from dra_power_analysis.sub_agents.power_calculation_agent import PowerCalculationAgent
from dra_power_analysis.sub_agents.statistical_analysis_agent import StatisticalAnalysisAgent

# ============================================================================
# ADAPTATION POINT 1: Update test methods to match your agent interfaces
# ============================================================================

class DataQualityAgentTest(BaseAgentTest):
    """
    Adapted test suite for your Data Quality Agent.
    Update the method calls to match your actual agent interface.
    """
    
    def __init__(self):
        super().__init__("DataQualityAgent")
    
    async def run_basic_tests(self, agent) -> List[TestResult]:
        """Test basic data validation functionality."""
        tests = []
        
        # Load test data
        with open('tests/test_data/sample_sensor_data.json', 'r') as f:
            test_data = json.load(f)
        
        # ADAPT THIS: Update to match your agent's actual method signature
        # Example: If your agent has a different method name or parameters
        
        # Test 1: Valid sensor data
        test_result = await self.execute_test(
            "valid_sensor_data",
            self._call_agent_method,  # Wrapper method
            agent=agent,
            method_name="process",    # Your agent's actual method name
            data=test_data["valid_data"]
        )
        tests.append(test_result)
        
        # Test 2: Data with missing values
        test_result = await self.execute_test(
            "missing_values_handling",
            self._call_agent_method,
            agent=agent,
            method_name="process",
            data=test_data["missing_values_data"]
        )
        tests.append(test_result)
        
        return tests
    
    async def _call_agent_method(self, agent, method_name: str, data: Any) -> Dict[str, Any]:
        """
        Wrapper to call agent methods with proper format.
        ADAPT THIS: Match your agent's actual interface
        """
        try:
            # If your agent uses tools format:
            if hasattr(agent, 'tools') and len(agent.tools) > 0:
                # Assuming first tool is the main processing function
                tool_func = agent.tools[0].func
                
                # Create input in your agent's expected format
                from your_agent_module import DataQualityInput  # Import your input model
                input_data = DataQualityInput(
                    sensor_data=data,
                    validation_config={}
                )
                
                result = await tool_func(input_data)
                
                # Convert result to standard format
                return {
                    "success": result.success if hasattr(result, 'success') else True,
                    "data": result.validated_data if hasattr(result, 'validated_data') else None,
                    "error": result.error_message if hasattr(result, 'error_message') else None
                }
            
            # If your agent has direct methods:
            elif hasattr(agent, method_name):
                method = getattr(agent, method_name)
                result = await method(data)
                
                # Standardize the response
                if isinstance(result, dict):
                    return result
                else:
                    return {"success": True, "data": result}
            
            else:
                return {"success": False, "error": f"Method {method_name} not found"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run_advanced_tests(self, agent) -> List[TestResult]:
        """Test advanced validation scenarios."""
        tests = []
        
        # Add your advanced test cases here
        # These might include:
        # - Large datasets
        # - Complex validation rules
        # - Performance benchmarks
        
        return tests
    
    async def run_edge_case_tests(self, agent) -> List[TestResult]:
        """Test edge cases and error handling."""
        tests = []
        
        # Add edge case tests
        # - Empty data
        # - Malformed data
        # - Extreme values
        
        return tests

# ============================================================================
# ADAPTATION POINT 2: Agent initialization and coordination
# ============================================================================

class AgentCoordinator:
    """
    Handles agent initialization and inter-agent communication if needed.
    """
    
    def __init__(self, config_path: str = "configs/agent_configs.yaml"):
        self.agents = {}
        self.config_path = config_path
        self.message_queue = asyncio.Queue()  # For inter-agent messages if needed
    
    async def initialize_agents(self) -> Dict[str, Any]:
        """
        Initialize all agents with proper configuration.
        ADAPT THIS: Based on how your agents are initialized
        """
        
        # Load configurations
        import yaml
        with open(self.config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        # Initialize each agent
        # ADAPT THIS: Match your actual agent initialization
        
        # Example 1: If agents are initialized with configs
        self.agents["DataQualityAgent"] = DataQualityAgent(
            config=configs.get("data_quality", {})
        )
        
        # Example 2: If agents need API keys or connections
        self.agents["PowerCalculationAgent"] = PowerCalculationAgent(
            pump_database=configs.get("pump_database"),
            calculation_params=configs.get("power_calculation", {})
        )
        
        # Example 3: If agents need references to other agents
        self.agents["StatisticalAnalysisAgent"] = StatisticalAnalysisAgent(
            config=configs.get("statistical_analysis", {}),
            data_quality_agent=self.agents["DataQualityAgent"]  # If needed
        )
        
        return self.agents
    
    async def coordinate_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]):
        """
        Handle inter-agent communication if needed.
        This is optional - only if your agents need to communicate during tests.
        """
        await self.message_queue.put({
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": datetime.now()
        })
    
    async def process_messages(self):
        """Process queued messages between agents."""
        while not self.message_queue.empty():
            msg = await self.message_queue.get()
            if msg["to"] in self.agents:
                # Deliver message to target agent
                # ADAPT THIS: Based on how your agents receive messages
                if hasattr(self.agents[msg["to"]], "receive_message"):
                    await self.agents[msg["to"]].receive_message(msg)