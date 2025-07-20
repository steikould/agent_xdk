"""
test_framework.py
=================

Base testing framework for multi-agent system.
This provides the core classes and utilities that specific agent tests will use.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Test Result Models
# ============================================================================

class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    agent_name: str
    status: TestStatus
    execution_time: float
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "test_name": self.test_name,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "has_output": self.output_data is not None
        }

@dataclass
class AgentTestSuite:
    """Test suite results for an agent."""
    agent_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_execution_time: float
    test_results: List[TestResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

# ============================================================================
# Base Test Class
# ============================================================================

class BaseAgentTest(ABC):
    """
    Base class for agent tests.
    All agent-specific test classes should inherit from this.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.test_results: List[TestResult] = []
    
    @abstractmethod
    async def run_basic_tests(self, agent) -> List[TestResult]:
        """
        Run basic functionality tests.
        Must be implemented by each agent test class.
        """
        pass
    
    @abstractmethod
    async def run_advanced_tests(self, agent) -> List[TestResult]:
        """
        Run advanced scenario tests.
        Must be implemented by each agent test class.
        """
        pass
    
    @abstractmethod
    async def run_edge_case_tests(self, agent) -> List[TestResult]:
        """
        Run edge case and error handling tests.
        Must be implemented by each agent test class.
        """
        pass
    
    async def execute_test(self, test_name: str, test_func: Callable, *args, **kwargs) -> TestResult:
        """
        Execute a single test and capture results.
        
        Args:
            test_name: Name of the test
            test_func: Async function to execute
            *args, **kwargs: Arguments to pass to test function
            
        Returns:
            TestResult object
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Execute the test function
            result = await test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Determine test status based on result
            if isinstance(result, dict):
                if result.get("success", False):
                    status = TestStatus.PASSED
                else:
                    status = TestStatus.FAILED
                error_message = result.get("error")
            else:
                # If result is not a dict, assume success if no exception
                status = TestStatus.PASSED
                error_message = None
                result = {"data": result}
            
            return TestResult(
                test_name=test_name,
                agent_name=self.agent_name,
                status=status,
                execution_time=execution_time,
                input_data=kwargs,
                output_data=result,
                error_message=error_message,
                timestamp=timestamp
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                agent_name=self.agent_name,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                input_data=kwargs,
                output_data=None,
                error_message=str(e),
                timestamp=timestamp
            )

# ============================================================================
# Test Orchestrator
# ============================================================================

class AgentTestOrchestrator:
    """
    Orchestrates testing across all agents.
    Manages test execution, result collection, and reporting.
    """
    
    def __init__(self):
        self.test_suites: Dict[str, BaseAgentTest] = {}
        self.test_results: Dict[str, AgentTestSuite] = {}
    
    def register_test_suite(self, agent_name: str, test_suite: BaseAgentTest):
        """Register a test suite for an agent."""
        self.test_suites[agent_name] = test_suite
    
    async def run_all_tests(self, agents: Dict[str, Any], test_level: str = "basic") -> Dict[str, Any]:
        """
        Run tests for all agents.
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            test_level: "basic", "advanced", "edge_case", or "all"
        
        Returns:
            Comprehensive test report
        """
        logger.info(f"Starting {test_level} tests for {len(agents)} agents")
        
        for agent_name, agent in agents.items():
            if agent_name in self.test_suites:
                await self.run_agent_tests(agent_name, agent, test_level)
            else:
                logger.warning(f"No test suite registered for {agent_name}")
        
        return self.generate_test_report()
    
    async def run_agent_tests(self, agent_name: str, agent: Any, test_level: str):
        """Run tests for a specific agent."""
        test_suite = self.test_suites[agent_name]
        all_results = []
        
        try:
            # Run tests based on level
            if test_level in ["basic", "all"]:
                logger.info(f"Running basic tests for {agent_name}")
                basic_results = await test_suite.run_basic_tests(agent)
                all_results.extend(basic_results)
            
            if test_level in ["advanced", "all"]:
                logger.info(f"Running advanced tests for {agent_name}")
                advanced_results = await test_suite.run_advanced_tests(agent)
                all_results.extend(advanced_results)
            
            if test_level in ["edge_case", "all"]:
                logger.info(f"Running edge case tests for {agent_name}")
                edge_results = await test_suite.run_edge_case_tests(agent)
                all_results.extend(edge_results)
        
        except Exception as e:
            logger.error(f"Error running tests for {agent_name}: {e}")
            # Create an error result for the entire suite
            all_results.append(TestResult(
                test_name="suite_execution",
                agent_name=agent_name,
                status=TestStatus.ERROR,
                execution_time=0,
                input_data={},
                output_data=None,
                error_message=f"Suite execution failed: {str(e)}",
                timestamp=datetime.now()
            ))
        
        # Compile results
        passed = sum(1 for r in all_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in all_results if r.status == TestStatus.SKIPPED)
        total_time = sum(r.execution_time for r in all_results)
        
        self.test_results[agent_name] = AgentTestSuite(
            agent_name=agent_name,
            total_tests=len(all_results),
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            skipped_tests=skipped,
            total_execution_time=total_time,
            test_results=all_results
        )
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "summary": {
                "total_agents_tested": len(self.test_results),
                "overall_success_rate": self._calculate_overall_success_rate(),
                "total_tests": sum(suite.total_tests for suite in self.test_results.values()),
                "total_passed": sum(suite.passed_tests for suite in self.test_results.values()),
                "total_failed": sum(suite.failed_tests for suite in self.test_results.values()),
                "total_errors": sum(suite.error_tests for suite in self.test_results.values()),
                "total_execution_time": sum(suite.total_execution_time for suite in self.test_results.values()),
                "timestamp": datetime.now().isoformat()
            },
            "agent_results": {},
            "failed_tests": [],
            "recommendations": []
        }
        
        # Add agent-specific results
        for agent_name, suite in self.test_results.items():
            report["agent_results"][agent_name] = {
                "success_rate": suite.success_rate,
                "total_tests": suite.total_tests,
                "passed": suite.passed_tests,
                "failed": suite.failed_tests,
                "errors": suite.error_tests,
                "execution_time": round(suite.total_execution_time, 3),
                "test_details": [result.to_dict() for result in suite.test_results]
            }
            
            # Collect failed tests
            for result in suite.test_results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    report["failed_tests"].append({
                        "agent": agent_name,
                        "test": result.test_name,
                        "status": result.status.value,
                        "error": result.error_message,
                        "execution_time": round(result.execution_time, 3)
                    })
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        return report
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all agents."""
        total_tests = sum(suite.total_tests for suite in self.test_results.values())
        total_passed = sum(suite.passed_tests for suite in self.test_results.values())
        
        if total_tests == 0:
            return 0.0
        return round((total_passed / total_tests) * 100, 2)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for agent_name, suite in self.test_results.items():
            if suite.success_rate < 50:
                recommendations.append(
                    f"{agent_name}: Critical - Success rate below 50% ({suite.success_rate:.1f}%). "
                    "Review implementation and error handling."
                )
            elif suite.success_rate < 80:
                recommendations.append(
                    f"{agent_name}: Warning - Success rate below 80% ({suite.success_rate:.1f}%). "
                    "Consider improving edge case handling."
                )
            
            if suite.total_execution_time > 30:
                recommendations.append(
                    f"{agent_name}: Performance - Total execution time is {suite.total_execution_time:.1f}s. "
                    "Consider optimizing for better performance."
                )
            
            # Check for specific error patterns
            error_messages = [r.error_message for r in suite.test_results 
                            if r.error_message and r.status == TestStatus.ERROR]
            if any("timeout" in msg.lower() for msg in error_messages if msg):
                recommendations.append(
                    f"{agent_name}: Timeout issues detected. Consider increasing timeout limits."
                )
        
        if not recommendations:
            recommendations.append("All agents performing well! Consider adding more edge case tests.")
        
        return recommendations

# ============================================================================
# Enhancement Iterator
# ============================================================================

class AgentEnhancementIterator:
    """
    Iterates through agents to enhance their configurations and capabilities
    based on test results.
    """
    
    def __init__(self, test_orchestrator: AgentTestOrchestrator):
        self.test_orchestrator = test_orchestrator
        self.enhancement_history = []
    
    async def iterate_enhancements(self, agents: Dict[str, Any], iterations: int = 3):
        """
        Iterate through enhancement cycles.
        
        Args:
            agents: Dictionary of agent instances
            iterations: Number of enhancement iterations
        """
        for i in range(iterations):
            logger.info(f"Starting enhancement iteration {i + 1}/{iterations}")
            
            # Run tests
            test_results = await self.test_orchestrator.run_all_tests(agents, "all")
            
            # Check if we've reached satisfactory performance
            success_rate = test_results["summary"]["overall_success_rate"]
            logger.info(f"Iteration {i + 1} success rate: {success_rate}%")
            
            if success_rate >= 95:
                logger.info("Achieved 95% success rate. Stopping iterations.")
                break
            
            # Analyze results and generate enhancements
            enhancements = self._analyze_and_generate_enhancements(test_results)
            
            # Apply enhancements
            applied_enhancements = {}
            for agent_name, agent_enhancements in enhancements.items():
                if agent_name in agents:
                    applied = await self._apply_enhancements(agents[agent_name], agent_enhancements)
                    if applied:
                        applied_enhancements[agent_name] = agent_enhancements
            
            # Record enhancement history
            self.enhancement_history.append({
                "iteration": i + 1,
                "success_rate": success_rate,
                "test_summary": test_results["summary"],
                "enhancements_applied": applied_enhancements,
                "timestamp": datetime.now().isoformat()
            })
    
    def _analyze_and_generate_enhancements(self, test_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze test results and generate enhancement recommendations."""
        enhancements = {}
        
        for agent_name, agent_results in test_results["agent_results"].items():
            agent_enhancements = []
            
            # Analyze failed tests
            failed_tests = [
                test for test in agent_results["test_details"]
                if test["status"] in ["failed", "error"]
            ]
            
            # Look for patterns in failures
            for failed_test in failed_tests:
                error_msg = (failed_test.get("error_message") or "").lower()
                
                if "timeout" in error_msg:
                    agent_enhancements.append({
                        "type": "performance",
                        "config": {"timeout": 60, "batch_size": 100},
                        "reason": "Timeout errors detected"
                    })
                elif "validation" in error_msg or "invalid" in error_msg:
                    agent_enhancements.append({
                        "type": "validation",
                        "config": {"strict_mode": False, "auto_correct": True},
                        "reason": "Validation errors detected"
                    })
                elif "memory" in error_msg:
                    agent_enhancements.append({
                        "type": "resource",
                        "config": {"max_memory": "2GB", "streaming": True},
                        "reason": "Memory errors detected"
                    })
                elif "connection" in error_msg:
                    agent_enhancements.append({
                        "type": "connectivity",
                        "config": {"retry_count": 3, "connection_timeout": 30},
                        "reason": "Connection errors detected"
                    })
            
            # Remove duplicate enhancement types
            seen_types = set()
            unique_enhancements = []
            for enh in agent_enhancements:
                if enh["type"] not in seen_types:
                    seen_types.add(enh["type"])
                    unique_enhancements.append(enh)
            
            if unique_enhancements:
                enhancements[agent_name] = unique_enhancements
        
        return enhancements
    
    async def _apply_enhancements(self, agent: Any, enhancements: List[Dict[str, Any]]) -> bool:
        """Apply enhancements to an agent."""
        applied_any = False
        
        for enhancement in enhancements:
            try:
                # Try different methods of applying configuration
                if hasattr(agent, "update_config"):
                    await agent.update_config(enhancement["config"])
                    applied_any = True
                elif hasattr(agent, "config"):
                    # Direct config update
                    for key, value in enhancement["config"].items():
                        if hasattr(agent.config, key):
                            setattr(agent.config, key, value)
                            applied_any = True
                else:
                    # Try setting attributes directly
                    for key, value in enhancement["config"].items():
                        if hasattr(agent, key):
                            setattr(agent, key, value)
                            applied_any = True
                
                if applied_any:
                    logger.info(f"Applied {enhancement['type']} enhancement: {enhancement['reason']}")
            
            except Exception as e:
                logger.warning(f"Failed to apply enhancement: {e}")
        
        return applied_any
    
    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get summary of enhancement iterations."""
        if not self.enhancement_history:
            return {"message": "No enhancement iterations performed"}
        
        initial_rate = self.enhancement_history[0]["success_rate"]
        final_rate = self.enhancement_history[-1]["success_rate"]
        
        return {
            "iterations_performed": len(self.enhancement_history),
            "initial_success_rate": initial_rate,
            "final_success_rate": final_rate,
            "improvement": round(final_rate - initial_rate, 2),
            "history": self.enhancement_history
        }

# ============================================================================
# Utility Functions
# ============================================================================

def save_test_report(report: Dict[str, Any], filename: str = "test_report.json"):
    """Save test report to JSON file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Test report saved to {filename}")

def print_test_summary(report: Dict[str, Any]):
    """Print a formatted test summary."""
    summary = report["summary"]
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total Agents Tested: {summary['total_agents_tested']}")
    print(f"Overall Success Rate: {summary['overall_success_rate']}%")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"  - Passed: {summary['total_passed']}")
    print(f"  - Failed: {summary['total_failed']}")
    print(f"  - Errors: {summary['total_errors']}")
    print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
    
    print("\nAgent Results:")
    for agent_name, results in report["agent_results"].items():
        print(f"\n  {agent_name}:")
        print(f"    Success Rate: {results['success_rate']}%")
        print(f"    Tests: {results['total_tests']} (P:{results['passed']} F:{results['failed']} E:{results['errors']})")
        print(f"    Time: {results['execution_time']}s")
    
    if report["failed_tests"]:
        print("\n" + "-"*60)
        print("FAILED TESTS:")
        for failure in report["failed_tests"][:10]:  # Show first 10
            print(f"  - {failure['agent']}: {failure['test']} ({failure['error']})")
        if len(report["failed_tests"]) > 10:
            print(f"  ... and {len(report['failed_tests']) - 10} more")
    
    if report["recommendations"]:
        print("\n" + "-"*60)
        print("RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  * {rec}")  # Changed from bullet point to asterisk
    
    print("="*60 + "\n")