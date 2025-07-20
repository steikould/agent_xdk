# File: tests/run_tests.py

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import adapted test classes
from test_agents import DataQualityAgentTest, AgentCoordinator
from test_framework import AgentTestOrchestrator, AgentEnhancementIterator

async def main():
    """Main test execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run agent tests")
    parser.add_argument("--level", choices=["basic", "advanced", "edge_case", "all"], 
                       default="basic", help="Test level to run")
    parser.add_argument("--agents", nargs="+", help="Specific agents to test")
    parser.add_argument("--enhance", action="store_true", help="Run enhancement iterations")
    parser.add_argument("--output", default="test_results.json", help="Output file for results")
    args = parser.parse_args()
    
    # Initialize agent coordinator
    print("Initializing agents...")
    coordinator = AgentCoordinator()
    agents = await coordinator.initialize_agents()
    
    # Filter agents if specified
    if args.agents:
        agents = {k: v for k, v in agents.items() if k in args.agents}
    
    # Initialize test orchestrator with adapted test classes
    orchestrator = AgentTestOrchestrator()
    
    # ADAPT THIS: Register your adapted test classes
    orchestrator.test_suites = {
        "DataQualityAgent": DataQualityAgentTest(),
        # Add other adapted test classes here
    }
    
    # Run tests
    print(f"\nRunning {args.level} tests for {len(agents)} agents...")
    results = await orchestrator.run_all_tests(agents, args.level)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total Agents Tested: {results['summary']['total_agents_tested']}")
    print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.2f}%")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['total_passed']}")
    print(f"Failed: {results['summary']['total_failed']}")
    print(f"Errors: {results['summary']['total_errors']}")
    print(f"Execution Time: {results['summary']['total_execution_time']:.2f}s")
    
    # Run enhancement iterations if requested
    if args.enhance:
        print("\n" + "="*60)
        print("RUNNING ENHANCEMENT ITERATIONS")
        print("="*60)
        enhancer = AgentEnhancementIterator(orchestrator)
        await enhancer.iterate_enhancements(agents, iterations=3)
        
        # Run final tests
        print("\nRunning final tests after enhancements...")
        final_results = await orchestrator.run_all_tests(agents, "all")
        
        # Compare improvement
        initial_success = results['summary']['overall_success_rate']
        final_success = final_results['summary']['overall_success_rate']
        print(f"\nImprovement: {initial_success:.2f}% -> {final_success:.2f}%")
        
        results = final_results
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Print failed tests for debugging
    if results['failed_tests']:
        print("\n" + "="*60)
        print("FAILED TESTS")
        print("="*60)
        for failure in results['failed_tests']:
            print(f"- {failure['agent']}: {failure['test']}")
            print(f"  Error: {failure['error']}")

if __name__ == "__main__":
    asyncio.run(main())
