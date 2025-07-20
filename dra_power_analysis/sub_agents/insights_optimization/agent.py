"""
Insights and Optimization Agent for DRA Power Analysis
Location: dra_power_analysis/sub_agents/insights_optimization/agent.py
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
class InsightsInput(BaseModel):
    """Input model for insights generation."""
    statistical_summary: Dict[str, Any] = Field(..., description="Statistical analysis summary")
    efficiency_trends: Dict[str, Any] = Field(..., description="Efficiency trend data")
    data_quality_report: Optional[Dict[str, Any]] = Field(
        default={},
        description="Data quality assessment report"
    )
    user_preferences: Optional[Dict[str, Any]] = Field(
        default={},
        description="User preferences for insights"
    )


class InsightsOutput(BaseModel):
    """Output model for insights and recommendations."""
    success: bool
    optimization_opportunities: Optional[List[Dict[str, Any]]] = None
    executive_summary_points: Optional[List[str]] = None
    risk_assessment_notes: Optional[List[str]] = None
    final_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class InsightsOptimizationAgent(BaseAgent):
    """
    Generates actionable recommendations for operations based on analysis results.
    """

    # Class constants
    EFFICIENCY_DEGRADATION_THRESHOLD = 0.05  # 5% drop from baseline/average
    HIGH_POWER_CONSUMPTION_FACTOR = 2.0  # Max power > X times median power
    LOW_EFFICIENCY_ABSOLUTE_THRESHOLD = 0.60

    def __init__(self, name: str = "InsightsOptimizationAgent",
                 description: str = "Generates optimization insights", **kwargs):
        """Initialize the Insights Optimization Agent."""
        super().__init__(name=name, description=description, **kwargs)
        logger.info(f"InsightsOptimizationAgent '{self.name}' initialized.")
        
        # Set up tools
        self._setup_tools()

    def _setup_tools(self):
        """Set up agent tools."""
        
        async def generate_optimization_insights(input_data: InsightsInput) -> InsightsOutput:
            """
            Generate optimization insights from analysis results.
            
            Args:
                input_data: Statistical summary and efficiency trends
                
            Returns:
                Optimization insights and recommendations
            """
            try:
                stats_summary = input_data.statistical_summary
                efficiency_trends_data = input_data.efficiency_trends
                dq_report = input_data.data_quality_report or {}
                
                # Generate insights
                opportunities, summary_points, risk_notes = self._generate_insights(
                    stats_summary, efficiency_trends_data, dq_report
                )
                
                # Rank opportunities by priority
                ranked_opportunities = sorted(opportunities, key=lambda x: x.get("priority", 99))
                
                # Create final report
                final_report = self._create_final_report(
                    ranked_opportunities, summary_points, risk_notes, 
                    stats_summary, efficiency_trends_data, dq_report
                )
                
                return InsightsOutput(
                    success=True,
                    optimization_opportunities=ranked_opportunities,
                    executive_summary_points=list(set(summary_points)),
                    risk_assessment_notes=list(set(risk_notes)),
                    final_report=final_report
                )
                
            except Exception as e:
                logger.error(f"Insights generation failed: {e}", exc_info=True)
                return InsightsOutput(
                    success=False,
                    error_message=f"Insights generation error: {str(e)}"
                )
        
        async def prioritize_actions(input_data: InsightsInput) -> InsightsOutput:
            """
            Prioritize optimization actions based on impact and feasibility.
            
            Args:
                input_data: Analysis results for prioritization
                
            Returns:
                Prioritized action plan
            """
            try:
                # Generate basic insights first
                stats_summary = input_data.statistical_summary
                efficiency_trends_data = input_data.efficiency_trends
                
                # Create prioritized action items
                actions = []
                
                # Check for critical efficiency issues
                for pump_name, stats in stats_summary.items():
                    if pump_name != "SystemTotal" and isinstance(stats, dict):
                        # Check efficiency trends
                        pump_trend = efficiency_trends_data.get(pump_name, [])
                        if pump_trend and isinstance(pump_trend, list) and len(pump_trend) > 0:
                            # Get latest efficiency
                            latest_efficiency = pump_trend[-1].get("calculated_pump_efficiency", 1.0)
                            if latest_efficiency < self.LOW_EFFICIENCY_ABSOLUTE_THRESHOLD:
                                actions.append({
                                    "action": f"Immediate maintenance for {pump_name}",
                                    "priority": 1,
                                    "impact": "High",
                                    "effort": "Medium",
                                    "reason": f"Efficiency at {latest_efficiency:.1%} - below critical threshold"
                                })
                
                # Sort by priority
                actions.sort(key=lambda x: x["priority"])
                
                return InsightsOutput(
                    success=True,
                    optimization_opportunities=actions,
                    executive_summary_points=[
                        f"{len(actions)} high-priority actions identified",
                        "Focus on pumps with efficiency below 60%"
                    ]
                )
                
            except Exception as e:
                return InsightsOutput(
                    success=False,
                    error_message=f"Action prioritization error: {str(e)}"
                )
        
        # Register tools
        self.tools = [
            FunctionTool(func=generate_optimization_insights),
            FunctionTool(func=prioritize_actions)
        ]

    def _generate_insights(self, stats_summary: Dict, efficiency_trends_data: Dict, 
                          dq_report: Dict) -> Tuple[List[Dict], List[str], List[str]]:
        """Generate insights from analysis data."""
        opportunities: List[Dict[str, Any]] = []
        summary_points: List[str] = []
        risk_notes: List[str] = []

        for pump_name, pump_stats in stats_summary.items():
            if pump_name == "SystemTotal" or not isinstance(pump_stats, dict):
                continue
            
            logger.info(f"Generating insights for pump: {pump_name}")

            # Check efficiency trends
            pump_eff_trend = efficiency_trends_data.get(pump_name, [])
            if pump_eff_trend and isinstance(pump_eff_trend, list) and len(pump_eff_trend) > 0:
                # Calculate average and current efficiency
                efficiencies = [row.get("calculated_pump_efficiency", 0) for row in pump_eff_trend]
                valid_efficiencies = [e for e in efficiencies if e > 0]
                
                if valid_efficiencies:
                    avg_efficiency = np.mean(valid_efficiencies)
                    current_efficiency = valid_efficiencies[-1]

                    if current_efficiency < self.LOW_EFFICIENCY_ABSOLUTE_THRESHOLD:
                        opportunities.append({
                            "pump_name": pump_name,
                            "type": "Low Efficiency Operation",
                            "description": f"Pump {pump_name} operating at low efficiency ({current_efficiency:.2%}).",
                            "recommendation": f"Investigate {pump_name} for mechanical/operational issues. Consider refurbishment.",
                            "priority": 1,
                            "estimated_savings_potential": "High"
                        })
                        summary_points.append(f"{pump_name}: Critically low efficiency ({current_efficiency:.2%}). Investigation needed.")
                        risk_notes.append(f"Delaying {pump_name} low efficiency investigation risks failure/high costs.")
                    
                    elif avg_efficiency > 1e-3 and (avg_efficiency - current_efficiency) / avg_efficiency > self.EFFICIENCY_DEGRADATION_THRESHOLD:
                        opportunities.append({
                            "pump_name": pump_name,
                            "type": "Efficiency Degradation",
                            "description": f"{pump_name} efficiency degraded by {((avg_efficiency - current_efficiency) / avg_efficiency):.2%}.",
                            "recommendation": f"Schedule inspection for {pump_name} for wear/fouling.",
                            "priority": 2,
                            "estimated_savings_potential": "Medium"
                        })
                        summary_points.append(f"{pump_name}: Efficiency degraded. Plan maintenance.")

            # Check power consumption patterns
            max_power = pump_stats.get("max", 0)
            median_power = pump_stats.get("median", 0)
            if pd.notna(max_power) and pd.notna(median_power) and median_power > 1e-3:
                if max_power > median_power * self.HIGH_POWER_CONSUMPTION_FACTOR:
                    opportunities.append({
                        "pump_name": pump_name,
                        "type": "Intermittent High Power Consumption",
                        "description": f"{pump_name} shows high power peaks (max: {max_power:.2f} kW vs median: {median_power:.2f} kW).",
                        "recommendation": f"Investigate {pump_name} operations during peak power. Check for blockages/control issues.",
                        "priority": 3,
                        "estimated_savings_potential": "Variable"
                    })
                    summary_points.append(f"{pump_name}: Shows intermittent high power peaks.")

            # Check data quality issues
            if dq_report and dq_report.get("detailed_issues"):
                pump_dq_issues = [
                    iss for iss in dq_report["detailed_issues"] 
                    if isinstance(iss, dict) and pump_name in str(iss.get("tag_name", ""))
                ]
                is_low_eff = any(
                    opp.get("pump_name") == pump_name and opp.get("type") == "Low Efficiency Operation" 
                    for opp in opportunities
                )
                if pump_dq_issues and is_low_eff:
                    opportunities.append({
                        "pump_name": pump_name,
                        "type": "Combined Maintenance Needed",
                        "description": f"{pump_name} has low efficiency and {len(pump_dq_issues)} sensor data quality issues.",
                        "recommendation": f"Prioritize combined maintenance for {pump_name}: address efficiency and sensor issues.",
                        "priority": 1,
                        "estimated_savings_potential": "High"
                    })
                    summary_points.append(f"{pump_name}: Low efficiency & sensor DQ issues. Priority maintenance.")

        # Add general insights
        if not opportunities and stats_summary and any(k != "SystemTotal" for k in stats_summary.keys()):
            summary_points.append("System components analyzed appear to be operating within expected parameters. Continuous monitoring advised.")
        elif not stats_summary or all(k == "SystemTotal" for k in stats_summary.keys()):
            summary_points.append("No per-pump statistical summary available to generate specific insights.")

        return opportunities, summary_points, risk_notes

    def _create_final_report(self, opportunities: List[Dict], summary_points: List[str], 
                           risk_notes: List[str], stats_summary: Dict,
                           efficiency_trends: Dict, dq_report: Dict) -> Dict[str, Any]:
        """Create the final consolidated report."""
        final_report = {
            "optimization_opportunities": opportunities,
            "executive_summary_points": summary_points,
            "risk_assessment_notes": risk_notes,
            "statistical_highlights": {
                "pumps_analyzed": [k for k in stats_summary.keys() if k != "SystemTotal"],
                "system_avg_power": stats_summary.get("SystemTotal", {}).get("mean", "N/A")
            },
            "data_quality_summary": {
                "issues_found": len(dq_report.get("detailed_issues", [])),
                "quality_score": 1.0 - (len(dq_report.get("detailed_issues", [])) / 100)  # Simple score
            },
            "efficiency_summary": {}
        }
        
        # Add efficiency summary
        for pump_name, trend_data in efficiency_trends.items():
            if isinstance(trend_data, list) and len(trend_data) > 0:
                efficiencies = [row.get("calculated_pump_efficiency", 0) for row in trend_data]
                valid_effs = [e for e in efficiencies if e > 0]
                if valid_effs:
                    final_report["efficiency_summary"][pump_name] = {
                        "current": valid_effs[-1],
                        "average": np.mean(valid_effs),
                        "trend": "degrading" if valid_effs[-1] < np.mean(valid_effs) else "stable"
                    }
        
        # Create executive summary string
        exec_summary = "Executive Summary:\n"
        if summary_points:
            for point in summary_points:
                exec_summary += f"- {point}\n"
        else:
            exec_summary += "- Analysis complete. No specific high-priority points generated.\n"
        
        # Add key metrics
        system_stats = stats_summary.get("SystemTotal", {})
        if system_stats:
            exec_summary += f"- System average power: {system_stats.get('mean', 'N/A')} kW\n"
            exec_summary += f"- Peak power: {system_stats.get('max', 'N/A')} kW\n"
        
        final_report["executive_summary"] = exec_summary
        final_report["status"] = "success"
        
        return final_report


# Create singleton instance for backward compatibility
insights_agent = InsightsOptimizationAgent(
    name="InsightsOptimizationAgent",
    description="Generates actionable recommendations for operations."
)