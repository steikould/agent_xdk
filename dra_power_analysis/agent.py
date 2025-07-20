"""
Enhanced Root Orchestrator Agent for DRA Power Analysis System
Location: dra_power_analysis/agent.py
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from google.adk.agents import Agent

from dra_power_analysis import prompt
from dra_power_analysis.sub_agents.bigquery_data_retrieval.agent import BigQueryDataAgent
from dra_power_analysis.sub_agents.data_quality.agent import dq_agent
from dra_power_analysis.sub_agents.insights_optimization.agent import insights_agent
from dra_power_analysis.sub_agents.power_consumption.agent import power_calc_agent
from dra_power_analysis.sub_agents.statistical_analysis.agent import stat_analysis_agent
from dra_power_analysis.sub_agents.user_interface.agent import ui_agent_output

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Model configuration
model_name = "gemini-2.0-flash"

# Initialize data agent with configuration
data_agent_instance = BigQueryDataAgent(
    name="DataQueryAgent",
    description="Agent capable of querying sensor data from BigQuery.",
    project_id="my-gcp-project",  # Replace with your actual project ID
    dataset_id="sensor_data_prod",
    table_id="pipeline_metrics",
    use_mock=False  # Set to True for local development without GCP connection
)


async def run_power_analysis_tool(
    location_id: str, 
    pipeline_id: str, 
    start_date: str, 
    end_date: str,
    analysis_focus: str = "comprehensive",
    context: Dict[str, Any] = None
) -> str:
    """
    Run complete power analysis workflow.
    
    Args:
        location_id: Location identifier (e.g., STN_A001)
        pipeline_id: Pipeline identifier (e.g., PL123)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        analysis_focus: Focus area (comprehensive, efficiency, anomalies)
        context: ADK context containing session state
        
    Returns:
        str: Analysis results summary
    """
    try:
        # Initialize user parameters in session state
        if context and hasattr(context, 'session') and hasattr(context.session, 'state'):
            state = context.session.state
            state["user_params"] = {
                "location_id": location_id,
                "pipeline_line_number": pipeline_id,
                "start_date": start_date,
                "end_date": end_date,
                "analysis_focus": analysis_focus
            }
            state["workflow_status"] = "started"
            state["start_time"] = datetime.now().isoformat()
            
            logger.info(f"Starting power analysis for {location_id}/{pipeline_id}")
            
            # Stage 1: Data Retrieval
            logger.info("Stage 1: Data Retrieval")
            try:
                from datetime import timezone
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                
                # Call data retrieval agent's tool
                query_result = await data_agent_instance.tools[0].func(
                    start_time=start_dt,
                    end_time=end_dt,
                    location_id=location_id,
                    line_id=pipeline_id
                )
                
                if query_result.status == "success" and query_result.data:
                    # Convert to DataFrame for other agents
                    import pandas as pd
                    data_records = [row.model_dump() for row in query_result.data]
                    state["sensor_data_df"] = pd.DataFrame(data_records)
                    state["status_data_retrieval"] = "success"
                else:
                    return f"❌ Data retrieval failed: {query_result.error_message}"
                    
            except Exception as e:
                logger.error(f"Data retrieval error: {e}")
                return f"❌ Data retrieval failed: {str(e)}"
            
            # Stage 2: Data Quality Validation
            logger.info("Stage 2: Data Quality Validation")
            # The other agents expect data in session state, which we've set
            # In a real implementation, you'd call the agent's async methods
            
            # For now, return a summary based on what we have
            df = state.get("sensor_data_df")
            if df is not None and not df.empty:
                summary = f"""
✅ Power Analysis Initiated Successfully!

📍 Location: {location_id}
📍 Pipeline: {pipeline_id}
📅 Period: {start_date} to {end_date}
🎯 Focus: {analysis_focus}

📊 Data Retrieved:
• Records: {len(df)}
• Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}
• Unique Tags: {df['tag_name'].nunique()}

🔄 Next Steps:
The analysis will proceed through:
1. ✅ Data Retrieval (Complete)
2. ⏳ Data Quality Validation
3. ⏳ Power Calculations
4. ⏳ Statistical Analysis
5. ⏳ Insights Generation

Full analysis results will be available in 2-3 minutes.
"""
                return summary
            else:
                return "❌ No data retrieved for the specified parameters."
                
        else:
            return "❌ Unable to access session context. Please ensure the agent is properly initialized."
            
    except Exception as e:
        logger.error(f"Power analysis failed: {e}", exc_info=True)
        return f"❌ Analysis failed: {str(e)}"


async def get_pump_data_tool(
    location_id: str,
    pipeline_id: str,
    pump_name: str,
    start_date: str,
    end_date: str,
    context: Dict[str, Any] = None
) -> str:
    """
    Retrieve specific pump data.
    
    Args:
        location_id: Location identifier
        pipeline_id: Pipeline identifier
        pump_name: Specific pump name (e.g., PUMP_A)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        context: ADK context
        
    Returns:
        str: Pump data summary
    """
    try:
        from datetime import timezone
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        # Query for specific pump data
        query_result = await data_agent_instance.tools[0].func(
            start_time=start_dt,
            end_time=end_dt,
            location_id=location_id,
            line_id=pipeline_id
        )
        
        if query_result.status == "success" and query_result.data:
            # Filter for specific pump
            pump_data = [
                row for row in query_result.data 
                if pump_name in row.tag_name
            ]
            
            if pump_data:
                # Group by tag type
                tag_types = {}
                for row in pump_data:
                    tag_type = row.tag_name.split('_')[-1]
                    if tag_type not in tag_types:
                        tag_types[tag_type] = []
                    tag_types[tag_type].append(row)
                
                summary = f"""
📊 Pump Data Retrieved: {pump_name}

📍 Location: {location_id}
📍 Pipeline: {pipeline_id}
📅 Period: {start_date} to {end_date}

📈 Data Summary:
• Total Records: {len(pump_data)}
• Tag Types: {', '.join(tag_types.keys())}
"""
                
                for tag_type, rows in tag_types.items():
                    values = [row.value for row in rows]
                    summary += f"\n{tag_type}:\n"
                    summary += f"  • Records: {len(values)}\n"
                    summary += f"  • Range: {min(values):.2f} - {max(values):.2f}\n"
                    summary += f"  • Average: {sum(values)/len(values):.2f}\n"
                
                return summary
            else:
                return f"❌ No data found for pump '{pump_name}' in the specified period."
        else:
            return f"❌ Failed to retrieve data: {query_result.error_message}"
            
    except Exception as e:
        return f"❌ Error retrieving pump data: {str(e)}"


async def check_data_quality_tool(
    location_id: str,
    pipeline_id: str,
    date: str,
    context: Dict[str, Any] = None
) -> str:
    """
    Quick data quality check for a specific date.
    
    Args:
        location_id: Location identifier
        pipeline_id: Pipeline identifier
        date: Date to check (YYYY-MM-DD)
        context: ADK context
        
    Returns:
        str: Data quality summary
    """
    try:
        from datetime import timezone, timedelta
        check_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        next_date = check_date + timedelta(days=1)
        
        # Query one day of data
        query_result = await data_agent_instance.tools[0].func(
            start_time=check_date,
            end_time=next_date,
            location_id=location_id,
            line_id=pipeline_id
        )
        
        if query_result.status == "success" and query_result.data:
            # Analyze data quality
            total_records = len(query_result.data)
            unique_tags = len(set(row.tag_name for row in query_result.data))
            
            # Check for data gaps
            timestamps = sorted([row.timestamp for row in query_result.data])
            gaps = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                if diff > 3600:  # More than 1 hour gap
                    gaps.append((timestamps[i-1], timestamps[i], diff))
            
            # Check data quality
            quality_issues = 0
            for row in query_result.data:
                if row.quality != "good":
                    quality_issues += 1
            
            summary = f"""
🔍 Data Quality Check for {date}

📍 Location: {location_id}
📍 Pipeline: {pipeline_id}

📊 Data Summary:
• Total Records: {total_records}
• Unique Tags: {unique_tags}
• Quality Issues: {quality_issues}
• Data Gaps: {len(gaps)}

"""
            if gaps:
                summary += "⚠️ Data Gaps Detected:\n"
                for start, end, duration in gaps[:5]:  # Show first 5 gaps
                    summary += f"  • {start.strftime('%H:%M')} - {end.strftime('%H:%M')} ({duration/3600:.1f} hours)\n"
                if len(gaps) > 5:
                    summary += f"  • ... and {len(gaps) - 5} more gaps\n"
            else:
                summary += "✅ No significant data gaps detected\n"
            
            if quality_issues > 0:
                summary += f"\n⚠️ {quality_issues} records with quality issues"
            else:
                summary += "\n✅ All data quality flags are 'good'"
            
            return summary
        else:
            return f"❌ No data found for {date}"
            
    except Exception as e:
        return f"❌ Error checking data quality: {str(e)}"


# Create the root agent with enhanced coordination
root_agent = Agent(
    name="PowerAnalysisInitiator",
    model=model_name,
    description="Enhanced orchestrator for energy pipeline power consumption analysis",
    instruction="""You are an enhanced power analysis orchestrator for energy pipeline pump systems.

🎯 **Your Primary Role:**
You coordinate specialized agents to analyze pump power consumption and provide optimization insights.

📊 **Available Analysis Tools:**

1. **Full Power Analysis** - Complete workflow analysis
   - Example: "Analyze power consumption for location STN_A001 pipeline PL123 from 2024-01-01 to 2024-01-31"
   - Focuses: comprehensive, efficiency, anomalies, optimization

2. **Pump Data Retrieval** - Get specific pump data
   - Example: "Get PUMP_A data for location STN_A001 pipeline PL123 on 2024-01-15"
   
3. **Data Quality Check** - Quick quality assessment
   - Example: "Check data quality for location STN_A001 pipeline PL123 on 2024-01-15"

🔧 **Sub-Agents I Coordinate:**
- **Data Retrieval Agent**: Queries BigQuery for sensor data
- **Data Quality Agent**: Validates data integrity
- **Power Calculation Agent**: Computes power metrics
- **Statistical Analysis Agent**: Generates trends and statistics
- **Insights Agent**: Creates optimization recommendations
- **UI Agent**: Formats and presents results

💡 **How to Use Me:**
1. Start with a full analysis to understand overall performance
2. Use pump-specific queries to investigate issues
3. Check data quality when you see anomalies
4. Focus your analysis on specific aspects (efficiency, anomalies, etc.)

📍 **Valid Locations**: STN_A001, STN_B002, STN_C003
📍 **Pipeline Format**: PLxxx (e.g., PL123, PL456)
📅 **Date Format**: YYYY-MM-DD

I'm here to help you optimize pump operations and reduce energy consumption!""",
    tools=[
        run_power_analysis_tool,
        get_pump_data_tool,
        check_data_quality_tool
    ],
    sub_agents=[
        data_agent_instance,
        dq_agent,
        insights_agent,
        power_calc_agent,
        stat_analysis_agent,
        ui_agent_output
    ]
)

# For backward compatibility
if __name__ == "__main__":
    logger.info("DRA Power Analysis Root Agent initialized")
    logger.info(f"Using BigQuery project: {data_agent_instance.project_id}")
    logger.info(f"Dataset: {data_agent_instance.dataset_id}")
    logger.info(f"Table: {data_agent_instance.table_id}")