# Industrial Data Agents Collection

This directory contains specialized agents for industrial data management and digital twin applications, built using Google's Agent Development Kit (ADK).

## 📋 Table of Contents

- [Overview](#overview)
- [Available Agents](#available-agents)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Integration Patterns](#integration-patterns)
- [Best Practices](#best-practices)

## Overview

The Industrial Data Agents collection provides AI-powered tools for managing industrial data, equipment documentation, and digital twin implementations. These agents work together to provide comprehensive industrial asset management capabilities.

### Key Features

- 🏭 Industrial equipment documentation management
- 📊 BigQuery data inventory and analysis
- 🔧 Digital twin guidance and implementation
- 🔄 Multi-agent coordination for complex workflows
- 📈 Real-time data integration capabilities

## Available Agents

### 1. BigQuery Inventory Agent 📊
**Purpose**: Inventory and analyze BigQuery datasets with SharePoint integration

**Key Features**:
- Complete BigQuery project inventory
- Dataset and table discovery
- SQL query execution with validation
- SharePoint document integration
- Cross-reference data lake with documentation

**Use Cases**:
- Data lake auditing
- Schema documentation
- Data lineage tracking
- Compliance reporting

### 2. Digital Twinning Agent 🔧
**Purpose**: Expert guidance on digital twin implementation

**Key Features**:
- Data requirements specification
- Simulation guidance
- Technology stack recommendations
- Implementation roadmaps
- ROI and benefits analysis

**Use Cases**:
- Digital twin project planning
- Technology selection
- Requirements gathering
- Best practices consultation

### 3. Industrial Equipment Agent 🏭
**Purpose**: Access and manage industrial equipment documentation

**Key Features**:
- Equipment manual retrieval
- Specification lookup
- Maintenance documentation
- Safety data sheets access
- Equipment categorization

**Use Cases**:
- Maintenance planning
- Equipment troubleshooting
- Compliance documentation
- Training material access

## Architecture

### System Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Industrial Data Platform                  │
├─────────────────┬────────────────┬─────────────────────┤
│   BigQuery      │   SharePoint   │   Equipment DB      │
│   Data Lake     │   Documents    │   Specifications    │
└────────┬────────┴───────┬────────┴──────────┬─────────┘
         │                │                    │
    ┌────▼────┐     ┌─────▼─────┐      ┌─────▼─────┐
    │BigQuery │     │Industrial │      │ Digital   │
    │Inventory│     │Equipment  │      │ Twinning  │
    │ Agent   │     │  Agent    │      │  Agent    │
    └────┬────┘     └─────┬─────┘      └─────┬─────┘
         │                │                    │
         └────────────────┴────────────────────┘
                          │
                   ┌──────▼──────┐
                   │   Google     │
                   │   ADK Core   │
                   └─────────────┘
```

### Agent Coordination Pattern

```python
# Example: Coordinated digital twin data discovery
async def discover_digital_twin_data(asset_name: str):
    # 1. Get equipment specifications
    equipment_docs = await equipment_agent.chat(
        f"Find specifications for {asset_name}"
    )
    
    # 2. Query data availability in BigQuery
    data_inventory = await bigquery_agent.chat(
        f"Search for tables containing {asset_name} sensor data"
    )
    
    # 3. Get digital twin requirements
    requirements = await digital_twin_agent.chat(
        f"What data is needed for {asset_name} digital twin?"
    )
    
    # 4. Gap analysis
    return analyze_gaps(equipment_docs, data_inventory, requirements)
```

## Installation

### Prerequisites

- Python 3.8+
- Google Cloud Project (for BigQuery agent)
- SharePoint access (for document agents)
- Google ADK installed

### Setup Steps

```bash
# Clone repository
git clone <repository-url>
cd industrial-data-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install base requirements
pip install google-adk

# Install agent-specific requirements
cd bigquery-inventory-agent
pip install -r requirements.txt

cd ../digital-twinning-agent
pip install -r requirements.txt

cd ../industrial-equipment-agent
pip install -r requirements.txt
```

## Configuration

### Environment Variables

#### BigQuery Inventory Agent
```bash
# Google Cloud Configuration
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# SharePoint Configuration (optional)
export SHAREPOINT_SITE_URL="https://company.sharepoint.com/sites/data"
export SHAREPOINT_CLIENT_ID="your-client-id"
export SHAREPOINT_CLIENT_SECRET="your-client-secret"
```

#### Industrial Equipment Agent
```bash
# SharePoint Configuration
export INDUSTRIAL_EQUIPMENT_SHAREPOINT_URL="https://company.sharepoint.com/sites/equipment"
export INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_ID="your-client-id"
export INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_SECRET="your-client-secret"
```

#### Digital Twinning Agent
```bash
# No external configuration required - knowledge-based agent
export DIGITAL_TWIN_MODEL="gemini-2.0-flash-exp"  # Optional: specify model
```

## Usage Examples

### BigQuery Inventory Agent

```python
from bigquery_inventory_agent.agent import BigQueryInventoryAgent

# Initialize agent
agent = BigQueryInventoryAgent(
    project_id="your-project-id",
    sharepoint_site_url="https://company.sharepoint.com/sites/data",
    sharepoint_client_id="client-id",
    sharepoint_client_secret="client-secret"
)

# Get inventory
response = agent.chat_sync("Show me the BigQuery inventory with schema details")
print(response)

# Execute query
response = agent.chat_sync("""
    Execute this query: 
    SELECT table_name, row_count 
    FROM `project.dataset.INFORMATION_SCHEMA.TABLES` 
    WHERE table_type = 'BASE TABLE'
""")
print(response)

# Cross-reference with documentation
response = agent.chat_sync(
    "Compare sensor_data tables with 'Data Dictionary.xlsx' in SharePoint"
)
print(response)
```

### Digital Twinning Agent

```python
from digital_twinning_agent.agent import DigitalTwinningAgent

# Initialize agent
agent = DigitalTwinningAgent()

# Get requirements
response = agent.chat_sync(
    "What data do I need for a pump digital twin?"
)
print(response)

# Get implementation guide
response = agent.chat_sync(
    "Provide a roadmap for implementing a digital twin for industrial compressors"
)
print(response)

# Technology recommendations
response = agent.chat_sync(
    "What technology stack should I use for real-time digital twins?"
)
print(response)
```

### Industrial Equipment Agent

```python
from industrial_equipment_agent.agent import IndustrialEquipmentAgent

# Initialize agent
agent = IndustrialEquipmentAgent(
    site_url="https://company.sharepoint.com/sites/equipment",
    client_id="client-id",
    client_secret="client-secret"
)

# Find equipment documentation
response = agent.chat_sync(
    "Find all pump documentation"
)
print(response)

# Get specific manual
response = agent.chat_sync(
    "Get the maintenance manual for Centrifugal-Pump-Model-X200"
)
print(response)

# Search by equipment type
response = agent.chat_sync(
    "Show me all compressor specifications"
)
print(response)
```

## Integration Patterns

### 1. Data Discovery Workflow

```python
async def industrial_data_discovery(equipment_type: str):
    """Discover all data sources for equipment type."""
    
    # Step 1: Find equipment in documentation
    equipment_docs = await equipment_agent.chat(
        f"List all {equipment_type} equipment"
    )
    
    # Step 2: Search for data tables
    data_tables = await bigquery_agent.chat(
        f"Find tables containing {equipment_type} data"
    )
    
    # Step 3: Get digital twin requirements
    dt_requirements = await digital_twin_agent.chat(
        f"Data requirements for {equipment_type} digital twin"
    )
    
    # Step 4: Generate report
    return {
        "equipment": equipment_docs,
        "data_sources": data_tables,
        "requirements": dt_requirements,
        "gaps": identify_gaps(equipment_docs, data_tables, dt_requirements)
    }
```

### 2. Maintenance Intelligence System

```python
async def maintenance_intelligence(asset_id: str):
    """Provide intelligent maintenance recommendations."""
    
    # Get equipment specs and history
    specs = await equipment_agent.chat(
        f"Get specifications for asset {asset_id}"
    )
    
    # Query operational data
    operational_data = await bigquery_agent.chat(f"""
        Execute query:
        SELECT * FROM sensor_data.{asset_id}_metrics
        WHERE timestamp > CURRENT_DATE - 30
    """)
    
    # Analyze for digital twin insights
    insights = await digital_twin_agent.chat(
        f"Based on this operational data, what maintenance is recommended?"
    )
    
    return combine_intelligence(specs, operational_data, insights)
```

### 3. Compliance Documentation

```python
async def generate_compliance_report(regulation: str):
    """Generate compliance documentation."""
    
    # Get required equipment docs
    required_docs = await equipment_agent.chat(
        f"What documentation is required for {regulation} compliance?"
    )
    
    # Verify data retention
    data_retention = await bigquery_agent.chat(
        "Show data retention policies for all datasets"
    )
    
    # Digital twin compliance
    dt_compliance = await digital_twin_agent.chat(
        f"How do digital twins support {regulation} compliance?"
    )
    
    return create_compliance_report(required_docs, data_retention, dt_compliance)
```

## Best Practices

### 1. Error Handling

```python
try:
    result = await agent.chat("Your query")
except ConnectionError:
    # Handle connection issues
    logger.error("Failed to connect to external system")
except Exception as e:
    # General error handling
    logger.error(f"Unexpected error: {e}")
```

### 2. Batch Operations

```python
# Process multiple queries efficiently
queries = ["query1", "query2", "query3"]
results = await asyncio.gather(*[
    agent.chat(query) for query in queries
])
```

### 3. Caching Strategies

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_equipment_specs(equipment_id: str):
    """Cache equipment specifications."""
    return agent.chat_sync(f"Get specs for {equipment_id}")
```

### 4. Security Considerations

- Use service accounts with minimal permissions
- Rotate credentials regularly
- Audit access logs
- Implement row-level security in BigQuery
- Use SharePoint app-only authentication

## Troubleshooting

### Common Issues

1. **BigQuery Permission Errors**
   ```
   Solution: Ensure service account has BigQuery Data Viewer role
   ```

2. **SharePoint Authentication Failures**
   ```
   Solution: Verify app registration and API permissions
   ```

3. **Memory Issues with Large Datasets**
   ```
   Solution: Use query pagination and streaming
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

## Future Enhancements

1. **Real-time Data Streaming**
   - Integration with Pub/Sub
   - Live digital twin updates
   - Event-driven architectures

2. **ML Model Integration**
   - Predictive maintenance models
   - Anomaly detection
   - Performance optimization

3. **Extended Platform Support**
   - SAP integration
   - SCADA systems
   - IoT platforms

## Contributing

See main repository contributing guidelines. Ensure all agents follow the Google ADK pattern.

## License

[Specify license]

## Support

- GitHub Issues: [repository-url]/issues
- Documentation: [docs-url]
- Email: industrial-support@example.com