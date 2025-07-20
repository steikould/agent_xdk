# Google ADK Agents Collection

This directory contains a collection of specialized agents built using Google's Agent Development Kit (ADK). These agents provide integrations with various enterprise systems and development tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Available Agents](#available-agents)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Contributing](#contributing)

## Overview

Each agent in this collection is designed to integrate with specific external systems while leveraging Google's Gemini models through the ADK framework. All agents follow consistent patterns for:

- 🔧 Tool-based functionality
- 🔄 Async/sync operation support
- 📝 Rich formatted responses
- 🚨 Comprehensive error handling
- 💬 Natural language interfaces

## Available Agents

### 1. Azure DevOps Agent 📘
**Purpose**: Manage work items in Azure DevOps projects

**Features**:
- Create work items (User Stories, Tasks, Bugs, Epics)
- Search and retrieve work items
- Update work item fields
- Query using WIQL
- Get project information

**Location**: `azure-devops-agent/`

### 2. Jira Agent 🎫
**Purpose**: Manage issues in Jira projects

**Features**:
- Create issues with various types
- Search issues using JQL
- Update issue fields and status
- Add comments to issues
- Get project metadata

**Location**: `jira-agent/`

### 3. SharePoint Agent 📁
**Purpose**: Access and manage SharePoint documents

**Features**:
- Read files from libraries
- List library contents
- Search for documents
- Analyze file content
- Get library information

**Location**: `sharepoint-agent/`

### 4. Console LLM 💻
**Purpose**: Interactive console application with SharePoint integration

**Features**:
- Interactive chat interface
- SharePoint document access
- General AI assistance
- Batch command processing
- Context-aware responses

**Location**: `console-llm/`

## Installation

### Prerequisites

- Python 3.8+
- Google ADK installed
- API credentials for target systems

### General Installation

```bash
# Clone the repository
git clone <repository-url>
cd google-adk-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base requirements
pip install google-adk
```

### Agent-Specific Installation

Each agent has its own requirements:

```bash
# Azure DevOps Agent
cd azure-devops-agent
pip install -r requirements.txt

# Jira Agent
cd jira-agent
pip install -r requirements.txt

# SharePoint Agent
cd sharepoint-agent
pip install -r requirements.txt

# Console LLM
cd console-llm
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Each agent requires specific environment variables:

#### Azure DevOps Agent
```bash
export AZURE_DEVOPS_ORG_URL="https://dev.azure.com/yourorg"
export AZURE_DEVOPS_PROJECT_NAME="YourProject"
export AZURE_DEVOPS_PAT="your-personal-access-token"
```

#### Jira Agent
```bash
export JIRA_SERVER_URL="https://yourcompany.atlassian.net"
export JIRA_USERNAME="your-email@company.com"
export JIRA_API_TOKEN="your-api-token"
export JIRA_PROJECT_KEY="PROJ"
```

#### SharePoint Agent
```bash
export SHAREPOINT_SITE_URL="https://yourcompany.sharepoint.com/sites/yoursite"
export SHAREPOINT_CLIENT_ID="your-app-client-id"
export SHAREPOINT_CLIENT_SECRET="your-app-client-secret"
```

## Usage Examples

### Azure DevOps Agent

```python
from azure_devops_agent.agent import AzureDevOpsAgent

# Initialize agent
agent = AzureDevOpsAgent(
    org_url="https://dev.azure.com/yourorg",
    project_name="YourProject",
    personal_access_token="your-pat"
)

# Use the agent
response = agent.chat_sync("Create a user story for implementing user authentication")
print(response)

# Direct tool usage
response = agent.chat_sync("Search for all active bugs")
print(response)
```

### Jira Agent

```python
from jira_agent.agent import JiraAgent

# Initialize agent
agent = JiraAgent(
    server_url="https://yourcompany.atlassian.net",
    username="your-email@company.com",
    api_token="your-token",
    project_key="PROJ"
)

# Create an issue
response = agent.chat_sync("Create a bug report for login page not loading on mobile")
print(response)

# Search issues
response = agent.chat_sync("Find all high priority issues assigned to me")
print(response)
```

### SharePoint Agent

```python
from sharepoint_agent.agent import SharePointAgent

# Initialize agent
agent = SharePointAgent(
    site_url="https://yourcompany.sharepoint.com/sites/yoursite",
    client_id="your-client-id",
    client_secret="your-client-secret"
)

# List files
response = agent.chat_sync("List all files in the 'Shared Documents' library")
print(response)

# Read specific document
response = agent.chat_sync("Read the file 'project-plan.docx' from 'Documents'")
print(response)
```

### Console LLM

```python
from console_llm.llm import ConsoleLLM

# Initialize and run interactive session
console = ConsoleLLM(model_name="gemini-2.0-flash-exp")
console.start_chat()

# Or use programmatically
response = console.chat_sync("Connect to SharePoint and list available documents")
print(response)
```

## Architecture

### Common Pattern

All agents follow this architectural pattern:

```
┌─────────────────────────────────────────┐
│          External System API            │
│    (Azure DevOps/Jira/SharePoint)       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         System-Specific Client          │
│    (Native Python SDK/REST Client)      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           Agent Class                   │
│  • Connection Management                │
│  • Tool Definitions                     │
│  • Error Handling                      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         Google ADK Agent                │
│  • Natural Language Processing         │
│  • Tool Orchestration                  │
│  • Response Generation                 │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      Session & Runner (ADK)             │
│  • State Management                     │
│  • Async Execution                     │
│  • Event Handling                      │
└─────────────────────────────────────────┘
```

### Key Components

1. **Agent Class**: Wraps system-specific functionality
2. **Tool Functions**: Define available operations
3. **ADK Agent**: Provides NLP interface
4. **Session Management**: Handles conversation state
5. **Error Handling**: Graceful failure management

## Best Practices

### 1. Tool Design
```python
async def tool_name(param1: str, param2: str = "default") -> str:
    """
    Clear description of what the tool does.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
        
    Returns:
        str: Formatted response
    """
    # Implementation
```

### 2. Error Handling
```python
try:
    result = perform_operation()
    return f"✅ Success: {result}"
except SpecificException as e:
    return f"❌ Error: {e}"
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return f"❌ Unexpected error occurred"
```

### 3. Response Formatting
- Use emojis for visual clarity
- Structure responses with headers and sections
- Include actionable information
- Provide helpful error messages

## Testing

Each agent includes test files:

```bash
# Run tests for specific agent
cd azure-devops-agent
python -m pytest test_agent.py

# Run all tests
python -m pytest google-adk-agents/
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify environment variables are set
   - Check API token/credential validity
   - Ensure proper permissions

2. **Connection Errors**
   - Verify network connectivity
   - Check firewall settings
   - Validate URLs and endpoints

3. **ADK Import Errors**
   - Ensure google-adk is installed
   - Check Python version compatibility
   - Verify virtual environment activation

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

## Contributing

### Adding New Agents

1. Create new directory: `new-system-agent/`
2. Follow the established pattern:
   ```
   new-system-agent/
   ├── __init__.py
   ├── agent.py
   ├── requirements.txt
   ├── test_agent.py
   └── README.md
   ```

3. Implement required methods:
   - `__init__`: Initialize connections
   - `_create_agent`: Define ADK agent with tools
   - `chat`/`chat_sync`: Provide chat interface

4. Add comprehensive documentation
5. Include unit tests
6. Update this README

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Include error handling
- Format with black/isort

## License

[Specify license]

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [docs-url]
- Email: support@example.com