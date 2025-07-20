# DRA Power Analysis System

## Overview

The DRA (Drag Reducing Agent) Power Analysis System is a sophisticated multi-agent orchestration platform built using Google's Agent Development Kit (ADK). It analyzes pump power consumption in energy pipelines, providing insights for optimization and predictive maintenance.

## Table of Contents

- [System Architecture](#system-architecture)
- [Agent Components](#agent-components)
- [Data Flow](#data-flow)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                    (Console/Web/API)                            │
└───────────────────┬─────────────────────────┬───────────────────┘
                    │                         │
                    ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Root Orchestrator Agent                        │
│                 (PowerAnalysisInitiator)                        │
│  • Coordinates sub-agents                                       │
│  • Manages session state                                        │
│  • Routes tasks to appropriate agents                          │
└───────────────────┬─────────────────────────┬───────────────────┘
                    │                         │
    ┌───────────────┴───────────────┬─────────┴────────────┐
    ▼                               ▼                       ▼
┌──────────────┐          ┌──────────────────┐   ┌─────────────────┐
│   BigQuery   │          │  Data Quality    │   │ Power Consumption│
│Data Retrieval│          │  Validation      │   │  Calculation     │
│    Agent     │          │     Agent        │   │     Agent        │
└──────────────┘          └──────────────────┘   └─────────────────┘
        │                           │                       │
        │                           ▼                       │
        │                 ┌──────────────────┐            │
        │                 │   Statistical    │            │
        │                 │    Analysis      │◄───────────┘
        │                 │     Agent        │
        │                 └──────────────────┘
        │                           │
        │                           ▼
        │                 ┌──────────────────┐
        │                 │    Insights &    │
        │                 │  Optimization    │
        │                 │     Agent        │
        │                 └──────────────────┘
        │                           │
        └───────────────────────────┴───────────────────────┐
                                                            ▼
                                                ┌──────────────────┐
                                                │  User Interface  │
                                                │  Output Agent    │
                                                └──────────────────┘
```

### Component Details

#### 1. **Root Orchestrator Agent** (`agent.py`)
- **Role**: Central coordinator
- **Responsibilities**:
  - Receives user requests
  - Delegates tasks to sub-agents
  - Manages workflow state
  - Handles error escalation

#### 2. **BigQuery Data Retrieval Agent** (`sub_agents/bigquery_data_retrieval/agent.py`)
- **Role**: Data access layer
- **Responsibilities**:
  - Queries historical sensor data
  - Handles time-series data retrieval
  - Manages BigQuery connections
  - Parameterized SQL execution

#### 3. **Data Quality Validation Agent** (`sub_agents/data_quality/agent.py`)
- **Role**: Data integrity assurance
- **Responsibilities**:
  - Validates sensor data completeness
  - Detects anomalies and outliers
  - Checks timestamp consistency
  - Identifies data gaps

#### 4. **Power Consumption Calculation Agent** (`sub_agents/power_consumption/agent.py`)
- **Role**: Engineering calculations
- **Responsibilities**:
  - Calculates hydraulic power
  - Computes electrical power consumption
  - Applies pump-specific efficiencies
  - Generates energy metrics

#### 5. **Statistical Analysis Agent** (`sub_agents/statistical_analysis/agent.py`)
- **Role**: Advanced analytics
- **Responsibilities**:
  - Time-series aggregations
  - Correlation analysis
  - Efficiency trend analysis
  - Statistical summaries

#### 6. **Insights & Optimization Agent** (`sub_agents/insights_optimization/agent.py`)
- **Role**: Decision support
- **Responsibilities**:
  - Generates actionable recommendations
  - Identifies optimization opportunities
  - Creates executive summaries
  - Risk assessment

#### 7. **User Interface Agent** (`sub_agents/user_interface/agent.py`)
- **Role**: User interaction
- **Responsibilities**:
  - Collects and validates user inp