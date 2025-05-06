# Agent with State and Human-in-the-Loop

A Databricks-based procurement analytics agent that uses a multi-agent architecture with LangGraph to provide insights from procurement data. This agent maintains state throughout conversations and incorporates human-in-the-loop functionality for resolving ambiguities.

## Overview

This procurement agent uses a graph-based approach to analyze spend data across material and location hierarchies. It extracts key information from user queries, resolves ambiguities through human interaction when necessary, and generates procurement insights using Databricks Genie.

## Architecture

The solution follows a multi-agent architecture:

1. **Material Hierarchy Resolver**: Extracts material information from user queries
2. **Location Hierarchy Resolver**: Extracts location information from user queries
3. **Genie Agent**: Processes queries using specialized procurement knowledge
4. **Summary Agent**: Formats and presents findings in a business-friendly way
5. **Supervisor Agent**: Coordinates workflow between agents

All agents work together in a LangGraph workflow that maintains conversational state.

## Setup Requirements

- Databricks workspace with Unity Catalog
- Access to Azure OpenAI or equivalent LLM service
- Required libraries: langchain, langgraph, databricks-langchain, unitycatalog-ai
- Configured Databricks Genie space with procurement data

## Key Features

- Maintains conversation state across multiple interactions
- Resolves hierarchy ambiguities through human-in-the-loop interactions
- Leverages Databricks UC functions for data validation
- Processes complex procurement queries with appropriate context
- Formats responses in business-friendly language

## Usage

This agent can be deployed as:

1. An MLflow model registered to Unity Catalog
2. A model serving endpoint
3. A Databricks agent in the Agents review app

Example queries:
- "What is the total spend of tea in India 2023?"
- "What is the spend on tomatoes in 2023?"

## Development

The codebase consists of:

- `Driver.py`: Handles model training, evaluation, and deployment
- `agent.py`: Contains the multi-agent implementation with LangGraph
