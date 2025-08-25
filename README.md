---
title: "Agent with State and Human-in-the-Loop"
language: python
author: "Puneet Jain"
date: 2025-01-07
---
# Agent with State and Human-in-the-Loop

A Databricks-based procurement analytics agent that uses a multi-agent architecture with LangGraph to provide insights from procurement data. This agent maintains state throughout conversations and incorporates human-in-the-loop functionality for resolving ambiguities.

## Overview

This procurement agent uses a graph-based approach to analyze spend data across material and location hierarchies. It extracts key information from user queries, resolves ambiguities through human interaction when necessary, and generates procurement insights using Databricks Genie.

## Architecture

The solution follows a multi-agent architecture with comprehensive observability:

1. **Material Hierarchy Resolver**: Extracts material information from user queries
2. **Location Hierarchy Resolver**: Extracts location information from user queries
3. **Genie Agent**: Processes queries using specialized procurement knowledge
4. **Summary Agent**: Formats and presents findings in a business-friendly way
5. **Supervisor Agent**: Coordinates workflow between agents

All agents work together in a LangGraph workflow that maintains conversational state and provides detailed tracing through MLflow 3.0 integration.

## Features

- 🚀 **Multi-Agent Architecture**: Coordinated workflow using LangGraph
- 💾 **State Persistence**: Maintains conversation context across interactions
- 🔄 **Human-in-the-Loop**: Resolves hierarchy ambiguities through user interaction
- ⚡ **Real-time Processing**: Fast response times with streaming support
- 🔒 **Databricks Integration**: Leverages Unity Catalog and Genie for data access
- 🎯 **Procurement Focus**: Specialized for spend analysis and procurement insights
- 📊 **Hierarchy Resolution**: Automatically resolves material and location hierarchies
- 🔍 **MLflow 3.0 Tracing**: Comprehensive observability and experiment tracking
- 🏗️ **Lakebase Integration**: Seamless deployment to Databricks Lakehouse Apps

## Project Structure

```
├── agent_build/           # Multi-agent implementation
│   ├── utils.py          # Utility functions and UC function calls
│   ├── mlflowlogger.py   # MLflow integration for model logging
│   ├── deploy_model.py   # Deply model to Endpoint
│   └── run_toolnode.py   # Tool execution framework
├── frontend/             # React-based chat interface
├── utils/                # Backend utilities and database handling
├── main.py               # FastAPI application with chat endpoints
├── run_agent_v2.py      # Agent execution and conversation management
├── deploy.sh            # Deployment script for Databricks
├── Makefile             # Build and deployment automation
└── requirements.txt     # Python dependencies
```

## Setup Requirements

- **Databricks Workspace** with Unity Catalog enabled
- **Claude Sonnet** or equivalent LLM service access
- **Python 3.10+** with virtual environment support
- **Required Libraries**: langchain, langgraph, databricks-langchain, unitycatalog-ai, mlflow[databricks]
- **Configured Databricks Genie** space with procurement data
- **MLflow 3.0** for experiment tracking and model management

## Quick Setup

### 1. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Configure your .env file with:
# - DATABRICKS_HOST: Your Databricks workspace URL
# - CLIENT_ID: OAuth client ID for authentication
# - CLIENT_SECRET: OAuth client secret
# - LAKEHOUSE_APP_NAME: Your app name for deployment
# - APP_FOLDER_IN_WORKSPACE: Target folder in Databricks
# - MLFLOW_EXPERIMENT_ID: Your MLflow experiment ID for tracing
```

### 2. Install Dependencies

```bash
# Complete setup (recommended)
make setup

# Or manual installation
make install-backend
make install-frontend
```

### 3. Database Setup

The application uses PostgreSQL via Databricks Database for production and SQLite for local development. For production deployment, you'll need to set up the database schema:

```bash
# Setup database schema with proper roles and permissions
make db-schema-setup

# Or run manually:
python utils/setup_database_schema.py
```

**Required Environment Variables for Database:**
```bash
# Database Configuration
DB_INSTANCE_NAME=your-database-instance-name
CLIENT_ID=your-oauth-client-id
CLIENT_SECRET=your-oauth-client-secret
DATABRICKS_HOST=your-workspace-url
```

**What the Database Setup Script Does:**
1. **Creates `chatbot_schema`**: Dedicated schema for the application
2. **Sets up `chatbot_app` role**: Non-login role for application operations
3. **Configures permissions**: Grants necessary access to the schema
4. **Verifies setup**: Confirms all components are properly configured

### 4. Run the Application

```bash
# Development mode
make dev

# Production mode
make run

# Deploy to Databricks
make deploy
```

## Database Configuration

### PostgreSQL Setup (Production)

The application uses Databricks Database (PostgreSQL) for production deployments. The `setup_database_schema.py` script automates the database setup process:

#### **Prerequisites**
- Databricks workspace with Database instances enabled
- OAuth app configured with proper permissions
- Database instance created and accessible

#### **Environment Variables**
```bash
# Required for database setup
DB_INSTANCE_NAME=your-database-instance-name
CLIENT_ID=your-oauth-client-id
CLIENT_SECRET=your-oauth-client-secret
DATABRICKS_HOST=your-workspace-url

# Optional: Database user (defaults to OAuth client)
DB_USERNAME=your-username@databricks.com
```

#### **Automatic Setup**
```bash
# Use Makefile (recommended)
make db-schema-setup

# Or run manually
python utils/setup_database_schema.py
```

#### **Manual Setup Steps**
If you prefer to set up the database manually:

1. **Connect to Database Instance**:
   ```bash
   # Use Databricks CLI or workspace UI
   databricks database get --name your-instance-name
   ```

2. **Create Schema**:
   ```sql
   CREATE SCHEMA IF NOT EXISTS chatbot_schema;
   ```

3. **Create Application Role**:
   ```sql
   CREATE ROLE chatbot_app NOLOGIN;
   GRANT chatbot_app TO "your-client-id";
   GRANT USAGE, CREATE ON SCHEMA chatbot_schema TO chatbot_app;
   ```

#### **Verification**
The setup script automatically verifies the configuration:
```bash
# Check if setup was successful
python utils/setup_database_schema.py

# Look for these success messages:
# ✅ Schema created
# ✅ chatbot_app role configured
# ✅ Schema setup completed successfully!
```

### SQLite (Local Development)

For local development, the application automatically falls back to SQLite if PostgreSQL configuration is not provided. No additional setup is required.

## Makefile Commands

The project includes a comprehensive Makefile for easy development:

- `make setup` - Complete project setup (install uv, create venv, install deps)
- `make install` - Install all dependencies
- `make dev` - Run development server with hot reload
- `make run` - Run production server with gunicorn
- `make deploy` - Deploy application to Lakehouse Apps using deploy.sh
- `make db-schema-setup` - Setup database schema and permissions
- `make clean` - Clean up generated files
- `make lint` - Run linting with ruff
- `make format` - Format code with ruff
- `make check-deps` - Check for outdated dependencies

## Usage Examples

### Basic Procurement Queries

The agent can handle complex procurement queries:

- **Spend Analysis**: "What is the total spend of tea in India 2023?"
- **Material Queries**: "What is the spend on tomatoes in 2023?"
- **Location Analysis**: "Show me spend by location for coffee products"
- **Hierarchy Resolution**: Automatically resolves material and location hierarchies

### Human-in-the-Loop Features

When the agent encounters ambiguous hierarchies:

1. **Automatic Detection**: Identifies multiple possible hierarchy levels
2. **User Prompting**: Asks for clarification on specific hierarchies
3. **State Persistence**: Maintains context throughout the resolution process
4. **Seamless Continuation**: Continues processing after resolution

## Deployment

### Local Development

```bash
# Start development server
make dev

# Access at http://localhost:8000
```

### Databricks Deployment

```bash
# Deploy to Databricks workspace
make deploy

# The deploy.sh script handles:
# - Environment variable injection
# - App configuration
# - Databricks app deployment
# - MLflow experiment setup
```

### MLflow Model Registration

The agent can be deployed as:

1. **MLflow Model**: Registered to Unity Catalog with full resource tracking
2. **Model Serving Endpoint**: For production inference with monitoring
3. **Databricks Agent**: In the Agents review app with tracing
4. **Lakehouse App**: Direct deployment with automatic resource management

## Key Components

### Backend (FastAPI)

- **Multi-Agent Workflow**: LangGraph-based agent coordination
- **State Management**: Persistent conversation context
- **Database Integration**: PostgreSQL via Databricks Database
- **Authentication**: Databricks OAuth integration

### Frontend (React)

- **Chat Interface**: Real-time messaging with streaming
- **State Persistence**: Maintains chat history
- **Responsive Design**: Works across devices
- **Modern UI**: Clean, intuitive interface

### Agent Framework

- **LangGraph Integration**: Workflow orchestration
- **UC Function Calls**: Databricks Unity Catalog integration
- **Error Handling**: Graceful fallbacks and user guidance
- **Performance Optimization**: Efficient resource usage

### MLflow Integration

- **LangGraphChatAgent**: MLflow ChatAgent API adapter for LangGraph workflows
- **Automatic Tracing**: Complete workflow execution visibility
- **Resource Tracking**: Monitors all Databricks dependencies
- **Model Registry**: Unity Catalog integration with versioning

## Development

### Adding New Agents

1. Create agent logic in `agent_build/`
2. Register with the supervisor in the workflow
3. Update state management for new agent outputs
4. Test with sample queries

### Extending UC Functions

1. Create new functions in Databricks Unity Catalog
2. Add function calls in `agent_build/utils.py`
3. Update agent prompts to use new capabilities
4. Test integration end-to-end

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all dependencies are installed
2. **Databricks Connection**: Verify credentials and workspace access
3. **UC Function Access**: Check permissions for Unity Catalog functions
4. **State Persistence**: Verify database connectivity

### Database Setup Issues

#### **Connection Errors**
```bash
# Check if environment variables are set
echo $DB_INSTANCE_NAME
echo $CLIENT_ID
echo $CLIENT_SECRET
echo $DATABRICKS_HOST

# Verify Databricks connection
databricks workspace list
```

#### **Permission Errors**
If you encounter permission issues during schema setup:

1. **Check OAuth App Permissions**:
   - Ensure your OAuth app has `workspace:read` and `database:write` permissions
   - Verify the app is properly configured in your Databricks workspace

2. **Database Instance Access**:
   - Confirm your user has access to the database instance
   - Check if the instance is running and accessible

3. **Role Creation Issues**:
   ```sql
   -- Check if role already exists
   SELECT rolname FROM pg_roles WHERE rolname = 'chatbot_app';
   
   -- Check current user permissions
   SELECT current_user, session_user;
   ```

#### **Schema Setup Failures**
```bash
# Run with verbose output
python -u utils/setup_database_schema.py

# Check database logs in Databricks workspace
# Navigate to: SQL Warehouses > Your Instance > Logs
```

#### **Common Error Messages and Solutions**

| Error | Solution |
|-------|----------|
| `CLIENT_ID and DB_INSTANCE_NAME must be set` | Add missing environment variables to `.env` |
| `Failed to create database connection` | Verify Databricks credentials and network access |
| `Permission denied` | Check OAuth app permissions and database access |
| `Role already exists` | This is normal - the script handles existing roles gracefully |

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
make dev
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Insert your license information here]

## Support

For issues and questions:

- Check the troubleshooting section
- Review Databricks documentation
- Open an issue in the repository
