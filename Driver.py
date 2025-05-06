# Databricks notebook source
# MAGIC %md
# MAGIC # Procurement Agent - Model Training and Deployment
# MAGIC 
# MAGIC This notebook creates, trains, and deploys an agent for procurement analytics.

# COMMAND ----------

# MAGIC %pip install -U langgraph langchain langchain_experimental databricks-sdk mlflow databricks-langchain langchain-openai graphviz networkx langchain-openai databricks-agents uv
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Define configuration variables
CATALOG = "sl_uc_master_tda_admin"
SCHEMA = "pi_spend"
MODEL_NAME = "procurement_agent_pj"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Define the Python model implementation class name
PYTHON_MODEL_IMPLEMENTATION = "agent"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Dependencies

# COMMAND ----------

import mlflow
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksTable,
    DatabricksVectorSearchIndex,
    DatabricksSQLWarehouse,
    DatabricksGenieSpace,
    DatabricksUCConnection
)
from databricks import agents

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resource Configuration

# COMMAND ----------

def get_required_resources():
    """
    Define all resources required by the model.
    """
    resources = [
        DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_material_hierarchy_level"),
        DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_location_hierarchy_level"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_calendar"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_hierarchyglobalproduct"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_hierarchyplant_hg"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_hierarchyprocurementteam"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_hierarchypurchasematerialcommodity"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_hierarchypurchasematerialuse"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_hierarchysupplier"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_invoicespendmonthlyfinishedgoodssplit"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_material"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.vw_materialgroup"),
        DatabricksGenieSpace(genie_space_id="01f018f55a561e5a9afdad51da1e8c3f"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.materialhierarchyuniquevalues"),
        DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.GeographyHierarchyUniqueValues"),
    ]
    
    # Add LLM endpoint if needed
    # if LLM_ENDPOINT_NAME:
    #     resources.append(DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME))
    
    return resources

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Example Input

# COMMAND ----------

def create_input_example(query="What is the total spend of tea in 2023?", thread_id=7):
    """
    Create an example input for the model.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "custom_inputs": {
            "thread_id": thread_id
        }
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register Model

# COMMAND ----------

def log_and_register_model():
    """
    Log the model to MLflow and register it to Unity Catalog.
    """
    # Create example input
    input_example = create_input_example()
    
    # Log model to MLflow
    with mlflow.start_run():
        logged_agent_info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=PYTHON_MODEL_IMPLEMENTATION,
            input_example=input_example,
            resources=get_required_resources(),
            pip_requirements=[
                "mlflow",
                "langchain",
                "langgraph",
                "databricks-langchain",
                "langchain-openai",
                "unitycatalog-langchain[databricks]",
                "pydantic",
            ],
        )
    
    # Set registry URI to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # Register the model
    uc_registered_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, 
        name=UC_MODEL_NAME
    )
    
    return logged_agent_info, uc_registered_model_info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Model Predictions

# COMMAND ----------

def test_model_prediction(model_uri, query, thread_id=None, state=None):
    """
    Test model prediction with the given query.
    
    Args:
        model_uri (str): The model URI to load
        query (str): The query to send to the model
        thread_id (int, optional): Thread ID for the conversation
        state (dict, optional): Current state of the conversation
        
    Returns:
        The model response
    """
    input_data = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "custom_inputs": {
            "thread_id": thread_id if thread_id else 1234
        }
    }
    
    if state:
        input_data["custom_inputs"]["state"] = state
    
    return mlflow.models.predict(
        model_uri=model_uri,
        input_data=input_data,
    )

# COMMAND ----------

def continue_conversation(model, response, query, thread_id=None):
    """
    Continue the conversation with the model.
    
    Args:
        model: The loaded model
        response (dict): The previous model response
        query (str): The new user query
        thread_id (int, optional): Thread ID for the conversation
        
    Returns:
        The model response
    """
    messages = response['messages'] + [{'role': 'user', 'content': query}]
    
    input_data = {
        "messages": messages,
        "custom_inputs": {"thread_id": thread_id if thread_id else 1234}
    }
    
    if 'state' in response.get('custom_inputs', {}):
        input_data["custom_inputs"]["state"] = response['custom_inputs']['state']
    
    return model.predict(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model

# COMMAND ----------

def deploy_model(model_name, version):
    """
    Deploy the model to the review app and a model serving endpoint.
    
    Args:
        model_name (str): The fully qualified model name
        version (str): The model version to deploy
    """
    agents.deploy(
        model_name,
        version,
        tags={"Project": "ProcurementGPT"}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Execution

# COMMAND ----------

# Log and register the model
logged_agent_info, uc_registered_model_info = log_and_register_model()

# COMMAND ----------

# Test basic prediction
test_result = test_model_prediction(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    query="What is the total spend of tea in 2023?"
)
test_result

# COMMAND ----------

# Load the model for interactive testing
pyfunc_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{logged_agent_info.run_id}/agent")

# COMMAND ----------

# Example interactive testing session for tea in India
response = pyfunc_model.predict({
    "messages": [
        {
            "role": "user",
            "content": "What is the total spend of tea in india 2023?"
        }
    ],
    "custom_inputs": {"thread_id": 1231}
})
response

# COMMAND ----------

# Continue conversation by specifying hierarchy
response = continue_conversation(
    model=pyfunc_model,
    response=response,
    query="network name", 
    thread_id=1231
)
response

# COMMAND ----------

# Continue conversation by specifying location hierarchy
response = continue_conversation(
    model=pyfunc_model,
    response=response,
    query="ISOCountryName",
    thread_id=1231
)
response

# COMMAND ----------

# Start a new conversation about tomatoes
response = pyfunc_model.predict({
    "messages": [
        {
            "role": "user",
            "content": "What is the total spend of tomato in india 2023?"
        }
    ],
    "custom_inputs": {"thread_id": 1234535353}
})
response

# COMMAND ----------

# Continue by specifying material hierarchy
response = continue_conversation(
    model=pyfunc_model,
    response=response,
    query="Purchase class",
    thread_id=1234535353
)
response

# COMMAND ----------

# Deploy the model to a serving endpoint
deploy_model(UC_MODEL_NAME, uc_registered_model_info.version)

# COMMAND ----------

