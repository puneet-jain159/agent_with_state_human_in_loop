# Define configuration variables
CATALOG = "malay_demo"
SCHEMA = "procurement"
MODEL_NAME = "procurement_bot_v3"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"



# Define the Python model implementation class name
PYTHON_MODEL_IMPLEMENTATION = "agent_build_v2/mlflowlogger.py"

import os
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
from dotenv import load_dotenv


# Load environment variables
load_dotenv(".env")

def get_required_resources():
    """
    Define all resources required by the model.
    """
    resources = [
      DatabricksServingEndpoint(endpoint_name="databricks-claude-3-7-sonnet"),
      DatabricksGenieSpace(genie_space_id="01f0421695fe1fb694762b30f68f799d"),
      DatabricksGenieSpace(genie_space_id="01f042195bd1169e8f1cb3e85a85f7e7"),
      DatabricksTable(table_name="malay_demo.procurement.calendar"),
      DatabricksTable(table_name="malay_demo.procurement.capex_invoice"),
      DatabricksTable(table_name="malay_demo.procurement.department"),
      DatabricksTable(table_name="malay_demo.procurement.materials"),
      DatabricksTable(table_name="malay_demo.procurement.p2p_invoice"),
      DatabricksTable(table_name="malay_demo.procurement.plants"),
      DatabricksTable(table_name="malay_demo.procurement.suppliers"),
      DatabricksFunction(function_name="malay_demo.procurement.get_location_hierarchy_level"),
      DatabricksFunction(function_name="malay_demo.procurement.get_material_hierarchy_level"),
      DatabricksSQLWarehouse(warehouse_id="148ccb90800933a1")
    ]
    
    # Add LLM endpoint if needed
    # if LLM_ENDPOINT_NAME:
    #     resources.append(DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME))
    
    return resources

def create_input_example(
    query="What is the total spend of tea in 2023?",
    thread_id=None,
    state=None
):
    """
    Create an example input for the model.

    Matches ChatAgentRequest schema exactly.
    """
    example = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "custom_inputs": {
            "thread_id": thread_id or "1234"
        }
    }

    if state:
        example["custom_inputs"]["state"] = state

    return example



def log_and_register_model():
    """
    Log the model to MLflow and register it to Unity Catalog.
    """
    # Create example input
    input_example = create_input_example()
    
    # Log model to MLflow
    with mlflow.start_run():
        logged_agent_info = mlflow.pyfunc.log_model(
            name="procurement_agent",
            python_model=PYTHON_MODEL_IMPLEMENTATION,
            input_example=input_example,
            resources=get_required_resources(),
            code_paths=["agent_build","agent_build_v2","chat_database.py"]
        )
    
    # Set registry URI to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # Register the model
    uc_registered_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, 
        name=UC_MODEL_NAME
    )
    
    return logged_agent_info, uc_registered_model_info



def test_model_prediction(model_uri, query, thread_id=None, state=None):
    """
    Test model prediction with the given query.
    """
    input_data = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "custom_inputs": {
            "thread_id": str(thread_id or "1234")
        }
    }

    if state is not None:
        input_data["custom_inputs"]["state"] = state

    return mlflow.models.predict(
        model_uri=model_uri,
        input_data=input_data,
        env_manager="uv",
        extra_envs={"DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"), 
                    "CLIENT_ID": os.getenv("CLIENT_ID"), 
                    "CLIENT_SECRET": os.getenv("CLIENT_SECRET")}
    )


def continue_conversation(model, response, query, thread_id=None):
    """
    Continue the conversation with the model.
    """
    # Ensure response['messages'] is a list of dicts
    messages = response["messages"] + [
        {
            "role": "user",
            "content": query
        }
    ]

    input_data = {
        "messages": messages,
        "custom_inputs": {
            "thread_id": str(thread_id or "1234")
        }
    }

    # Carry over state if available
    if "custom_inputs" in response and "state" in response["custom_inputs"]:
        input_data["custom_inputs"]["state"] = response["custom_inputs"]["state"]

    return model.predict(input_data)

extra_envs={"DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"), 
                    "CLIENT_ID": os.getenv("CLIENT_ID"), 
                    "CLIENT_SECRET": os.getenv("CLIENT_SECRET"),
                    "DB_INSTANCE_NAME": os.getenv("DB_INSTANCE_NAME")}


from databricks import agents

# Deploy the model to the review app and a model serving endpoint
# agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"project": "tech_summit25"})


if __name__ == "__main__":
    logged_agent_info, uc_registered_model_info = log_and_register_model()
    print(logged_agent_info)
    print(uc_registered_model_info)

    # Test basic prediction
    # test_result = test_model_prediction(
    #     model_uri=f"models:/malay_demo.procurement.procurement_bot_v3/4",
    #     query="What is the total spend of tea in 2023?"
    # )
    # test_result

    agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"project": "tech_summit25"},environment_vars=extra_envs)