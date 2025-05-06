# Databricks notebook source
# MAGIC %md
# MAGIC #  Multi-agent with Genie

# COMMAND ----------

# MAGIC %md
# MAGIC Install required libraries

# COMMAND ----------

# MAGIC %pip install langchain-core langchain-community langgraph langchain-openai databricks-langchain unitycatalog-ai pyspark mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Note: This notebook uses Databricks-specific packages like `unitycatalog.ai`, `databricks_langchain.genie`, and other Databricks-specific functionality. This code is designed to run in a Databricks environment with access to Unity Catalog and other proprietary services.

# COMMAND ----------
import uuid
import json
from typing import TypedDict, List, Dict, Any, Optional
from functools import partial # To pass LLM to the node


import openai
import httpx
import requests

import mlflow
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from pyspark.sql.functions import col, lower, collect_set, size

# Langchain imports
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import ChatDatabricks

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph

# Databricks-specific imports
from databricks_langchain.genie import GenieAgent
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
client = DatabricksFunctionClient()


mlflow.langchain.autolog()

# COMMAND ----------
def get_access_token(url, client_id, client_secret, scope):
    response = requests.post(
        url,
        data={"grant_type": "client_credentials", "scope": scope},
        auth=(client_id, client_secret),
        timeout=30  # Add timeout in seconds
    )
    return response.json()['access_token']


 
token = partial(get_access_token,
                "https://login.microsoftonline.com/YOUR_TENANT_ID/oauth2/v2.0/token", 
                'YOUR_CLIENT_ID',
                'YOUR_CLIENT_SECRET',
                "YOUR_SCOPE")
 

def update_base_url(request: httpx.Request) -> None:
    if "/chat/completions" in request.url.path:
      request.url = request.url.copy_with(path=f"/openai4/az_openai_gpt-4o_chat")

def lang_chain_run_test(url_endpoint):
    endpoint = '/openai4/az_openai_gpt-4o_chat'
    openai.api_base=f"https://YOUR_AZURE_ENDPOINT{endpoint}"
    
    llm = AzureChatOpenAI( 
        azure_endpoint="https://YOUR_AZURE_ENDPOINT/openai4/az_openai_gpt-4o_chat",
        temperature=0.3,
        api_version="",
        azure_ad_token_provider = token,
        default_headers= {
                'Ocp-Apim-Subscription-Key': "YOUR_SUBSCRIPTION_KEY",
                'Content-Type': 'application/json',
                'Authorization':f"Bearer {token}"
            },
        http_client=httpx.Client(
        event_hooks={
        "request": [update_base_url],
        }),   
)

    return llm

llm = lang_chain_run_test("/openai4/az_openai_gpt-4o_chat")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the State

# COMMAND ----------

# Updated state to include combined_prompt and worker_outputs
class AgentState(TypedDict):
    original_prompt: str
    user_confirmed_hierarchy: Optional[str] # Material Hierarchy confirmed by the user
    extracted_material: Optional[str]
    extracted_location: Optional[str]
    user_confirmed_location_hierarchy: Optional[str] # Location Hierarchy confirmed by the user
    combined_prompt: Optional[str] # Prompt for Genie (original + confirmed hierarchy)
    final_response: Optional[str] # Kept for compatibility, but worker_outputs used now
    messages: List[Dict[str, Any]] # History of interactions
    # Dictionary to store outputs from different workers (like Genie)
    worker_outputs: Optional[Dict[str, str]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Agent Nodes

# COMMAND ----------


# add your genie space id here
genie_space_id = "YOUR_GENIE_SPACE_ID"
spend_genie_agent = GenieAgent(genie_space_id, "Genie", description="This Genie space has access to procurement data like invoice data, material hierarchy, supplier hierarchy data etc")

# COMMAND ----------

def call_get_material_hierarchy_level(material_name):
    toolkit = UCFunctionToolkit(function_names=["sl_uc_master_tda_admin.pi_spend.get_material_hierarchy_level"], client=client)

    tools = toolkit.tools
    python_exec_tool = tools[0]
    result = python_exec_tool.invoke({"material_name": material_name})

    # Parse the JSON string into a Python dictionary
    parsed_result = json.loads(result)

    # Now, parsed_result is a dictionary, and you can access the 'value' key
    csv_content = parsed_result.get("value", "")

    # Split the CSV content into lines and extract the hierarchy label (second line)
    csv_lines = csv_content.splitlines()

    # Extract the hierarchy label (second line)
    hierarchy_label = csv_lines[1] if len(csv_lines) > 1 else None
    return hierarchy_label

# COMMAND ----------


# Define the agent
def material_hierarchy_resolver_agent(state: AgentState, llm_model):
    """
    Extracts a material from the original prompt using the provided LLM.
    """
    print("--- Hierarchy resolver agent (LLM) ---")
    if not llm_model:
        print("LLM not available. Skipping Material extraction.")
        return {"extracted_material": None, "messages": state.get("messages", []) + [{"role": "system", "content": "LLM unavailable for material extraction."}]}

    prompt = state['original_prompt']
    messages = state.get("messages", [])
     # Get existing messages or start new list
    user_confirmed_hierarchy = state.get("user_confirmed_hierarchy", []) # check if heirarchy has been specified
    user_confirmed_material_name = state.get("extracted_material", None) # check if material name has been specified
    print(f"Original user_confirmed_material_name: {user_confirmed_material_name}") # For debugging purpose 
    print(f"Original user_confirmed_hierarchy: {user_confirmed_hierarchy}") # For debugging purpose 
    print(f"Original prompt: {prompt}") # For debugging purpose

    if user_confirmed_hierarchy == []:

        extraction_prompt = f"""
        You are a helpful assistant working with procurement data for a CPG company.

        Each material in the data is tagged using the following hierarchy levels:
        - NetworkName
        - ProcurementTeamName
        - PurchaseFamilyName
        - PurchaseClassName
        - PurchaseCommodityName
        - MaterialGrp (users may mention this hierarchy as "material group")
        - PLM Specification
        - MaterialDescription
        - MaterialNumber
        
        Some examples of materials include: teas, coffees, food ingredients, packaging materials (e.g., Label, Pallet, Display Item, Bottle, Case/Tray, Adhesive and Tape, Liner, Cartridge, Seal, Canister/Tight Head Pail), fragrances, flavors, chemicals (e.g., acids, alcohols, esters, aloxylated compounds), surfactants, soft commodities, etc.

        Your job is to:
        1. Identify the **material name** mentioned in the prompt — usually a noun representing what is being procured.
        2. Identify the **hierarchy level** being used to classify that material — this tells how the material is categorized within the procurement data.
        3. Look for hints such as labels in parentheses (e.g., "Standard Bottle Multi-layer (Purchase Commodity)") and map them to the standard hierarchy name: For example,
            - "Purchase Commodity" or "Commodity" → "PurchaseCommodityName"
            - "Purchase Class" or "Class" or "Class name" etc → "PurchaseClassName"
            - "Purchase Family" → "PurchaseFamilyName"
            - "Material Group" or "MaterialGrp" → "MaterialGrp"
        4. Only extract a hierarchy if it clearly refers to how the material itself is categorized — ignore references to filtering or grouping logic (e.g., "group by" or "filter by").
        5. Do **not infer or assume** hierarchy levels if not explicitly mentioned.

        Output format (strict):
        Material: <material name or None>  
        Hierarchy: <one of: NetworkName, ProcurementTeamName, PurchaseFamilyName, PurchaseClassName, PurchaseCommodityName, MaterialGrp, PLM Specification, MaterialDescription, MaterialNumber, or None>

        Prompt:
        \"{prompt}\"
        """



        print("Calling LLM to extract material and hierarchy...")
        llm_response = llm_model.invoke(extraction_prompt)
        llm_output = llm_response.content.strip()
        print(f"LLM output:{llm_output}") # Debugging

        # Parse LLM output
        extracted_material = None
        extracted_hierarchy = None

        for line in llm_output.splitlines():
            if line.lower().startswith("material:"):
                value = line.split(":", 1)[1].strip()
                extracted_material = value if value.lower() != "none" else None
                print(extracted_material)
            elif line.lower().startswith("hierarchy:"):
                value = line.split(":", 1)[1].strip()
                extracted_hierarchy = value if value.lower() != "none" else None
                print(extracted_hierarchy)
        
        if extracted_material and extracted_hierarchy:
            print(f"Extracted material: {extracted_material}" and f"Extracted hierarchy: {extracted_hierarchy}")
            confirmed_hierarchy = extracted_hierarchy
            combined_prompt = f"Original request: '{prompt}'"
            messages.append({"role": "assistant", "content": f"Extracted material is '{extracted_material}' and the extracted hierarchy is '{extracted_hierarchy}."})
        elif extracted_material and extracted_hierarchy is None:
            print("Checking hierarchy ambiguity using UC function agent...")
            hierarchy_level = call_get_material_hierarchy_level(extracted_material)
            print(f"UC function result: {hierarchy_level}")
            
            if hierarchy_level in ["multiple", "no hierarchy identified"]:
                messages.append({"role": "assistant", "content": f"LLM attempted material extraction: Found '{extracted_material}' is either present in multiple hierarchies or there is no hierarchy identified.Please confirm the material hierarchy level"})
                confirmed_hierarchy = None
                while True:
                    user_input = interrupt(
                        {
                            "messages": messages, 
                            "message": "Please confirm the hierarchy level"
                            }
                    ).strip()
                
                    if confirmed_hierarchy:
                        if isinstance(user_input, str) and user_input.strip():
                            confirmed_hierarchy = user_input
                            print(f"confirmed_hierarchy: {confirmed_hierarchy}")
                            break
                        else:
                            print("Corrected hierarchy cannot be empty. Let's try confirming again.")
                    else:
                        if user_input.lower() == 'skip':
                            confirmed_hierarchy = None # Explicitly no material provided
                            print("Skipping hierarchy confirmation.")
                            break
                        else:
                            confirmed_hierarchy = user_input
                            print(f"Using provided hierarchy: {confirmed_hierarchy}")
                            break
            else:
                confirmed_hierarchy = hierarchy_level
                combined_prompt = f"Original request: '{prompt}'\nHiearchy level for {extracted_material} is: '{confirmed_hierarchy}'"
                messages.append({"role": "system", "content": f"Single hierarchy found: '{hierarchy_level}'."})
        else:
            print("No material or hierarchy found. Skipping material extraction.")
            confirmed_hierarchy = None
            combined_prompt = f"Original request: '{prompt}"
            messages.append({"role": "assistant", "content": "No material or hierarchy found. Skipping material extraction."})
        
    else:
        if user_confirmed_hierarchy == 'skip':
            confirmed_hierarchy = None
            extracted_material = None
            print("user decided to skip hierarchy confirmation.")
        else:
            confirmed_hierarchy = user_confirmed_hierarchy
            if user_confirmed_material_name == []:
                extracted_material = None
            else:
                extracted_material = user_confirmed_material_name

        # Prepare the combined prompt for Genie
        if confirmed_hierarchy:
            combined_prompt = f"Original request: '{prompt}'\nHiearchy level for {extracted_material} is: '{confirmed_hierarchy}'"
            messages.append({"role": "assistant", "content": f"Confirmed hierarchy is '{confirmed_hierarchy}' for the material '{extracted_material}."})
        else:
            combined_prompt = f"Original request: '{prompt}'"
            messages.append({"role": "assistant", "content": "User indicated no specific hierarchy even if there is hierarchy level ambiguity identified"})


    print(f"Final Combined prompt for Genie: {combined_prompt}") # Debugging
    
    state.setdefault("worker_outputs", {})
    state["worker_outputs"]["material_hierarchy_resolver_agent"] = {
        "combined_prompt": combined_prompt
    }

    return {
    "user_confirmed_hierarchy": confirmed_hierarchy,
    "extracted_material": extracted_material,
    "combined_prompt": combined_prompt,
    "messages": messages
    }

# COMMAND ----------

# Databricks-specific function using Unity Catalog
def call_get_location_hierarchy_level(location):
    toolkit = UCFunctionToolkit(function_names=["<Enter your function name here>"], client=client)

    tools = toolkit.tools
    python_exec_tool = tools[0]
    result = python_exec_tool.invoke({"location": location})

    # Parse the JSON string into a Python dictionary
    parsed_result = json.loads(result)

    # Now, parsed_result is a dictionary, and you can access the 'value' key
    csv_content = parsed_result.get("value", "")

    # Split the CSV content into lines and extract the hierarchy label (second line)
    csv_lines = csv_content.splitlines()

    # Extract the hierarchy label (second line)
    hierarchy_label = csv_lines[1] if len(csv_lines) > 1 else None
    return hierarchy_label

# COMMAND ----------

# Define the agent
def location_hierarchy_resolver_agent(state: AgentState, llm_model):
    """
    Extracts a location from the original prompt using the provided LLM.
    """
    print("--- Location Hierarchy resolver agent (LLM) ---")
    if not llm_model:
        print("LLM not available. Skipping Location extraction.")
        return {"extracted_location": None, "messages": state.get("messages", []) + [{"role": "assistant", "content": "LLM unavailable for location extraction."}]}

    prompt = state['original_prompt']
    combined_prompt = state.get("combined_prompt", None)
    messages = state.get("messages", []) # Get existing messages or start new list
    print(f"Combined prompt: {combined_prompt}") # For debugging purpose
    user_confirmed_location_hierarchy = state.get("user_confirmed_location_hierarchy", []) # check if heirarchy has been specified
    user_confirmed_extracted_location = state.get("extracted_location", None) # check if material name has been specified

    if user_confirmed_location_hierarchy == []:

        extraction_prompt = f"""
        You are a helpful assistant working with procurement data for a CPG company. The data contains location hierarchies.

        Each location follows a 3-level hierarchy:
        - RegionName
        - ClusterName
        - ISOCountryName

        Your job is to extract:
        1. The **location name** mentioned in the prompt — this is typically a geographic noun (e.g., "India", "Europe", "North America", "APAC", "PTAB", "NALI" etc).
        2. The **hierarchy level** ONLY IF it is *explicitly stated* using keywords like:
        - "at the country level"
        - "for the region"
        - "by cluster"
        - "at ISOCountryName level"
        - "at RegionName"
        - "at the Cluster level"

        Do **NOT infer or assume** the hierarchy based on the location or any general terms like "by country", "split by country", "grouped by country", or "by plant". Only use **explicit keywords** tied to the hierarchy level names as shown above.

        Examples:
        - Prompt: "What is the spend in India?"
        → Location: India  
        → Hierarchy: None

        - Prompt: "Spend in India at country level"
        → Location: India  
        → Hierarchy: ISOCountryName

        - Prompt: "What is spend for North America by country and plant for year 2024?"
        → Location: North America  
        → Hierarchy: None  (Do not assume ISOCountryName just because of the phrase "by country")

        Format your response strictly as:
        location: <location name or None>  
        hierarchy: <RegionName | ClusterName | ISOCountryName | None>

        User Prompt:
        \"{prompt}\"
        """


        print("Calling LLM to extract location and hierarchy...")
        llm_response = llm_model.invoke(extraction_prompt)
        llm_output = llm_response.content.strip()
        print(f"LLM output:{llm_output}") # Debugging

        # Parse LLM output
        extracted_location = None
        extracted_hierarchy = None

        for line in llm_output.splitlines():
            if line.lower().startswith("location:"):
                value = line.split(":", 1)[1].strip()
                extracted_location = value if value.lower() != "none" else None
                print(extracted_location)
            elif line.lower().startswith("hierarchy:"):
                value = line.split(":", 1)[1].strip()
                extracted_hierarchy = value if value.lower() != "none" else None
                print(extracted_hierarchy)
        
        if extracted_location and extracted_hierarchy:
            print(f"Extracted location: {extracted_location}" and f"Extracted hierarchy: {extracted_hierarchy}")
            confirmed_location_hierarchy = extracted_hierarchy
            combined_prompt = f"combined prompt: '{combined_prompt}'\nHiearchy level for {extracted_location} is: '{confirmed_location_hierarchy}'"
            messages.append({"role": "assistant", "content": f"Extracted location is '{extracted_location}' and the extracted hierarchy is '{extracted_hierarchy}."})
        elif extracted_location and extracted_hierarchy is None:
            print("Checking location hierarchy ambiguity using UC function agent...")
            hierarchy_level = call_get_location_hierarchy_level(extracted_location)
            print(f"UC function result: {hierarchy_level}")
            
            if hierarchy_level == "multiple":
                messages.append({"role": "assistant", "content": f"LLM attempted location extraction: Found '{extracted_location}' is present in multiple hierarchies. Please confirm the location hierarchy level."})
                confirmed_location_hierarchy = None
                while True:
                    user_input = interrupt(
                        {
                            "messages": messages, 
                            "message": "Please confirm the location hierarchy level"
                            }
                    ).strip()
                
                    if confirmed_location_hierarchy:
                        if isinstance(user_input, str) and user_input.strip():
                            confirmed_location_hierarchy = user_input
                            print(f"confirmed_location_hierarchy: {confirmed_location_hierarchy}")
                            break
                        else:
                            print("Corrected hierarchy cannot be empty. Let's try confirming again.")
                    else:
                        if user_input.lower() == 'skip':
                            confirmed_location_hierarchy = None # Explicitly no location provided
                            print("Skipping hierarchy confirmation.")
                            break
                        else:
                            confirmed_location_hierarchy = user_input
                            print(f"Using provided hierarchy: {confirmed_location_hierarchy}")
                            break


            elif hierarchy_level == "no hierarchy identified":
                confirmed_location_hierarchy = None
                combined_prompt = f"combined prompt: '{combined_prompt}'"
                messages.append({"role": "assistant", "content": "No matching location found in hierarchy table."})
            else:
                confirmed_location_hierarchy = hierarchy_level
                combined_prompt = f"combined prompt: '{combined_prompt}'\nHiearchy level for {extracted_location} is: '{confirmed_location_hierarchy}'"
                messages.append({"role": "assistant", "content": f"Single hierarchy found: '{hierarchy_level}'."})
        else:
            print("No location or hierarchy found. Skipping location extraction.")
            confirmed_location_hierarchy = None
            combined_prompt = f"combined prompt: '{combined_prompt}"
            messages.append({"role": "assistant", "content": "No location or hierarchy found. Skipping location extraction."})
    else:
        if user_confirmed_location_hierarchy == 'skip':
            confirmed_location_hierarchy = None
            extracted_location = None
            print("user decided to skip location hierarchy confirmation.")
        else:
            confirmed_location_hierarchy = user_confirmed_location_hierarchy
            if user_confirmed_extracted_location == []:
                extracted_location = None
            else:
                extracted_location = user_confirmed_extracted_location
                # Prepare the combined prompt for Genie
            if confirmed_location_hierarchy:
                combined_prompt = f"Combined prompt: '{combined_prompt}'\nHiearchy level for {extracted_location} is: '{confirmed_location_hierarchy}'"
                messages.append({"role": "user", "content": f"Confirmed hierarchy is '{confirmed_location_hierarchy}' for the location '{extracted_location}."})
            else:
                combined_prompt = f"combined prompt: '{combined_prompt}'"
                messages.append({"role": "user", "content": "User indicated no specific hierarchy even if there is hierarchy level ambiguity identified"})
    
    print(f"Final Combined prompt for Genie: {combined_prompt}") # Debugging

    state.setdefault("worker_outputs", {})
    state["worker_outputs"]["location_hierarchy_resolver_agent"] = {
        "combined_prompt": combined_prompt
    }
    
    return {
    "user_confirmed_location_hierarchy": confirmed_location_hierarchy,
    "extracted_location" : extracted_location,
    "combined_prompt": combined_prompt,
    "messages": messages
    }

# COMMAND ----------

# Use the pi_spend_genie_agent implementation provided by the user
def pi_spend_genie_agent(state: AgentState):
    """
    Calls the instantiated GenieAgent with the combined prompt.
    (This is the function provided by the user).
    """
    print("--- Pi Spend Genie Agentode ---")
    combined_prompt = state.get("combined_prompt")
    print(f"Combined prompt for Genie: {combined_prompt}")
    messages = state.get("messages", []) # Get current messages
    print(f"Current messages: {messages}")
    if not combined_prompt:
        # This case should ideally be prevented by the graph logic
        print("Error: Missing combined_prompt for Genie. Aborting Genie step.")
        messages.append({"role": "assistant", "content": "Error: Genie called without combined_prompt."})
        # Return state without calling Genie, maybe add an error flag?
        return {"messages": messages}

    print(f"Calling PI Spend Genie with combined prompt...")
    try:
        # Call Genie agent with the combined prompt using the structure it expects
        # The user's example passed HumanMessage, let's adapt
        result = spend_genie_agent.invoke({
            "messages": [HumanMessage(content=combined_prompt)]
        })
        print("PI Spend Genie invocation successful.")

        # Process result - ensure 'messages' key exists and is a list
        genie_response_content = "Genie processing failed or returned unexpected format." # Default
        if isinstance(result.get("messages"), list) and result["messages"]:
                # Assuming the relevant response is the last message content
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    genie_response_content = last_message.content
                    # Convert AIMessage to a dictionary
                    genie_message = {"role": "assistant", "content": genie_response_content, "name": "Genie"}
                    messages.append(genie_message) # Append the dictionary

        state.setdefault("worker_outputs", {})
        state["worker_outputs"]["pi_spend_genie_agent"] = {
            "combined_prompt": combined_prompt
        }
            
        print(f"Genie response content: {genie_response_content}") # for debugging purpose
        return {
            "messages": messages,
            #"worker_outputs": worker_outputs,
            # Optional: clear combined_prompt if no longer needed
            # "combined_prompt": None,
        }
    except Exception as e:
        print(f"Error during PI Spend Genie agent invocation: {e}")
        error_message = f"Error calling PI Spend Genie: {e}"
        return {
            "messages": messages + [{"role": "assistant", "content": error_message}],
            "worker_outputs": {**state.get("worker_outputs", {}), "Genie": f"ERROR: {e}"}
        }

# COMMAND ----------

def summary_agent(state: AgentState,llm_model):
    messages = state.get("messages", [])
    print(f"Current messages: {messages}")
    summary_prompt = f"""
    You are a Procurement Insights Assistant. Your task is to summarize Genie agent's response in a clear, friendly, and actionable format for a business user.


    Content to summarize:
    "{messages}"
    
    Instructions:
    - Focus only on the response from the Genie agent.
    - Use the exact number format returned (do NOT convert to millions, use scientific notation, or round off values).
    - If Genie provided valid data (like total spend), summarize it using:
        • Plain bullet points or short paragraphs or in tabular format, as applicable.
    - Exact figures (e.g., $693,986,110.05)
    - The spend amount returned by the Genie agent is always represented in Euros (€). Never assume or convert the currency to any other unit.
    - If Genie returned no results, null, error, or blank values:
        • Mention that no meaningful data was returned
        • Refer to earlier user messages like “no matching material” for context
        • Suggest the user refine or rephrase the question
    - Make the output sound helpful and business-friendly
    - Do NOT include technical phrases like "summary response" or agent names
    - You can optionally offer the user to ask for further breakdowns (e.g., by region, supplier)

    Be accurate, clear, and conversational.
    """

    summary_response = llm_model.invoke(summary_prompt)
    summary_response = summary_response.content
    print(f"LLM summary: '{summary_response}'")
    messages.append({"role": "assistant", "content": f"summary: '{summary_response}'."})

    state.setdefault("worker_outputs", {})
    state["worker_outputs"]["summary_agent"] = {
        "summary_response": summary_response
    }

    return {
    "summary_response": summary_response,
    "messages": messages,
    "next": END
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Supervisor Logic (LLM-Driven Super Agent)

# COMMAND ----------

def supervisor_agent(state: AgentState, llm_model: ChatDatabricks):
    """
    Supervisor agent that coordinates the multi-agent workflow using LLM logic.
    Ensures no agent is called more than once.
    """
    print("--- Supervisor agent (LLM based) Deciding Next Step ---")
    
    # List of all valid agent steps in order of execution
    allowed_steps = [
        "material_hierarchy_resolver_agent",
        "location_hierarchy_resolver_agent",
        "pi_spend_genie_agent",
        "summary_agent"
    ]

    # Extract which agents have already been called
    worker_outputs = state.get("worker_outputs", {})
    agents_called = set(worker_outputs.keys())
    print(f"Agents already called: {agents_called}")

    # If all agents have already been called, return summary_agent as the final step
    if "summary_agent" in agents_called:
        print("Summary already generated. Workflow complete.")
        return {"next_node": None}

    # If any agents remain unused, pick the first one in order
    remaining_steps = [agent for agent in allowed_steps if agent not in agents_called]

    if not llm_model:
        print("LLM not available for supervisor. Falling back to default next agent.")
        return {"next_node": remaining_steps[0] if remaining_steps else "summary_agent"}

    # Provide a clean summary of state to LLM
    prompt = f"""
    You are a supervisor coordinating a multi-step workflow for answering procurement questions.

    Here is the current context:
    - User prompt: {state.get("original_prompt", "")}
    - Messages so far: {state.get("messages", [])}
    - Worker outputs so far: {worker_outputs}

    You must choose the **next agent to call**, strictly following these rules:
    1. Each agent must be called **only once** per workflow run.
    2. Agents already present in 'worker_outputs' have been called. Do not call them again.
    3. Proceed in the following order:
    - First: material_hierarchy_resolver_agent
    - Then: location_hierarchy_resolver_agent
    - Then: pi_spend_genie_agent
    - Finally: summary_agent
    4. If all previous steps are done, call 'summary_agent' to complete the workflow.

    Choose exactly one of:
    'material_hierarchy_resolver_agent', 'location_hierarchy_resolver_agent', 'pi_spend_genie_agent', 'summary_agent'

    Next agent to call:
    """.strip()

    try:
        print("Asking LLM to decide the next step...")
        llm_response = llm_model.invoke(prompt)
        next_step_raw = llm_response.content.strip().strip("'").strip('"')
        print(f"Supervisor LLM raw response: '{next_step_raw}'")

        # Final validation
        if next_step_raw in allowed_steps:
            if next_step_raw in agents_called:
                print(f"LLM suggested '{next_step_raw}', but it was already called. Picking next available agent.")
                for step in allowed_steps:
                    if step not in agents_called:
                        print(f"Next available step: {step}")
                        return {"next_node": step}
                print("All steps already completed. Returning summary_agent.")
                return {"next_node": "summary_agent"}
            else:
                print(f"Supervisor LLM suggests next step: '{next_step_raw}'")
                return {"next_node": next_step_raw}
        else:
            print(f"LLM suggested invalid or reused agent: '{next_step_raw}'. Falling back to next unused agent.")
            return {"next_node": remaining_steps[0] if remaining_steps else "summary_agent"}

    except Exception as e:
        print(f"Error during supervisor LLM call: {e}")
        return {"next_node": remaining_steps[0] if remaining_steps else "summary_agent"}


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the Graph

# COMMAND ----------

# Optional: Setup memory for persistence
# memory = SqliteSaver.from_conn_string(":memory:")

workflow = StateGraph(AgentState)

# Add an initial node
def start_node(state: AgentState):
    print("--- Start Node ---")
    print("Input State:", state)
    return state
workflow.add_node("start", start_node)

# Add the agent nodes
workflow.add_node("material_hierarchy_resolver_agent", partial(material_hierarchy_resolver_agent, llm_model=llm))
workflow.add_node("location_hierarchy_resolver_agent", partial(location_hierarchy_resolver_agent, llm_model=llm))

workflow.add_node("summary_agent", partial(summary_agent, llm_model=llm))

workflow.add_node("pi_spend_genie_agent", pi_spend_genie_agent)
#workflow.add_node("get_material_hierarchy_label_agent", partial(get_material_hierarchy_label_agent, llm_model=llm))
workflow.add_node("supervisor_agent", partial(supervisor_agent, llm_model=llm)) # Supervisor uses LLM

# Define the entry point
workflow.set_entry_point("start")

# Define the first transition to the supervisor
workflow.add_edge("start", "supervisor_agent")

# Define transitions based on the supervisor's output
workflow.add_conditional_edges(
    "supervisor_agent",
    lambda state: state["next_node"], # Get the next node from the state
    {
        "material_hierarchy_resolver_agent": "material_hierarchy_resolver_agent",
        "location_hierarchy_resolver_agent": "location_hierarchy_resolver_agent",
        "pi_spend_genie_agent": "pi_spend_genie_agent",
        "summary_agent": "summary_agent"
    }
)

# Define linear flow from other nodes back to the supervisor for the next decision
workflow.add_edge("material_hierarchy_resolver_agent", "supervisor_agent")
workflow.add_edge("location_hierarchy_resolver_agent", "supervisor_agent")
workflow.add_edge("pi_spend_genie_agent", "supervisor_agent")



# COMMAND ----------

# Enable Interrupt mechanism and compile the graph
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the Graph

# COMMAND ----------

import re

def check_if_heirarchy_resolver_exists(request):
    for message in request['messages'][-2:]:
        if "Please confirm the material hierarchy level" in message['content']:
            return message['content']
    return False


def check_if_location_heirarchy_resolver_exists(request):
    for message in request['messages'][-2:]:
        if "Please confirm the location hierarchy level" in message['content']:
            return message['content']
    return False

# check_if_heirarchy_resolver_exists(request)


# pre‑compile the pattern for speed if you'll call it many times
_PATTERN = re.compile(
    r"""LLM\s+attempted\s+material\s+extraction:\s*Found\s+['"]?      # fixed prefix
        (?P<word>[^'"\s]+)                                            # the word we want
    """,
    re.IGNORECASE | re.VERBOSE,
)

def extract_material(line: str) -> str | None:
    """
    Return the word that follows
    'LLM attempted material extraction: Found'
    or None if the pattern is not present.
    """
    match = _PATTERN.search(line)
    return match.group("word") if match else None


# pre‑compile the pattern for speed if you'll call it many times
_PATTERN_location = re.compile(
    r"""LLM\s+attempted\s+location\s+extraction:\s*Found\s+['"]?      # fixed prefix
        (?P<word>[^'"\s]+)                                            # the word we want
    """,
    re.IGNORECASE | re.VERBOSE,
)

def extract_location(line: str) -> str | None:
    """
    Return the word that follows
    'LLM attempted material extraction: Found'
    or None if the pattern is not present.
    """
    match = _PATTERN_location.search(line)
    return match.group("word") if match else None

# COMMAND ----------
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        print("check if thread_id exists in state")
        request = {"messages": self._convert_messages_to_dict(messages)}
        print(request)
        state = request['messages']
        if app.get_state({"configurable": {"thread_id": custom_inputs['thread_id']}}).values == {}:
            print("thread_id does not exists in state")
            if custom_inputs.get("state"):
                print("state provided")
                app.update_state({"configurable": {"thread_id": custom_inputs['thread_id']}},custom_inputs.get("state"),"material_hierarchy_resolver_agent")
            else:
                state = {
                "original_prompt": request['messages'][-1]['content'],
                "user_confirmed_hierarchy": [],
                "user_confirmed_location_hierarchy": [],
                "extracted_material" : [],
                "extracted_location" : [],
                "combined_prompt": [],
                "final_response": [],
                # "messages": request['messages'],
                "messages": [request['messages'][-1]],
                # Dictionary to store outputs from different workers (like Genie)
                "worker_outputs": {}}
                # app.update_state({"configurable": {"thread_id": custom_inputs['thread_id']}},initial_state,"supervisor_agent")
            
        else:
            print("thread_id exists in state")
            state = app.get_state({"configurable": {"thread_id": custom_inputs['thread_id']}}).values

        if check_if_heirarchy_resolver_exists(request):
            print("yes heirarchy_resolver ")
            updated_state = {
            "extracted_material" : extract_material(check_if_heirarchy_resolver_exists(request)),
            "user_confirmed_hierarchy" : request['messages'][-1]['content']}
            app.update_state({"configurable": {"thread_id": custom_inputs['thread_id']}},updated_state,"material_hierarchy_resolver_agent")
            state = app.get_state({"configurable": {"thread_id": custom_inputs['thread_id']}}).values
        
        if check_if_location_heirarchy_resolver_exists(request):
            print("yes location_resolver ")
            updated_state = {
            "extracted_location" : extract_location(check_if_location_heirarchy_resolver_exists(request)),
            "user_confirmed_location_hierarchy" : request['messages'][-1]['content']}
            app.update_state({"configurable": {"thread_id": custom_inputs['thread_id']}},updated_state,"location_hierarchy_resolver_agent")
            state = app.get_state({"configurable": {"thread_id": custom_inputs['thread_id']}}).values

        for chunk in app.stream(state, config={"configurable": {"thread_id": custom_inputs['thread_id']}}, stream_mode="values"):
            pass
        # get the final chunk
        ms= [ChatAgentMessage(**{**msg ,**{"id":str(uuid.uuid4())}}) for msg in chunk.get("messages", [])]

        # chunk.pop('messages', None)
                        
        custom_outputs = {"state" :chunk,
                        "thread_id": custom_inputs['thread_id']}
        return ChatAgentResponse(messages=ms,custom_outputs= custom_outputs)

# COMMAND ----------

mlflowapp = LangGraphChatAgent(app)
mlflow.models.set_model(mlflowapp)

# COMMAND ----------

# from IPython.display import display, Image

# display(Image(graph.get_graph().draw_mermaid_png()))