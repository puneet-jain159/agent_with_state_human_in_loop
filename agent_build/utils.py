import json
import logging
from typing import Optional, Tuple, List, Dict, Any
import uuid

# Add path for imports
import sys
import os

from databricks.sdk import WorkspaceClient

from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from langgraph.types import interrupt
from psycopg import connect
from langgraph.checkpoint.postgres import PostgresSaver

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging with structured format and appropriate level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("procurement_agent")
    
    # Set level
    try:
        logger.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        logger.setLevel(logging.INFO)
        print(f"Invalid log level '{level}', defaulting to INFO")
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler with structured format
    handler = logging.StreamHandler()
    
    # Use different formats based on log level
    if level.upper() == "DEBUG":
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-25s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def get_logger(name: str = "procurement_agent") -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

# Initialize Databricks client
client = DatabricksFunctionClient()


def call_get_material_hierarchy_level(material_name: str) -> str:
    """
    Call UC function to get material hierarchy level
    
    Args:
        material_name: Name of the material to analyze
        
    Returns:
        Hierarchy level or 'multiple'/'no hierarchy identified'
    """
    try:
        toolkit = UCFunctionToolkit(
            function_names=["malay_demo.procurement.get_material_hierarchy_level"],
            client=client
        )
        tools = toolkit.tools
        result = tools[0].invoke({"material_name": material_name})
        
        parsed_result = json.loads(result)
        hierarchy_level = parsed_result.get("value", "no hierarchy identified")
        
        # Get logger for this function
        logger = get_logger()
        logger.debug(f"UC function result for {material_name}: {hierarchy_level}")
        
        return hierarchy_level
    except Exception as e:
        logger = get_logger()
        logger.error(f"Error calling material hierarchy function: {e}")
        return "no hierarchy identified"


def call_get_location_hierarchy_level(location_name: str) -> str:
    """
    Call UC function to get location hierarchy level
    
    Args:
        location_name: Name of the location to analyze
        
    Returns:
        Hierarchy level or 'multiple'/'no hierarchy identified'
    """
    try:
        toolkit = UCFunctionToolkit(
            function_names=["malay_demo.procurement.get_location_hierarchy_level"],
            client=client
        )
        tools = toolkit.tools
        result = tools[0].invoke({"location_name": location_name})
        
        parsed_result = json.loads(result)
        hierarchy_level = parsed_result.get("value", "no hierarchy identified")
        
        # Get logger for this function
        logger = get_logger()
        logger.debug(f"UC function result for {location_name}: {hierarchy_level}")
        
        return hierarchy_level
    except Exception as e:
        logger = get_logger()
        logger.error(f"Error calling location hierarchy function: {e}")
        return "no hierarchy identified"


def parse_llm_extraction_output(llm_output: str) -> dict:
    """Parse LLM output for material/location extraction"""
    result = {
        "material": None,
        "hierarchy": None,
        "location": None,
        "location_hierarchy": None
    }
    
    for line in llm_output.splitlines():
        line = line.strip()
        if line.lower().startswith("material:"):
            value = line.split(":", 1)[1].strip()
            result["material"] = value if value.lower() != "none" else None
        elif line.lower().startswith("hierarchy:"):
            value = line.split(":", 1)[1].strip()
            result["hierarchy"] = value if value.lower() != "none" else None
        elif line.lower().startswith("location:"):
            value = line.split(":", 1)[1].strip()
            result["location"] = value if value.lower() != "none" else None
        elif line.lower().startswith("location_hierarchy:"):
            value = line.split(":", 1)[1].strip()
            result["location_hierarchy"] = value if value.lower() != "none" else None
    
    return result


def build_combined_prompt(original_prompt: str, material_info: dict = None, location_info: dict = None) -> str:
    """Build combined prompt for Genie agent"""
    prompt_parts = [f"Original request: '{original_prompt}'"]
    
    if material_info and material_info.get("material") and material_info.get("hierarchy"):
        prompt_parts.append(f"Hierarchy level for {material_info['material']}: '{material_info['hierarchy']}'")
    
    if location_info and location_info.get("location") and location_info.get("location_hierarchy"):
        prompt_parts.append(f"Hierarchy level for {location_info['location']}: '{location_info['location_hierarchy']}'")
    
    return "\n".join(prompt_parts)

# =============================================================================
# AUXILIARY FUNCTIONS (moved from main.py)
# =============================================================================

def extract_llm_response(llm_output: str, field_names: list) -> dict:
    """Parse LLM output to extract specified fields."""
    result = {}
    for line in llm_output.splitlines():
        line_lower = line.lower()
        for field in field_names:
            if line_lower.startswith(f"{field.lower()}:"):
                value = line.split(":", 1)[1].strip()
                result[field] = value if value.lower() != "none" else None
                break
    return result


def handle_hierarchy_ambiguity(extracted_item: str, hierarchy_level: str, 
                              item_type: str, messages: list, combined_prompt: str) -> Tuple[str, str, list]:
    """Handle ambiguous hierarchy cases by prompting user for confirmation."""
    logger = get_logger()
    logger.info(f"⚠️  Ambiguous {item_type} hierarchy: {hierarchy_level}")
    
    messages = add_message_if_not_exists(messages, {
        "role": "assistant",
        "content": f"'{extracted_item}' appears in multiple {item_type} hierarchies. Please confirm the {item_type} hierarchy level."
    })
    
    confirmed_hierarchy = None
    while True:
        user_input = interrupt({
            "messages": messages,
            "message": f"Please confirm the {item_type} hierarchy level"
        }).strip()
        
        if user_input.lower() == 'skip':
            confirmed_hierarchy = None
            logger.info(f"⏭️  User skipped {item_type} hierarchy confirmation")
            break
        elif user_input:
            confirmed_hierarchy = user_input
            logger.info(f"✅ User confirmed {item_type} hierarchy: {confirmed_hierarchy}")
            break
        else:
            logger.warning("Empty input, retrying...")
    
    # Update combined prompt
    if confirmed_hierarchy:
        combined_prompt = f"{combined_prompt}\nHierarchy level ({item_type}) for {extracted_item}: '{confirmed_hierarchy}'"
    
    return confirmed_hierarchy, combined_prompt, messages


def process_extraction_results(extracted_item: str, extracted_hierarchy: str, 
                              item_type: str, prompt: str, messages: list,
                              hierarchy_lookup_func, combined_prompt: str) -> Tuple[str, str, list]:
    """Process extraction results and handle hierarchy resolution."""
    logger = get_logger()
    
    if extracted_item and extracted_hierarchy:
        # Both extracted, use as-is
        confirmed_hierarchy = extracted_hierarchy
        combined_prompt = f"{combined_prompt}\nHierarchy level for {extracted_item}: '{confirmed_hierarchy}'"
        messages = add_message_if_not_exists(messages, {
            "role": "assistant",
            "content": f"Extracted {item_type}: '{extracted_item}' and hierarchy: '{extracted_hierarchy}'."
        })
        return confirmed_hierarchy, combined_prompt, messages
    
    elif extracted_item and extracted_hierarchy is None:
        # Item found but hierarchy ambiguous - check lookup
        logger.info(f"🔍 Checking {item_type} hierarchy ambiguity for '{extracted_item}'")
        hierarchy_level = hierarchy_lookup_func(extracted_item)
        
        if hierarchy_level in ["multiple", "no hierarchy identified"]:
            confirmed_hierarchy, combined_prompt, messages = handle_hierarchy_ambiguity(
                extracted_item, hierarchy_level, item_type, messages, combined_prompt
            )
        else:
            confirmed_hierarchy = hierarchy_level
            combined_prompt = f"{combined_prompt}\nHierarchy level for {extracted_item}: '{confirmed_hierarchy}'"
            messages = add_message_if_not_exists(messages, {
                "role": "assistant",
                "content": f"Resolved hierarchy from lookup: '{confirmed_hierarchy}'."
            })
            logger.info(f"✅ {item_type.title()} hierarchy resolved: {confirmed_hierarchy}")
        
        return confirmed_hierarchy, combined_prompt, messages
    
    else:
        # Nothing extracted
        logger.info(f"❌ No {item_type} or hierarchy found")
        messages = add_message_if_not_exists(messages, {
            "role": "assistant",
            "content": f"No {item_type} or hierarchy found."
        })
        return None, combined_prompt, messages


def _normalize_role_content(message: Any) -> tuple[str | None, str | None]:
    """Extract a comparable (role, content) tuple from dict or LangChain message objects."""
    # Dict-like
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    # LangChain/AnyMessage-like
    content = getattr(message, "content", None)
    inferred_type = str(getattr(message, "type", "")).lower()
    role: str | None
    if inferred_type in {"human", "humanmessage"}:
        role = "user"
    elif inferred_type in {"ai", "assistant", "aimessage"}:
        role = "assistant"
    elif inferred_type in {"system", "systemmessage"}:
        role = "system"
    else:
        role = getattr(message, "role", None)
    return role, content


def add_message_if_not_exists(messages: List[Any], new_message: Dict[str, Any]) -> List[Any]:
    """Add a message only if an identical role/content is not already present.

    Handles both dict messages and LangChain message objects in the existing list.
    """
    new_role, new_content = _normalize_role_content(new_message)
    for existing in messages:
        role, content = _normalize_role_content(existing)
        if role == new_role and content == new_content:
            return messages
    return messages + [new_message]


def deduplicate_messages(messages: List[Any]) -> List[Any]:
    """Remove duplicate messages based on normalized role and content.

    Preserves the first occurrence of each unique (role, content) pair.
    """
    seen: set[tuple[str | None, str | None]] = set()
    unique_messages: List[Any] = []
    for msg in messages:
        key = _normalize_role_content(msg)
        if key not in seen:
            seen.add(key)
            unique_messages.append(msg)
    return unique_messages 

# =============================================================================
# GENIE AGENT UTILITIES
# =============================================================================

def process_genie_response(result: Any, agent_name: str = "genie_agent") -> tuple[str, dict]:
    """
    Process Genie agent response and extract content.
    
    Args:
        result: Raw result from Genie agent
        agent_name: Name of the agent for logging
        
    Returns:
        Tuple of (response_content, message_dict)
    """
    logger = get_logger()
    
    # Default response
    response_content = "Genie processing failed or returned unexpected format."
    message_dict = {
        "role": "assistant",
        "content": response_content,
        "name": agent_name
    }
    
    try:
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and messages:
                # Get the last message content
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                    message_dict["content"] = response_content
                    logger.debug(f"✅ Successfully extracted content from {agent_name}")
                else:
                    logger.warning(f"⚠️  Last message from {agent_name} has no content attribute")
            else:
                logger.warning(f"⚠️  {agent_name} returned empty or invalid messages list")
        else:
            logger.warning(f"⚠️  {agent_name} returned unexpected result format: {type(result)}")
            
    except Exception as e:
        logger.error(f"❌ Error processing {agent_name} response: {e}")
        response_content = f"Error processing {agent_name} response: {e}"
        message_dict["content"] = response_content
    
    return response_content, message_dict


def validate_genie_input(combined_prompt: str, messages: list) -> tuple[bool, str]:
    """
    Validate input for Genie agent.
    
    Args:
        combined_prompt: The prompt to send to Genie
        messages: Current message history
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not combined_prompt:
        return False, "Missing combined_prompt for Genie, aborting"
    
    if not combined_prompt.strip():
        return False, "Empty combined_prompt for Genie, aborting"
    
    if len(combined_prompt.strip()) < 10:
        return False, "Combined prompt too short for meaningful analysis"
    
    return True, ""


def update_worker_outputs(state: dict, agent_name: str, output_data: dict) -> dict:
    """
    Safely update worker outputs in state.
    
    Args:
        state: Current workflow state
        agent_name: Name of the agent
        output_data: Data to store for this agent
        
    Returns:
        Updated worker outputs
    """
    worker_outputs = state.get("worker_outputs", {})
    worker_outputs[agent_name] = output_data
    return worker_outputs


def create_error_response(error: Exception, agent_name: str, messages: list, 
                         worker_outputs: dict = None) -> dict:
    """
    Create standardized error response for agent failures.
    
    Args:
        error: The exception that occurred
        agent_name: Name of the agent that failed
        messages: Current message history
        worker_outputs: Current worker outputs
        
    Returns:
        Error response dictionary
    """
    logger = get_logger()
    error_message = f"Error calling {agent_name}: {str(error)}"
    
    logger.error(f"❌ Error during {agent_name} execution: {error}")
    
    # Create error message
    error_msg = {
        "role": "assistant",
        "content": error_message,
        "name": f"{agent_name}_error"
    }
    
    # Update worker outputs with error
    if worker_outputs is None:
        worker_outputs = {}
    
    worker_outputs[f"{agent_name}_error"] = f"ERROR: {str(error)}"
    
    return {
        "messages": messages + [error_msg],
        "worker_outputs": worker_outputs
    } 

def build_db_uri(username, instance_name, use_sp=True):
    """
    Build PostgreSQL connection URI for Databricks database instance.
    
    Args:
        username (str): Database username
        instance_name (str): Databricks database instance name
        use_sp (bool): Whether to use SP credentials (default: True)
    
    Returns:
        str: PostgreSQL connection URI
    """
    w = WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST"), 
        client_id=os.getenv("CLIENT_ID"), 
        client_secret=os.getenv("CLIENT_SECRET")
    )
    
    # Get database instance details
    instance = w.database.get_database_instance(name=instance_name)
    host = instance.read_write_dns
    
    # Get access token for database authentication
    cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[instance_name])
    pgpassword =cred.token

    # Use dynamic username for non-SP mode, otherwise use provided username
    if not use_sp:
        username = w.current_user.me().emails[0].value
    
    if "@" in username:
        username = username.replace("@", "%40")

    # Connection parameters with chatbot_schema
    db_uri = f"postgresql://{username}:{pgpassword}@{host}:5432/databricks_postgres?sslmode=require&options=-csearch_path%3Dchatbot_schema"
    return db_uri


@contextmanager
def get_pg_checkpointer():
    """Yield a PostgresSaver backed by a psycopg connection with robust settings.

    Ensures autocommit and dict_row, and adds TCP keepalive parameters to reduce
    unexpected SSL socket closures on long-running streams.
    """
    username = os.getenv("CLIENT_ID")
    instance_name = os.getenv("DB_INSTANCE_NAME")
    databricks_host = os.getenv("DATABRICKS_HOST")

    if not (username and instance_name):
        yield None
        return


    uri = build_db_uri(username, instance_name, use_sp=True)

    # Append keepalive parameters to URI to reduce idle disconnects
    keepalive_params = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
    if "?" in uri:
        if "keepalives=" not in uri:
            uri = f"{uri}&{keepalive_params}"
    else:
        uri = f"{uri}?{keepalive_params}"

    # IMPORTANT: Keep the connection open for the duration of the caller using this saver.
    # Do NOT close the connection on context exit. The caller is responsible for lifecycle.
    conn = connect(uri, autocommit=True, row_factory=dict_row)
    saver = PostgresSaver(conn)
    try:
        yield saver
    finally:
        # Intentionally do not close the connection here to avoid breaking long-lived streams/state loads.
        # The process exit or explicit shutdown should close it.
        pass