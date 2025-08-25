# pip install -U langchain langgraph langchain-openai pydantic openai
# export OPENAI_API_KEY=sk-...
# pyright: reportMissingImports=false

from typing import Optional, Dict, Any, List, TypedDict, Annotated
from pydantic import BaseModel, Field
import json
import mlflow
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition  # type: ignore[import-not-found]
from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-not-found]

from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from agent_build.utils import (
    call_get_material_hierarchy_level,
    call_get_location_hierarchy_level,
    build_combined_prompt,
)
# Enable MLflow autologging
mlflow.langchain.autolog()

# Load environment variables
load_dotenv(".env")

# ───────────────────────────────────────────────────────────────────────────────
# 1) Tool schema + envelopes (result vs clarification)
# ───────────────────────────────────────────────────────────────────────────────

class GenieQueryInput(BaseModel):
    query: str = Field(description="User's analytic query for Genie")
    material: Optional[str] = Field(default=None, description="Material name if applicable")
    hierarchy: Optional[str] = Field(default=None, description="Resolved material hierarchy level if applicable (e.g., 'category', 'subcategory')")
    location: Optional[str] = Field(default=None, description="Location name if applicable")
    location_hierarchy: Optional[str] = Field(default=None, description="Resolved location hierarchy level if applicable (e.g., 'region', 'country')")

def clarification(payload: Dict[str, Any]) -> str:
    return json.dumps({"type": "clarification_request", **payload})

def result(payload: Dict[str, Any]) -> str:
    return json.dumps({"type": "result", **payload})

P2P_SPEND_GENIE_SPACE_ID = "01f0421695fe1fb694762b30f68f799d"
genie_agent = GenieAgent(P2P_SPEND_GENIE_SPACE_ID, "p2p_spend_genie")


@tool("genie_query", args_schema=GenieQueryInput)
def genie_query_tool(
    query: str,
    material: Optional[str] = None,
    hierarchy: Optional[str] = None,
    location: Optional[str] = None,
    location_hierarchy: Optional[str] = None,
) -> str:
    """
    Run a Genie query. to get data about procurement and spend. if hierarchy and location is not provided, ask for clarification.

    Behavior:
    - If material/location hierarchy is missing when likely required, attempts to resolve via UC
      lookup when the corresponding entity is provided. If still ambiguous, returns
      a clarification request indicating which hierarchy field(s) are missing.
    - The backend data contains rolls up at multiple levels of hierarchy
    - The backend data contains rolls up at multiple levels of location hence it needs to be specified even if country is specified i.e country ,region, continent, etc can be at cluster region level as well
    - Otherwise, invokes Genie and returns
      {"type": "result", "message": ..., "details": {...}}.

    Parameters
    - query: The user's analytics question for Genie.
    - material: The material in scope, if any.
    - hierarchy: The resolved material hierarchy level, if applicable (e.g., category, subcategory).
    - location: The location in scope, if any.
    - location_hierarchy: The resolved location hierarchy level, if applicable (e.g., region, country).
    """
    # Heuristics: determine which hierarchies are needed

    print("query:", query)
    print("material:", material)
    print("hierarchy:", hierarchy)
    print("location:", location)
    print("location_hierarchy:", location_hierarchy)

    lower_q = query.lower()
    needs_material_hierarchy = ((material is not None) or ("material" in lower_q)) and not hierarchy
    needs_location_hierarchy = ((location is not None) or ("location" in lower_q)) and not location_hierarchy

    missing_fields: list[str] = []

    if needs_material_hierarchy:
        if material:
            resolved = call_get_material_hierarchy_level(material)
            if resolved in ["multiple", "no hierarchy identified"]:
                missing_fields.append("hierarchy")
            else:
                hierarchy = resolved
        else:
            missing_fields.append("hierarchy")

    if needs_location_hierarchy:
        if location:
            loc_resolved = call_get_location_hierarchy_level(location)
            if loc_resolved in ["multiple", "no hierarchy identified"]:
                missing_fields.append("location_hierarchy")
            else:
                location_hierarchy = loc_resolved
        else:
            missing_fields.append("location_hierarchy")

    if missing_fields:
        # Build a concise message indicating what is needed
        needs = []
        if "hierarchy" in missing_fields:
            needs.append("material hierarchy level (e.g., category, subcategory)")
        if "location_hierarchy" in missing_fields:
            needs.append("location hierarchy level (e.g., region, country)")
        msg = "Please confirm the " + (" and ".join(needs))
        return clarification({
            "message": msg,
            "missing": missing_fields,
            "hint": "Reply with e.g., hierarchy=category, location_hierarchy=region",
        })

    # Build combined prompt if we have material + hierarchy context
    combined_prompt = query
    if material and hierarchy:
        combined_prompt = build_combined_prompt(
            combined_prompt, material_info={"material": material, "hierarchy": hierarchy}
        )
    if location and location_hierarchy:
        combined_prompt = build_combined_prompt(
            combined_prompt, location_info={"location": location, "location_hierarchy": location_hierarchy}
        )

    # Call Genie
    genie_result = genie_agent.invoke({"messages": [HumanMessage(content=combined_prompt)]})
    # Package raw Genie output as result. Caller can format further.
    return result({
        "message": "Genie query executed.",
        "details": {
            "query": combined_prompt,
            "hierarchy": hierarchy,
            "material": material,
            "location": location,
            "location_hierarchy": location_hierarchy,
            "genie_output": getattr(genie_result, "content", str(genie_result)),
        },
    })


# ───────────────────────────────────────────────────────────────────────────────
# 2) Graph state
# ───────────────────────────────────────────────────────────────────────────────

from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ───────────────────────────────────────────────────────────────────────────────
# 3) Nodes: agent (LLM) + tools
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM = """You are a procurement analytics assistant.
- Prefer using the `genie_query` tool to answer.
- If the last tool output is a JSON object with {"type":"clarification_request","message":...},
  ask the user for exactly those missing details (e.g., "hierarchy", "location_hierarchy") and do not invent values.
- If it's {"type":"result",...}, summarize the findings clearly and keep responses concise.
- Use bullet points for lists of metrics or dimensions when helpful.
"""

LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Bind the tool so the LLM can decide to call it
bound_llm = llm.bind_tools([genie_query_tool])

def agent_node(state: GraphState) -> GraphState:
    """
    The agent looks at conversation so far and decides:
    - Ask user for missing info (if prior tool asked for clarification), OR
    - Call the tool, OR
    - Answer directly (rare in this example).
    """
    messages = [SystemMessage(SYSTEM)] + state["messages"]

    ai = bound_llm.invoke(messages)
    return {"messages": [ai]}

# Prebuilt ToolNode executes any tool calls the model requested.
tools_node = ToolNode([genie_query_tool])

# ───────────────────────────────────────────────────────────────────────────────
# 4) Graph wiring: agent → (maybe tools) → agent → ...
#    We’ll loop until the agent produces no tool calls, then END.
# ───────────────────────────────────────────────────────────────────────────────

graph = StateGraph(GraphState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

# START -> agent
graph.add_edge(START, "agent")
# If the agent called tools, go to tools; otherwise END
# tools_condition returns either "tools" or "__end__"
graph.add_conditional_edges("agent", tools_condition, {
    "tools": "tools",
    "__end__": END
})
# After tools run, echo tool outputs into the state as ToolMessages, then go back to agent
def tool_to_messages(state: GraphState) -> GraphState:
    """
    ToolNode already ran tools and appended tool outputs as ToolMessages.
    We simply pass-through to go back to the agent to interpret tool output
    (e.g., ask user for clarification or present result).
    """
    return state

graph.add_edge("tools", "agent")




# ───────────────────────────────────────────────────────────────────────────────
# 5) Demo loop: multi-turn conversation with the same thread_id
# ───────────────────────────────────────────────────────────────────────────────

def pretty_print_last(ai_or_tool_text: str):
    # Try to show structured envelopes nicely if present
    try:
        data = json.loads(ai_or_tool_text)
        if data.get("type") == "clarification_request":
            print("Assistant:", data["message"])
            if data.get("missing"):
                print("Missing:", ", ".join(data["missing"]))
            return
        if data.get("type") == "result":
            print("Assistant:", data["message"])
            return
    except (ValueError, TypeError):
        pass
    print("Assistant:", ai_or_tool_text)

def run_chat():


    # Optional memory/checkpoint so each user turn resumes prior state
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    thread = {"configurable": {"thread_id": "trip-123"}}
    # Initial user turn
    print("Assistant: Hi! I’m your procurement analytics assistant. Ask me about spend, suppliers, categories, or trends.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        # Invoke one turn: we append the user's message and let the graph run until it stops
        graph_result = app.invoke({"messages": [HumanMessage(user)]}, config=thread)

        # Find the most recent assistant message and show it
        msgs = graph_result["messages"]
        last_ai = next((m for m in reversed(msgs) if isinstance(m, (AIMessage, ToolMessage))), None)
        if last_ai is None:
            print("Assistant: (no response?)")
            continue

        content = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
        pretty_print_last(content)

if __name__ == "__main__":
    run_chat()
