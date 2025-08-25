from typing import Dict, Any, Optional, List
import uuid
import mlflow
from mlflow.pyfunc import ChatAgent  # type: ignore
from langchain_community.adapters.openai import convert_message_to_dict
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext # type: ignore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Load environment variables early to ensure database connection works
from dotenv import load_dotenv
load_dotenv(".env")


class LangGraphChatAgent(ChatAgent):
    """Adapter that wraps a LangGraph compiled app to the MLflow ChatAgent API.

    Expects an app with methods: stream(state, config, stream_mode), and optionally
    get_state(thread_config) and update_state(thread_config, updates, node_id).
    """

    def __init__(self, app_or_workflow: Optional[Any] = None):
        # Accept either a compiled app (with stream) or a StateGraph-like workflow (with compile)
        self.graph = None
        
        # If not provided, fetch workflow from compile.py
        if app_or_workflow is None:
            try:
                from agent_build_v2.run_toolnode import graph # lazy import
                from agent_build.compile import get_pg_checkpointer  # lazy import to avoid cycles
                app_or_workflow = graph
            except Exception as _e:  # noqa: F841
                raise RuntimeError(f"Failed to import workflow from agent_build_v2.run_toolnode: {_e}") from _e
        
        # Check if it's already a compiled app
        if hasattr(app_or_workflow, "stream"):
            self.graph = app_or_workflow
        else:
            # Compile with checkpointer strictly; require Postgres when available
            from agent_build.compile import get_pg_checkpointer  # lazy import to avoid cycles
            with get_pg_checkpointer() as checkpointer:  # type: ignore[misc]
                self.graph = app_or_workflow.compile(checkpointer=checkpointer)
        
        if self.graph is None:
            raise RuntimeError("Failed to initialize graph - self.graph is None")

    def _convert_langchain_message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert LangChain message objects to dictionary format for MLflow."""
        
        try:
            # Use LangChain's built-in converter
            result = convert_message_to_dict(msg)
 
            # Ensure the message has an ID for MLflow compatibility
            if "id" not in result:
                result["id"] = str(uuid.uuid4())
            
            # Handle tool messages - ensure they have required fields for ChatAgentMessage validation
            if result.get("role") == "tool":
                # MLflow ChatAgentMessage requires both 'name' and 'tool_call_id' for tool messages
                if "name" not in result:
                    result["name"] = getattr(msg, 'name', 'unknown_tool')
                if "tool_call_id" not in result:
                    result["tool_call_id"] = getattr(msg, 'tool_call_id', str(uuid.uuid4()))
                
                # If we still don't have the required fields, convert to assistant message
                if not result.get("name") or not result.get("tool_call_id"):
                    result = {
                        "id": result.get("id", str(uuid.uuid4())),
                        "role": "assistant",
                        "content": f"Tool result: {result.get('content', '')}"
                    }
            
            return result
        except Exception:
            # Fallback for cases where convert_message_to_dict fails
            if isinstance(msg, dict):
                # If it's already a dict, ensure it has required fields
                if "id" not in msg:
                    msg["id"] = str(uuid.uuid4())
                
                # Handle tool messages in dict format
                if msg.get("role") == "tool":
                    if not msg.get("name") or not msg.get("tool_call_id"):
                        # Convert problematic tool messages to assistant messages
                        return {
                            "id": msg.get("id", str(uuid.uuid4())),
                            "role": "assistant", 
                            "content": f"Tool result: {msg.get('content', '')}"
                        }
                
                return msg
            else:
                # Fallback for unknown message types
                return {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": str(msg)
                }

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        try:
            request = {"messages": self._convert_messages_to_dict(messages)}
        except:   
            request = {"messages": messages}


        if not custom_inputs or not custom_inputs.get("thread_id"):
            raise ValueError("thread_id must be provided in custom_inputs")

        # Thread id for checkpointing/continuations
        thread_id = custom_inputs.get("thread_id")  # type: ignore[union-attr]
        thread_config = {"configurable": {"thread_id": thread_id}}

        messages = []
        for event in self.graph.stream(request, config=thread_config, stream_mode="updates"):
            for node_data in event.values():
                # Convert LangChain messages to MLflow format
                for msg in node_data.get("messages", []):
                    converted_msg = self._convert_langchain_message_to_dict(msg)
                    messages.append(ChatAgentMessage(**converted_msg))
        return ChatAgentResponse(messages=messages)




mlflowapp = LangGraphChatAgent()
mlflow.models.set_model(mlflowapp)

