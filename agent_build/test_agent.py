import uuid
from typing import Optional, Dict, Any

from agent_build.mlflowlogger import LangGraphChatAgent


def create_input_example(
    query: str = "What is the total spend of tea in 2023?",
    thread_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an example input for the model.

    Matches ChatAgentRequest schema exactly.
    """
    example = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "custom_inputs": {
            "thread_id": thread_id or "1234",
        },
    }

    if state is not None:
        example["custom_inputs"]["state"] = state

    return example


def _coerce_messages_to_dict(messages: list[dict] | list[Any]) -> list[dict]:
    coerced: list[dict] = []
    for m in messages:
        if isinstance(m, dict):
            # Filter out tool messages to avoid compatibility issues
            if m.get("role") == "tool":
                # Convert tool messages to assistant messages with the content
                content = m.get("content", "")
                if content:
                    # Try to extract meaningful content from tool messages
                    try:
                        import json
                        tool_data = json.loads(content)
                        if isinstance(tool_data, dict) and "message" in tool_data:
                            # Use the message field from tool response
                            content = tool_data["message"]
                        else:
                            # Use the entire content as a string
                            content = str(tool_data)
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, use as-is
                        pass
                    
                    coerced.append({
                        "role": "assistant",
                        "content": content,
                        "id": str(uuid.uuid4())
                    })
            else:
                # Preserve all fields from non-tool messages
                coerced.append(m)
        else:
            # For non-dict messages, extract what we can
            role = getattr(m, "role", "assistant")
            content = getattr(m, "content", "")
            
            # Skip tool messages
            if role == "tool":
                continue
                
            message_dict = {"role": role, "content": content}
            
            # Add other fields if they exist
            if hasattr(m, "id"):
                message_dict["id"] = m.id
            if hasattr(m, "name"):
                message_dict["name"] = m.name
            if hasattr(m, "tool_call_id"):
                message_dict["tool_call_id"] = m.tool_call_id
            
            coerced.append(message_dict)
    return coerced


def continue_conversation(model: LangGraphChatAgent, response, query: str, thread_id: str | None = None):
    """
    Continue the conversation with the model.
    """
    prev_msgs = _coerce_messages_to_dict(getattr(response, "messages", []))
    prev_msgs.append({"role": "user", "content": query})

    return model.predict(
        messages=prev_msgs,
        custom_inputs={"thread_id": str(thread_id or "1234")},
    )


def main() -> None:
    # Build agent (auto-loads workflow from agent_build.compile)
    agent = LangGraphChatAgent()

    # Prepare example request with a unique thread_id
    thread_id = str(uuid.uuid4())
    example = create_input_example(query="What is the total spend of tea in 2023?", thread_id=thread_id)

    print({"messages": example["messages"]})

    # Call predict using the example
    response = agent.predict(
        messages= example["messages"],
        custom_inputs=example["custom_inputs"],
    )

    # Print results
    print("Initial response messages:")
    for m in _coerce_messages_to_dict(response.messages):
        print(f"- {m['role']}: {m['content']}")

    # print("\nCustom outputs:")
    # print(response.custom_outputs)

    # Continue conversation: specify hierarchy
    response = continue_conversation(
        model=agent,
        response=response,
        query="skip",
        thread_id=thread_id,
    )
    print("\nAfter hierarchy response messages:")
    for m in _coerce_messages_to_dict(response.messages):
        print(f"- {m['role']}: {m['content']}")

    # # Continue conversation: specify location hierarchy
    # response = continue_conversation(
    #     model=agent,
    #     response=response,
    #     query="country_name",
    #     thread_id=thread_id,
    # )
    # print("\nAfter location hierarchy response messages:")
    # for m in _coerce_messages_to_dict(response.messages):
    #     print(f"- {m['role']}: {m['content']}")

    
    response = continue_conversation(
        model=agent,
        response=response,
        query="What is the total spend of coffee in 2023?",
        thread_id=thread_id,
    )
    print("\nAfter location hierarchy response messages:")
    for m in _coerce_messages_to_dict(response.messages):
        print(f"- {m['role']}: {m['content']}")


if __name__ == "__main__":
    main()

