
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ChatMessage:
    """Represents a chat message with content and optional tool calls."""
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    raw: Any = None

class LiteLLMModel:
    def __init__(self, model_id: str, api_base: str):
        self.model_id = model_id
        self.api_base = api_base

    async def __call__(self, messages: List[Dict[str, Any]], **kwargs) -> ChatMessage:
        # Simulate a call to the model
        # In a real scenario, this would be an actual API call
        response = await self._call_model(messages, **kwargs)
        return self._convert_to_chat_message(response)

    async def _call_model(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Simulate a model response
        # In a real scenario, this would be an actual API call
        # For example:
        # response = requests.post(f"{self.api_base}/v1/chat/completions", json={"messages": messages})
        # return response.json()
        return {
            "content": "This is a simulated response from the model.",
            "tool_calls": [
                {"name": "rag_agent", "arguments": {"prompt": "Simulated prompt"}}
            ]
        }

    def _convert_to_chat_message(self, response: Dict[str, Any]) -> ChatMessage:
        # Convert the model's response to a ChatMessage object
        return ChatMessage(
            content=response.get("content"),
            tool_calls=response.get("tool_calls"),
            raw=response
        )