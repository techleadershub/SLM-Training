"""The local small model as a normal LangChain chat model.

    from lc_slm import SmallChatModel
    model = SmallChatModel()                      # Qwen2.5-3B on the GPU, loaded once
    model.invoke("Hello")                         # returns an AIMessage
    agent = create_agent(model=model, tools=[...])   # works with create_agent, LangGraph nodes and middleware

Nothing here is specific to this course: it is the usual pattern for adding any local model to LangChain. It does three jobs:
  1. turn LangChain messages (human, AI, tool) into the chat format the model's template expects;
  2. hand the tool descriptions to the model, so it can ask for a tool;
  3. read the model's reply and turn "<tool_call>" text into a real tool_calls list on the AIMessage.
"""
import re
import uuid
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

import graph_slm


def _to_chat(message: BaseMessage) -> dict:
    """One LangChain message as the {"role", "content"} dict the chat template reads."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content}
    if isinstance(message, AIMessage):
        out = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            out["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                 for c in message.tool_calls]
        return out
    return {"role": "user", "content": str(message.content)}


class SmallChatModel(BaseChatModel):
    """Qwen2.5 running on this machine, behind the standard LangChain chat-model interface."""

    max_new_tokens: int = 200

    @property
    def _llm_type(self) -> str:
        return "local-small-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Remember the tool descriptions; they are passed to the chat template on every call."""
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> ChatResult:
        chat = [_to_chat(m) for m in messages]
        text = graph_slm.slm(chat, tools=kwargs.get("tools"), max_new_tokens=self.max_new_tokens)
        calls = graph_slm.parse_tool_calls(text)
        content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
        tool_calls = [{"name": c["name"], "args": c["arguments"], "id": "call_" + uuid.uuid4().hex[:8], "type": "tool_call"} for c in calls]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls))])
