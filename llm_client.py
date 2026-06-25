"""
Client for KoboldCPP's OpenAI-compatible endpoint using NATIVE tool/function
calling (Qwen3's Hermes-style tool use) instead of response_format/json_schema
grammar constraints.

REQUIRES KoboldCPP launched with --jinja --jinjatools so it uses the model's
own chat template (which for Qwen3 includes Hermes-style tool-call support)
rather than KoboldCPP's generic "universal" tool-call heuristics. See README.

This is less battle-tested than the old grammar-constrained approach — native
tool calling on KoboldCPP is a relatively recent, actively-changing feature,
and its own docs list Gemma3 (not Qwen) as the recommended tool-calling model.
The repair-retry logic below exists because of that uncertainty, not despite it.
"""
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

import config
from schemas import (
    ACTION_TOOL_REGISTRY,
    build_action_tools,
    CRITIC_TOOL,
    RESEARCH_TOOL,
    CriticVerdictArgs,
    ResearchAnswerArgs,
)

_client = OpenAI(api_key="EMPTY", base_url=config.KOBOLD_BASE_URL)
ACTION_TOOLS = build_action_tools()
_ACTION_MODELS = {name: model for name, (_, model) in ACTION_TOOL_REGISTRY.items()}


class LLMError(Exception):
    pass


class ToolCallResult:
    """A validated tool call: name + parsed/validated arguments + the id needed
    to correctly pair the assistant/tool messages in subsequent history."""

    def __init__(self, name: str, args: BaseModel, tool_call_id: str, raw_arguments: str):
        self.name = name
        self.args = args
        self.tool_call_id = tool_call_id
        self.raw_arguments = raw_arguments

    def as_assistant_message(self) -> dict:
        """The assistant-turn message to store in history, matching OpenAI's
        tool-calling message shape so it can be replayed back to the model."""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": self.tool_call_id,
                "type": "function",
                "function": {"name": self.name, "arguments": self.raw_arguments},
            }],
        }

    def as_tool_message(self, observation: str) -> dict:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": observation}


def get_next_action(messages: list) -> ToolCallResult:
    """Main-loop call: offers every action tool, forces exactly one call."""
    return _call_with_tools(
        messages,
        tools=ACTION_TOOLS,
        tool_choice="required",
        valid_models=_ACTION_MODELS,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
    )


def get_critic_verdict(messages: list) -> CriticVerdictArgs:
    result = _call_with_tools(
        messages,
        tools=[CRITIC_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_critic_verdict"}},
        valid_models={"submit_critic_verdict": CriticVerdictArgs},
        max_tokens=config.CRITIC_MAX_TOKENS,
        temperature=config.CRITIC_TEMPERATURE,
    )
    return result.args


def get_research_answer(messages: list) -> ResearchAnswerArgs:
    result = _call_with_tools(
        messages,
        tools=[RESEARCH_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_research_answer"}},
        valid_models={"submit_research_answer": ResearchAnswerArgs},
        max_tokens=config.RESEARCH_MAX_TOKENS,
        temperature=config.RESEARCH_TEMPERATURE,
    )
    return result.args


def _call_with_tools(messages, tools, tool_choice, valid_models, max_tokens, temperature):
    response = _raw_call(messages, tools, tool_choice, max_tokens, temperature)
    result, error = _extract_tool_call(response, valid_models)
    if result:
        return result

    # One repair attempt, telling the model exactly what went wrong rather
    # than just re-asking blind.
    repair_messages = messages + [{
        "role": "user",
        "content": (
            f"That didn't produce a usable tool call ({error}). You must call exactly one of "
            f"the available tools, with all required arguments filled in correctly. Try again."
        ),
    }]
    response2 = _raw_call(repair_messages, tools, tool_choice, max_tokens, temperature)
    result2, error2 = _extract_tool_call(response2, valid_models)
    if result2:
        return result2

    raise LLMError(f"Model failed to produce a valid tool call twice. First: {error}. Second: {error2}.")


def _raw_call(messages, tools, tool_choice, max_tokens, temperature):
    try:
        return _client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise LLMError(f"Request to KoboldCPP failed: {e}")


def _extract_tool_call(response, valid_models):
    try:
        message = response.choices[0].message
    except Exception as e:
        return None, f"malformed response ({e})"

    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        content_preview = (message.content or "")[:200]
        return None, f"no tool call returned (model said: {content_preview!r})"

    call = tool_calls[0]
    name = call.function.name
    model_cls = valid_models.get(name)
    if not model_cls:
        return None, f"unknown tool name '{name}'"

    try:
        args_dict = json.loads(call.function.arguments)
    except json.JSONDecodeError as e:
        return None, f"tool '{name}' arguments were not valid JSON ({e})"

    try:
        args = model_cls.model_validate(args_dict)
    except ValidationError as e:
        return None, f"tool '{name}' arguments failed validation ({e})"

    return ToolCallResult(name, args, call.id, call.function.arguments), None
