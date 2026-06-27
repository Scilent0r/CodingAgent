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
import re
import uuid
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

# Qwen3-Coder's native chat template uses a custom XML tool-call format —
# NOT the generic Hermes-style <tool_call>{"name":...}</tool_call> JSON that
# base Qwen3/Qwen2.5 use. As of this writing, KoboldCPP's --jinjatools mode
# applies the template correctly (so the model DOES emit valid native tool
# calls) but does not reliably parse this specific XML format back into the
# API's tool_calls field (confirmed open upstream issue: llama.cpp #15012).
# This means a real, valid tool call from the model can fall through as
# plain message.content instead of populating tool_calls — observed
# intermittently, not on every call. Rather than depend on upstream support
# that doesn't exist yet for this model variant, we parse it ourselves as a
# fallback whenever tool_calls comes back empty. Format:
#   <tool_call>
#   <function=name>
#   <parameter=key>
#   value
#   </parameter>
#   </function>
#   </tool_call>
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_OPEN_FUNCTION_RE = re.compile(r"<function=([^>]+)>")


def _looks_truncated_mid_tool_call(content: str) -> bool:
    """
    True if the model clearly started a tool call (opened <function=...>)
    but the response has no matching closing tag — i.e. generation got cut
    off by max_tokens before finishing, not a genuinely malformed tool call.
    Distinguishing this matters: the fix for "ran out of tokens" is raising
    MAX_TOKENS, not debugging the XML parser.
    """
    return bool(_OPEN_FUNCTION_RE.search(content)) and not _FUNCTION_BLOCK_RE.search(content)


def _parse_qwen3_coder_xml_tool_call(content: str):
    """Returns (name, arguments_dict) or (None, None) if no match found."""
    if not content:
        return None, None
    match = _FUNCTION_BLOCK_RE.search(content)
    if not match:
        return None, None
    name = match.group(1).strip()
    body = match.group(2)
    args = {}
    for pm in _PARAMETER_RE.finditer(body):
        key = pm.group(1).strip()
        raw_value = pm.group(2).strip("\n").strip()
        # Complex types (lists, etc.) get rendered as JSON text inside the
        # XML value in practice — try to recover the real type, falling
        # back to the raw string for genuinely plain-text parameters.
        try:
            value = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            value = raw_value
        args[key] = value
    return name, args


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
    """
    Main-loop call: offers every action tool. Deliberately NOT forcing
    tool_choice="required" here — that's the OpenAI-spec way to force a
    tool call, but evidence suggests it may not be cleanly grammar-enforced
    for this model/template combo on KoboldCPP, and forcing it has been
    observed correlating with degenerate repetition output rather than
    preventing prose responses. Relying on strong system-prompt wording +
    the repair-retry loop instead, which seems to be the more robust path.
    """
    return _call_with_tools(
        messages,
        tools=ACTION_TOOLS,
        tool_choice="auto",
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
    result, error, attempted = _extract_tool_call(response, valid_models)
    if result:
        return result

    if attempted:
        repair_text = (
            f"Your call to '{attempted['name']}' didn't work: {error}. Here is exactly what "
            f"you sent: {attempted['raw_arguments']}. Send the SAME tool call again, but with "
            f"every required argument actually filled in this time."
        )
    elif error.startswith("no tool call returned"):
        repair_text = (
            "You replied with plain text instead of calling a tool. You must NEVER do that — "
            "every single response must be exactly one tool call, with no other text. If you "
            "wanted to explain or summarize something, put that in the 'thought' argument of "
            "whichever tool you call now. Call a tool now."
        )
    elif error.startswith("response was cut off"):
        repair_text = (
            "Your last response got cut off before finishing the tool call — likely because "
            "the 'thought' argument was too long. Try again, but keep 'thought' to ONE short "
            "sentence so the full tool call (including all its other arguments) fits."
        )
    else:
        repair_text = (
            f"That didn't produce a usable tool call ({error}). You must call exactly one of "
            f"the available tools, with all required arguments filled in correctly. Try again."
        )

    repair_messages = messages + [{"role": "user", "content": repair_text}]
    response2 = _raw_call(repair_messages, tools, tool_choice, max_tokens, temperature)
    result2, error2, _ = _extract_tool_call(response2, valid_models)
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
            frequency_penalty=config.FREQUENCY_PENALTY,
        )
    except Exception as e:
        raise LLMError(f"Request to KoboldCPP failed: {e}")


def _extract_tool_call(response, valid_models):
    """Returns (ToolCallResult|None, error|None, attempted_info|None).
    attempted_info captures the name + raw arguments of a call that was
    recognized but failed validation/parsing — used to ground the repair
    message in what the model actually sent, rather than asking it to
    "try again" with no memory of what it just did."""
    try:
        message = response.choices[0].message
    except Exception as e:
        return None, f"malformed response ({e})", None

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        call = tool_calls[0]
        name = call.function.name
        raw_arguments = call.function.arguments
        tool_call_id = call.id
    else:
        name, args_dict = _parse_qwen3_coder_xml_tool_call(message.content or "")
        if not name:
            content_preview = (message.content or "")[:200]
            if _looks_truncated_mid_tool_call(message.content or ""):
                return None, (
                    f"response was cut off mid-tool-call, likely hit max_tokens before "
                    f"finishing (saw: {content_preview!r})"
                ), None
            return None, f"no tool call returned (model said: {content_preview!r})", None
        raw_arguments = json.dumps(args_dict)
        tool_call_id = f"fallback-{uuid.uuid4().hex[:8]}"

    model_cls = valid_models.get(name)
    if not model_cls:
        return None, f"unknown tool name '{name}'", {"name": name, "raw_arguments": raw_arguments}

    try:
        args_dict = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return None, f"tool '{name}' arguments were not valid JSON ({e})", {"name": name, "raw_arguments": raw_arguments}

    try:
        args = model_cls.model_validate(args_dict)
    except ValidationError as e:
        return None, f"tool '{name}' arguments failed validation ({e})", {"name": name, "raw_arguments": raw_arguments}

    return ToolCallResult(name, args, tool_call_id, raw_arguments), None, None
