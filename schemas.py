"""
Per-tool parameter schemas for native function/tool calling (Qwen3's
Hermes-style tool use, via KoboldCPP's --jinja --jinjatools mode).

Each model below becomes one OpenAI-style "tool" definition. The model is
given the full list each turn and must call exactly one (tool_choice).

Design note: every action tool includes a `thought` field. This is a
deliberate, pragmatic choice rather than a purist one — Qwen3 can expose
chain-of-thought separately via `reasoning_content` in "think" mode, but
whether KoboldCPP's jinja-template tool-calling path surfaces that cleanly
and consistently is unconfirmed as of this writing. Keeping `thought` as an
explicit required argument guarantees we always get a reasoning trace to
log/display/feed back into context, regardless of templating quirks.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class WriteFileArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    path: str = Field(description="Relative file path to create, or fully overwrite if it exists.")
    content: str = Field(description="Full file content to write.")


class EditFileArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    path: str = Field(description="Relative path of the existing file to edit.")
    old_str: str = Field(description="Exact existing text to find — must appear exactly once in the file.")
    new_str: str = Field(description="Replacement text.")


class ReadFileArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    path: str = Field(description="Relative file path to read.")


class ListFilesArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")


class SearchFilesArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    pattern: str = Field(description="Regex pattern to search for across all files in the task directory.")


class RunScratchArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    code: str = Field(
        description=(
            "Scratch source code to run, in the task's current language, for investigation "
            "or debugging (e.g. printing intermediate values, reproducing a bug). Runs with "
            "access to files already written in the task directory. Not a substitute for run_tests."
        )
    )


class UpdatePlanArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    plan: str = Field(
        description="Full updated plan as a markdown checklist (e.g. '- [x] step one\\n- [ ] step two'), replacing any previous plan."
    )


class UpdateMemoryArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    note: str = Field(
        description="A durable, project-level fact worth remembering for FUTURE tasks in this same project — not task-specific detail."
    )


class AskUserArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    question: str = Field(
        description="A clarifying question for the human operator. Use ONLY for genuine ambiguity you can't resolve yourself."
    )


class GitStatusArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")


class GitDiffArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    path: Optional[str] = Field(
        default=None, description="Optional relative path to scope the diff to one file. Omit for the full diff."
    )


class WebSearchArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    query: str = Field(description="A web search query. Returns titles/URLs/snippets only.")


class WebResearchArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    queries: List[str] = Field(
        description=(
            "2-3 DIFFERENTLY-PHRASED search queries about the same question. Fetches real page "
            "content for the top results and reasons across them before answering — use this "
            "instead of web_search when the answer is time-sensitive or numeric and worth "
            "cross-referencing multiple sources for."
        )
    )


class FetchUrlArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    url: str = Field(description="A full http(s) URL to fetch and read as plain text.")


class RunTestsArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")


class FinishArgs(BaseModel):
    thought: str = Field(description="Brief reasoning for this action.")
    summary: str = Field(description="Short summary of the completed solution.")


# Tool name -> (description shown to the model, args schema)
ACTION_TOOL_REGISTRY = {
    "write_file": (
        "Create a new file, or fully overwrite an existing one. Prefer edit_file for small "
        "changes to files that already exist.",
        WriteFileArgs,
    ),
    "edit_file": (
        "Surgical find/replace on an existing file. Preferred over write_file when changing "
        "something that already exists, since it's cheaper and safer.",
        EditFileArgs,
    ),
    "read_file": ("Read the full contents of a file.", ReadFileArgs),
    "list_files": ("List every file currently in the task directory.", ListFilesArgs),
    "search_files": ("Regex search across all files in the task directory.", SearchFilesArgs),
    "run_scratch": (
        "Run scratch code for investigation or debugging before deciding on a fix. Not a "
        "substitute for run_tests.",
        RunScratchArgs,
    ),
    "update_plan": ("Replace the current plan with an updated markdown checklist.", UpdatePlanArgs),
    "update_memory": (
        "Record a durable, project-level note that should persist into future, unrelated tasks "
        "in this same project.",
        UpdateMemoryArgs,
    ),
    "ask_user": (
        "Ask the human operator a clarifying question. Use sparingly, only for genuine ambiguity.",
        AskUserArgs,
    ),
    "git_status": ("Show git status. Only meaningful in --repo mode.", GitStatusArgs),
    "git_diff": (
        "Show git diff, optionally scoped to one file. Only meaningful in --repo mode.",
        GitDiffArgs,
    ),
    "web_search": ("Quick single web search returning titles/URLs/snippets only.", WebSearchArgs),
    "web_research": (
        "Multi-query web research: fetches real page content from multiple sources and reasons "
        "across them to produce one answer with a confidence level.",
        WebResearchArgs,
    ),
    "fetch_url": ("Fetch a URL and return its readable text content.", FetchUrlArgs),
    "run_tests": (
        "Run the FULL test suite. Takes no parameters — there is no way to scope it to one file.",
        RunTestsArgs,
    ),
    "finish": (
        "Declare the task complete. Only valid if the most recent run_tests call passed. "
        "Triggers an independent critic review before actually ending the run.",
        FinishArgs,
    ),
}


def build_action_tools() -> list:
    """Converts the registry into the OpenAI-style `tools` list for the main loop."""
    tools = []
    for name, (description, model) in ACTION_TOOL_REGISTRY.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
            },
        })
    return tools


class CriticVerdictArgs(BaseModel):
    """
    Submitted via a forced single tool call after a finish attempt. Field
    order matters: reasoning, then concerns, then approved LAST — so the
    model works through the evidence before committing to a verdict,
    rather than deciding first and rationalizing after (the original
    version of this schema had `approved` declared first, and the critic
    measurably hallucinated rejections as a result — see project history).
    """

    reasoning: str = Field(
        description=(
            "Work through this BEFORE deciding anything: what does the task literally require? "
            "What do the actual file contents and test output show? Cite concrete values "
            "(expected vs actual) for anything you're unsure about. Do not assume requirements "
            "beyond what the task explicitly states."
        )
    )
    concerns: List[str] = Field(
        default_factory=list,
        description=(
            "Specific, concrete issues found, if any — each must reference an actual value or "
            "behavior from what you were shown, not a general impression. Empty if none."
        ),
    )
    approved: bool = Field(
        description=(
            "Decide this LAST, after the reasoning above. True if the implementation genuinely "
            "and robustly solves the task. If the test suite already passes and you cannot cite "
            "a SPECIFIC concrete value that violates a literal requirement from the task, you "
            "must approve — passing tests are real evidence, not something to override with a "
            "vague stylistic objection."
        )
    )


class ResearchAnswerArgs(BaseModel):
    reasoning: str = Field(
        description=(
            "Step through what each source actually says BEFORE writing the answer below. Note "
            "agreement, disagreement, and how recent/reliable each source seems."
        )
    )
    answer: str = Field(
        description=(
            "The single best-effort answer, as concise as the question allows. If sources "
            "disagree meaningfully, say so concisely rather than picking one arbitrarily."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high = multiple sources agree closely and the topic isn't fast-changing. "
            "medium = sources roughly agree or only one decent source was found. "
            "low = sources disagree, are stale, or none actually answer the question."
        )
    )
    caveats: str = Field(
        default="",
        description="Anything the user should know before trusting this. Empty string if genuinely none.",
    )


CRITIC_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_critic_verdict",
        "description": "Submit your independent review verdict on whether the finished solution genuinely solves the task.",
        "parameters": CriticVerdictArgs.model_json_schema(),
    },
}

RESEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_research_answer",
        "description": "Submit your synthesized answer after reasoning across the provided sources.",
        "parameters": ResearchAnswerArgs.model_json_schema(),
    },
}
