"""
Edit these values for your machine.
"""

# KoboldCPP OpenAI-compatible endpoint
KOBOLD_BASE_URL = "http://localhost:5001/v1"

# Some KoboldCPP builds ignore the "model" field (single model loaded anyway),
# but the OpenAI client format requires something here.
MODEL_NAME = "qwen3-coder-30b-a3b-instruct"

# Path to the python.exe INSIDE the minimal "agent-sandbox" conda env.
# Find it with: conda env list --json
SANDBOX_PYTHON = r"C:\Users\PATH\miniconda3\envs\agent-sandbox\python.exe"

# Safety limits
MAX_ITERATIONS = 18
MAX_CONSECUTIVE_TOOL_ERRORS = 3  # abort if the model keeps making the same kind of mistake
MAX_CRITIC_REJECTIONS = 2  # after this many rejections, force-finish with a warning
MAX_ASK_USER_CALLS = 5  # cap on how many clarifying questions the agent can ask per run
FINISH_NUDGE_WINDOW = 3  # iterations-remaining threshold to nudge toward finishing if tests are green
TEST_TIMEOUT_SECONDS = 30
SCRATCH_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 4000  # truncate huge stdout/stderr before feeding back to model
MAX_SEARCH_RESULTS = 40  # cap regex search hits returned to the model

# Context management: keep the last N iterations in full detail in the prompt;
# everything older gets collapsed into a one-line-per-step condensed log.
MAX_RECENT_DETAILED_TURNS = 6

# Where durable, cross-task project notes live. Persists across separate
# `run_task.py` invocations (unlike PLAN.md, which is per-task and lives
# inside that task's own runs/<timestamp> folder).
PROJECT_MEMORY_FILE = "project_memory.md"

# Sampling
TEMPERATURE = 0.4
MAX_TOKENS = 1500
CRITIC_TEMPERATURE = 0.2  # more conservative than the worker — less prone to inventing nitpicks
CRITIC_MAX_TOKENS = 1200  # needs room for a real reasoning trace, not just a one-line verdict

# --- Multi-language support ---
# Each entry defines how to run the test suite and how to execute a scratch
# script for that language. {file} in run_command gets substituted with the
# scratch file's path. test_command is run as-is (most test runners
# auto-discover files in the cwd, so no {file} substitution needed there).
#
# node prerequisites: Node.js installed and on PATH. Uses the built-in
# `node:test` runner (Node 18+) so no npm install / extra packages required.
LANGUAGES = {
    "python": {
        "extension": "py",
        "test_command": [SANDBOX_PYTHON, "-m", "pytest", "-q", "--tb=short"],
        "run_command": [SANDBOX_PYTHON, "{file}"],
    },
    "node": {
        "extension": "js",
        "test_command": ["node", "--test"],
        "run_command": ["node", "{file}"],
    },
}
DEFAULT_LANGUAGE = "python"

# Per-project config file (housekeeping) — lives in the cwd (or --repo target)
# and lets you set language/memory/protect defaults once instead of retyping
# CLI flags every run. CLI flags always override this when both are given.
WORKSPACE_CONFIG_FILENAME = ".agent_workspace.json"

# --- Windows Sandbox (stronger isolation for executing model-written code) ---
# Off by default — this is the least-tested feature in the project. Validate
# with test_sandbox_session.py BEFORE enabling this for real task runs.
#
# Currently Python-only: the sandbox VM boots with nothing installed, so it
# needs a PORTABLE Python (the official embeddable zip) mapped in read-only —
# run sandbox_provision.py once to set that up. Node isn't wired up here yet
# (would need a portable Node distribution mapped in the same way).
USE_WINDOWS_SANDBOX = False
SANDBOX_PORTABLE_PYTHON_DIR = r"C:\agent-sandbox-runtime\python"
SANDBOX_ALLOW_NETWORK = False  # the whole point is isolating untrusted generated code
SANDBOX_MEMORY_MB = 2048
SANDBOX_BOOT_TIMEOUT_SECONDS = 90  # cold VM boot + logon + worker startup
SANDBOX_COMMAND_TIMEOUT_BUFFER_SECONDS = 15  # extra grace beyond the command's own timeout
SANDBOX_POLL_INTERVAL_SECONDS = 0.5

# In-sandbox command templates (paths as seen FROM INSIDE the VM, not host
# paths — these are deliberately separate from LANGUAGES above, which uses
# the host's conda env path and is meaningless inside the sandbox).
SANDBOX_LANGUAGES = {
    "python": {
        "test_command": [r"C:\pyruntime\python.exe", "-m", "pytest", "-q", "--tb=short"],
        "run_command": [r"C:\pyruntime\python.exe", "{file}"],
    },
}

# --- Web search / doc lookup ---
# Uses the 'ddgs' package (DuckDuckGo search, no API key needed) plus a plain
# requests+BeautifulSoup fetch for reading actual page content. Toggle off to
# disable both web_search and fetch_url actions entirely.
ENABLE_WEB_TOOLS = True
MAX_WEB_RESULTS = 5
WEB_FETCH_TIMEOUT_SECONDS = 15
WEB_FETCH_MAX_CHARS = 6000

# --- Multi-source research (web_research action) ---
# Unlike plain web_search (one query, snippets only), this issues several
# query variations, fetches actual page content for the top unique results,
# and runs a dedicated synthesis call to reason across sources before
# committing to one answer + confidence level. Slower and more expensive
# per call, but meant for things worth cross-referencing rather than a
# quick lookup.
RESEARCH_MAX_QUERIES = 3
RESEARCH_RESULTS_PER_QUERY = 4
RESEARCH_MAX_FETCHES = 3  # top unique URLs to actually fetch full content for
RESEARCH_SNIPPET_FALLBACK_COUNT = 4  # extra results kept as snippet-only, for breadth
RESEARCH_SOURCE_CHAR_LIMIT = 1500  # per-source truncation fed into the synthesis prompt
RESEARCH_MAX_TOKENS = 900
RESEARCH_TEMPERATURE = 0.2  # grounded synthesis, not creative
