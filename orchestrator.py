import json
from pathlib import Path
from typing import Optional, Iterable

import config
from llm_client import get_next_action, get_critic_verdict, get_research_answer, LLMError
from tools import (
    write_file,
    edit_file,
    read_file,
    list_files,
    search_files,
    run_scratch,
    run_tests,
    append_project_memory,
    git_status,
    git_diff,
    ensure_clean_working_tree,
    create_and_checkout_branch,
    git_commit_all,
    web_search,
    fetch_url,
    gather_research_sources,
    ToolError,
    EXCLUDED_DIR_NAMES,
)


def _build_system_prompt(language: str, git_mode: bool) -> str:
    lang_cfg = config.LANGUAGES.get(language, {})
    test_cmd = " ".join(lang_cfg.get("test_command", ["<unknown>"]))

    git_section = ""
    if git_mode:
        git_section = """
You are working inside a REAL, EXISTING git repository, on a dedicated branch \
created just for this task — not a prepared toy folder. There is no pre-staged \
context: use list_files and search_files to orient yourself in the actual \
codebase before assuming you understand its structure. Use git_status to see \
what you've touched so far, and git_diff to review your own changes before \
finishing. Be conservative — only change what the task actually requires.
"""

    return f"""You are a careful coding agent. You solve one task at a time \
by writing code and tests in {language}, then running the tests, then fixing \
any failures, repeating until the tests pass.

The current language for this task is: {language}. Test files/conventions \
should match what `{test_cmd}` expects to discover and run.
{git_section}
You have a set of tools available (offered to you natively — call exactly one \
per turn). General guidance on using them well:
- If the task gives you existing files to start from, use list_files and \
read_file first to see what's there before writing anything.
- For anything beyond a trivial one-file task, use update_plan early to lay \
out your steps as a markdown checklist, and update it as you go. This is how \
you stay coherent across many turns — the plan is always shown to you in full \
even when older history gets condensed.
- Use run_scratch to investigate or debug BEFORE deciding on a fix, if you're \
not sure what's wrong. It does not count as testing.
- Use update_memory ONLY for durable, project-level facts that should persist \
into FUTURE tasks in this same project — not task-specific details.
- Use ask_user ONLY when something is genuinely ambiguous and you can't resolve \
it yourself through investigation. You have a limited number of questions per run.
- Use web_search for a quick lookup; use web_research instead when the answer is \
time-sensitive or numeric and worth cross-referencing multiple sources for.
- Prefer edit_file over write_file when changing an existing file.
- Use run_tests after changes to check your work. Do not call finish until the \
most recent run_tests result was a full pass.
- If tests fail, fix the actual implementation — not by weakening or deleting \
test cases.
- Some files may be marked protected (e.g. a pre-written test suite) — \
attempting to write or edit them will be rejected.
- Your final solution will be independently reviewed before being accepted — \
make sure it genuinely and robustly solves the task, not just narrowly passes \
the literal test cases.
"""


CRITIC_SYSTEM_PROMPT = """You are an independent code reviewer. You will be shown \
a task, the agent's changes, and the last test run output. Your job is to judge \
whether the implementation GENUINELY and ROBUSTLY solves the task — not just \
whether the test suite happened to exit 0. Submit your verdict using the \
submit_critic_verdict tool.

Specifically check for:
- Tests that were weakened, deleted, or trivialized to force a pass instead of \
fixing the underlying implementation.
- Hardcoded outputs that satisfy only the specific test inputs shown, rather than \
a general solution.
- Obvious unhandled edge cases the task description implies should work.
- Any mismatch between what the task asked for and what was actually built.

CRITICAL — ground every claim in evidence, do not invent requirements:
- Base your verdict strictly on what the task ACTUALLY says. If the task gives \
example outputs, those examples ARE the spec — don't reject for not matching some \
other format you'd prefer instead.
- A passing, non-trivial test suite is real evidence of correctness. If you reject \
a solution whose tests pass, you must cite a SPECIFIC concrete value (an actual \
input and the actual vs. expected output) that violates a literal requirement from \
the task — not a general impression that something "should" be different.
- Be reasonably strict but fair — do not reject for purely stylistic preferences.
"""

RESEARCH_SYSTEM_PROMPT = """You are a careful research analyst. You will be shown a \
question (as one or more differently-phrased search queries) and the actual content \
of several web pages found while searching for it. Submit your synthesized answer \
using the submit_research_answer tool.

- Read what each source actually says before concluding anything.
- If sources agree, that's grounds for higher confidence. If they disagree, are vague, \
or might be stale (e.g. anything that changes minute-to-minute, like a stock price or \
exchange rate), say so explicitly and lower your confidence accordingly.
- Never invent a number or fact that isn't actually supported by the sources shown. If \
none of the sources actually answer the question, say so — low confidence — rather \
than guessing.
- For anything time-sensitive, explicitly caveat that this reflects what public web \
sources showed at fetch time, not a live/authoritative feed.
"""


class TaskResult:
    def __init__(self, success: bool, iterations: int, summary: str, task_dir: Path):
        self.success = success
        self.iterations = iterations
        self.summary = summary
        self.task_dir = task_dir


def _gather_task_files(task_dir: Path) -> str:
    parts = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_dir() or p.name == "transcript.jsonl" or p.name.startswith("_scratch."):
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in p.relative_to(task_dir).parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        rel = p.relative_to(task_dir)
        parts.append(f"--- {rel} ---\n{text}")
    return "\n\n".join(parts)


def _run_critic(task_description: str, task_dir: Path, last_test_output: str, git_mode: bool):
    if git_mode:
        try:
            diff_text = git_diff(task_dir)
        except ToolError as e:
            diff_text = f"(could not get git diff: {e})"
        files_blob = f"Diff of changes made so far (git diff):\n{diff_text}"
    else:
        files_blob = _gather_task_files(task_dir)

    critic_messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task:\n{task_description}\n\n"
                f"{files_blob}\n\n"
                f"Last test run output:\n{last_test_output}"
            ),
        },
    ]
    return get_critic_verdict(critic_messages)


def _run_research(queries: list, sources: list):
    if not sources:
        return None
    blocks = []
    for s in sources:
        tag = "fetched page content" if s["fetched"] else "snippet only (fetch failed/skipped)"
        blocks.append(
            f"Source: {s['href']}\nFound via query: \"{s['query']}\"\n"
            f"Title: {s['title']}\n({tag})\nContent:\n{s['content']}"
        )
    sources_blob = "\n\n---\n\n".join(blocks)
    research_messages = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question (searched as): {queries}\n\nSources found:\n\n{sources_blob}"},
    ]
    return get_research_answer(research_messages)


def _build_messages(
    task_description: str,
    language: str,
    git_mode: bool,
    current_plan: str,
    project_memory_text: str,
    action_log: list,
    nudge_finish: bool = False,
) -> list:
    """
    Rebuilds the prompt fresh each turn from action_log, instead of an
    ever-growing appended list. Older steps get condensed to one line each;
    only the last MAX_RECENT_DETAILED_TURNS keep full assistant/tool message
    pairs (required for native tool calling — OpenAI-style history needs a
    tool_calls-bearing assistant message immediately followed by a matching
    tool-role message). Plan and project memory are always shown in full.
    """
    messages = [
        {"role": "system", "content": _build_system_prompt(language, git_mode)},
        {"role": "user", "content": f"Task:\n{task_description}"},
    ]

    if project_memory_text:
        messages.append({
            "role": "user",
            "content": f"Project memory — durable notes from earlier tasks in this project:\n{project_memory_text}",
        })

    if current_plan:
        messages.append({"role": "user", "content": f"Current plan (most up to date):\n{current_plan}"})

    recent_k = config.MAX_RECENT_DETAILED_TURNS
    if len(action_log) > recent_k:
        older, recent = action_log[:-recent_k], action_log[-recent_k:]
        lines = []
        for e in older:
            line = f"- iter {e['iteration']}: {e['action']}"
            if e.get("path"):
                line += f" {e['path']}"
            lines.append(f"{line} -> {e['observation_short']}")
        messages.append({
            "role": "user",
            "content": (
                "Condensed log of earlier steps in this task (older detail trimmed to save "
                "context — call read_file again if you need exact content):\n" + "\n".join(lines)
            ),
        })
    else:
        recent = action_log

    for e in recent:
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": e["tool_call_id"],
                "type": "function",
                "function": {"name": e["action"], "arguments": e["raw_arguments"]},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": e["tool_call_id"], "content": e["observation"]})

    if nudge_finish:
        messages.append({
            "role": "user",
            "content": (
                f"Reminder: you are close to this run's iteration limit ({config.MAX_ITERATIONS} "
                f"max) and your most recent test run passed. Unless you have a SPECIFIC, concrete "
                f"reason to keep investigating or improving things, call finish now — don't let a "
                f"working solution run out the clock while you keep re-verifying it."
            ),
        })

    return messages


def run_task(
    task_description: str,
    task_dir: Path,
    protected_files: Optional[Iterable[str]] = None,
    language: str = config.DEFAULT_LANGUAGE,
    memory_path: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
    git_mode: bool = False,
    branch_name: Optional[str] = None,
) -> TaskResult:
    task_dir = Path(task_dir)
    meta_dir = Path(meta_dir) if meta_dir else task_dir
    meta_dir.mkdir(parents=True, exist_ok=True)
    if not git_mode:
        task_dir.mkdir(parents=True, exist_ok=True)

    protected_files = set(protected_files or [])
    transcript_path = meta_dir / "transcript.jsonl"
    memory_path = Path(memory_path) if memory_path else Path(config.PROJECT_MEMORY_FILE)

    def log(entry: dict):
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    if git_mode:
        try:
            ensure_clean_working_tree(task_dir)
            if branch_name:
                branch_msg = create_and_checkout_branch(task_dir, branch_name)
                print(f"[git] {branch_msg}")
                log({"iteration": 0, "git_setup": branch_msg})
            else:
                print("[git] Skipping branch creation (--no-branch) — operating on current branch.")
                log({"iteration": 0, "git_setup": "no-branch mode, operating on current branch"})
        except ToolError as e:
            print(f"[git] ABORTED: {e}")
            log({"iteration": 0, "fatal_error": f"Git setup failed: {e}"})
            return TaskResult(False, 0, f"Git setup failed: {e}", task_dir)

    action_log: list = []
    current_plan = ""
    project_memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    if project_memory_text.strip():
        print(f"\n[memory] Loaded project memory from {memory_path.resolve()}:")
        print(project_memory_text.strip())
        print(
            "[memory] If this isn't relevant to the current task, delete that file or pass "
            "--memory with a different path.\n"
        )

    last_tests_passed = False
    last_test_output = ""
    consecutive_tool_errors = 0
    critic_rejections = 0
    ask_user_count = 0

    for i in range(1, config.MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {i} ---")
        remaining = config.MAX_ITERATIONS - i + 1
        nudge_finish = last_tests_passed and remaining <= config.FINISH_NUDGE_WINDOW
        if nudge_finish:
            print(f"[nudge]    {remaining} iterations left and tests are green — nudging toward finish.")

        messages = _build_messages(
            task_description, language, git_mode, current_plan, project_memory_text,
            action_log, nudge_finish,
        )

        try:
            result = get_next_action(messages)
        except LLMError as e:
            print(f"LLM error: {e}")
            log({"iteration": i, "fatal_llm_error": str(e)})
            return TaskResult(False, i, f"LLM failed to respond validly: {e}", task_dir)

        tool_name = result.name
        args = result.args

        print(f"[thought]  {args.thought}")
        print(f"[action]   {tool_name}")

        tool_error_this_turn = False
        entry = {
            "iteration": i,
            "thought": args.thought,
            "action": tool_name,
            "tool_call_id": result.tool_call_id,
            "raw_arguments": result.raw_arguments,
            "path": getattr(args, "path", None),
        }

        try:
            if tool_name == "write_file":
                obs = write_file(task_dir, args.path, args.content, protected_files)
                last_tests_passed = False
                entry["content_len"] = len(args.content or "")

            elif tool_name == "edit_file":
                obs = edit_file(task_dir, args.path, args.old_str, args.new_str, protected_files)
                last_tests_passed = False
                entry["old_str"] = args.old_str
                entry["new_str"] = args.new_str

            elif tool_name == "read_file":
                obs = read_file(task_dir, args.path)

            elif tool_name == "list_files":
                obs = list_files(task_dir)

            elif tool_name == "search_files":
                obs = search_files(task_dir, args.pattern)

            elif tool_name == "run_scratch":
                obs = run_scratch(task_dir, args.code, language)

            elif tool_name == "update_plan":
                current_plan = args.plan or ""
                (meta_dir / "PLAN.md").write_text(current_plan, encoding="utf-8")
                obs = "Plan updated and saved to PLAN.md."
                entry["plan"] = current_plan

            elif tool_name == "update_memory":
                obs = append_project_memory(memory_path, args.note)
                project_memory_text += f"- {args.note}\n"
                entry["memory_note"] = args.note

            elif tool_name == "git_status":
                obs = git_status(task_dir)

            elif tool_name == "git_diff":
                obs = git_diff(task_dir, args.path)

            elif tool_name == "web_search":
                obs = web_search(args.query)

            elif tool_name == "web_research":
                sources = gather_research_sources(args.queries)
                if not sources:
                    obs = "No search results found across any of the queries tried."
                else:
                    research_result = _run_research(args.queries, sources)
                    entry["research_confidence"] = research_result.confidence
                    entry["research_sources"] = [s["href"] for s in sources]
                    obs = (
                        f"Answer: {research_result.answer}\n"
                        f"Confidence: {research_result.confidence}\n"
                        f"Caveats: {research_result.caveats or '(none)'}\n"
                        f"Sources consulted: {[s['href'] for s in sources]}"
                    )

            elif tool_name == "fetch_url":
                obs = fetch_url(args.url)

            elif tool_name == "ask_user":
                if ask_user_count >= config.MAX_ASK_USER_CALLS:
                    obs = (
                        f"You've already asked {ask_user_count} question(s) — that's the limit "
                        f"for this run. Proceed using your best judgment instead of asking further."
                    )
                else:
                    ask_user_count += 1
                    print(f"\n[ask_user] The agent has a question for you:")
                    print(f"  {args.question}")
                    answer = input("Your answer: ")
                    obs = f"User answered: {answer}"
                    entry["question"] = args.question
                    entry["user_answer"] = answer

            elif tool_name == "run_tests":
                passed, output = run_tests(task_dir, language)
                last_tests_passed = passed
                last_test_output = output
                status = "PASSED" if passed else "FAILED"
                obs = f"Test run {status}.\nOutput:\n{output}"
                print(f"[tests]    {status}")

            elif tool_name == "finish":
                if not last_tests_passed:
                    obs = (
                        "You cannot finish yet — tests have not passed since your most recent "
                        "file change. Run run_tests and fix any failures first."
                    )
                else:
                    print("[finish]   Tests pass — sending to critic review...")
                    verdict = _run_critic(task_description, task_dir, last_test_output, git_mode)
                    entry["critic_approved"] = verdict.approved
                    entry["critic_reason"] = verdict.reasoning
                    entry["critic_concerns"] = verdict.concerns

                    if verdict.approved:
                        print(f"[critic]   APPROVED — {verdict.reasoning}")
                        if git_mode:
                            try:
                                commit_msg = git_commit_all(task_dir, f"Agent: {args.summary or 'task complete'}")
                                print(f"[git]      {commit_msg}")
                            except ToolError as e:
                                print(f"[git]      commit failed (changes remain uncommitted): {e}")
                        entry["observation"] = "Critic approved. Task complete."
                        entry["observation_short"] = "Critic approved."
                        action_log.append(entry)
                        log(entry)
                        return TaskResult(True, i, args.summary or "", task_dir)

                    critic_rejections += 1
                    print(f"[critic]   REJECTED ({critic_rejections}) — {verdict.reasoning}")
                    if critic_rejections >= config.MAX_CRITIC_REJECTIONS:
                        warn = (
                            f"Force-finished after {critic_rejections} critic rejections. "
                            f"Unresolved concerns: {verdict.concerns}"
                        )
                        print(f"[critic]   {warn}")
                        if git_mode:
                            try:
                                commit_msg = git_commit_all(
                                    task_dir, f"Agent: {args.summary or 'task complete'} (unreviewed)"
                                )
                                print(f"[git]      {commit_msg}")
                            except ToolError as e:
                                print(f"[git]      commit failed (changes remain uncommitted): {e}")
                        entry["observation"] = warn
                        entry["observation_short"] = warn[:150]
                        action_log.append(entry)
                        log(entry)
                        return TaskResult(True, i, f"{args.summary} (WARNING: {warn})", task_dir)

                    obs = (
                        f"Independent review rejected this solution: {verdict.reasoning} "
                        f"Concerns: {verdict.concerns}. Address these and run tests again "
                        f"before calling finish."
                    )
            else:
                obs = f"Unknown tool: {tool_name}"

        except ToolError as e:
            obs = f"ERROR: {e}"
            tool_error_this_turn = True
        except Exception as e:
            obs = (
                f"INTERNAL ERROR ({type(e).__name__}): {e}. This action failed unexpectedly — "
                f"try a different approach."
            )
            tool_error_this_turn = True
            print(f"[INTERNAL ERROR] {type(e).__name__}: {e}")

        consecutive_tool_errors = consecutive_tool_errors + 1 if tool_error_this_turn else 0

        print(f"[observe]  {obs[:300]}")
        entry["observation"] = obs
        entry["observation_short"] = obs[:150].replace("\n", " ")
        action_log.append(entry)
        log(entry)

        if consecutive_tool_errors >= config.MAX_CONSECUTIVE_TOOL_ERRORS:
            msg = (
                f"Aborted: {consecutive_tool_errors} consecutive tool errors — the model "
                f"appears stuck on the same mistake."
            )
            print(msg)
            log({"iteration": i, "fatal_error": msg})
            return TaskResult(False, i, msg, task_dir)

    return TaskResult(
        False,
        config.MAX_ITERATIONS,
        "Max iterations reached without passing tests."
        + (" Changes remain uncommitted on the task branch." if git_mode else ""),
        task_dir,
    )
