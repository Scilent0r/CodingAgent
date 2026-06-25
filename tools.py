import re
import subprocess
import requests
from pathlib import Path
from typing import Optional, Iterable

import config

EXCLUDED_DIR_NAMES = {
    "__pycache__", ".pytest_cache", "runs", ".git", ".agent_scratch",
    "node_modules", ".agent_meta",
}


class ToolError(Exception):
    pass


def _lang_config(language: str) -> dict:
    cfg = config.LANGUAGES.get(language)
    if not cfg:
        raise ToolError(
            f"Unsupported language '{language}'. Supported: {list(config.LANGUAGES.keys())}"
        )
    return cfg


def _safe_path(task_dir: Path, relative_path: str) -> Path:
    """
    Resolve a model-supplied relative path and make sure it can't escape
    the task directory (no ../.. tricks).
    """
    if relative_path is None:
        raise ToolError("path is required for this action")
    target = (task_dir / relative_path).resolve()
    if task_dir.resolve() not in target.parents and target != task_dir.resolve():
        raise ToolError(f"Path escapes sandbox directory: {relative_path}")
    return target


def _check_protected(relative_path: str, protected: Optional[Iterable[str]]):
    if protected and relative_path in set(protected):
        raise ToolError(
            f"'{relative_path}' is a protected file (e.g. a pre-written test "
            f"suite) and cannot be modified. Fix the implementation instead."
        )


def write_file(
    task_dir: Path,
    relative_path: str,
    content: str,
    protected: Optional[Iterable[str]] = None,
) -> str:
    _check_protected(relative_path, protected)
    target = _safe_path(task_dir, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content if content is not None else "", encoding="utf-8")
    return f"Wrote {len(content or '')} chars to {relative_path}"


def edit_file(
    task_dir: Path,
    relative_path: str,
    old_str: str,
    new_str: str,
    protected: Optional[Iterable[str]] = None,
) -> str:
    """
    Surgical find/replace. old_str must appear exactly once in the file.
    Cheaper and safer than rewriting the whole file for small fixes.
    """
    _check_protected(relative_path, protected)
    if not old_str:
        raise ToolError("old_str is required and cannot be empty for edit_file")
    target = _safe_path(task_dir, relative_path)
    if not target.exists():
        raise ToolError(f"{relative_path} does not exist — use write_file to create it first.")

    text = target.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        snippet = text[:1500]
        more = f" ...[{len(text) - 1500} more chars]" if len(text) > 1500 else ""
        raise ToolError(
            f"old_str not found in {relative_path}. It must match the file's "
            f"current content exactly (whitespace included). Here is the file's "
            f"ACTUAL current content so you don't need a separate read_file call:\n"
            f"---\n{snippet}{more}\n---"
        )
    if count > 1:
        raise ToolError(
            f"old_str appears {count} times in {relative_path} — it must be unique. "
            f"Include more surrounding context so it matches only one location."
        )

    new_text = text.replace(old_str, new_str or "")
    target.write_text(new_text, encoding="utf-8")
    return f"Edited {relative_path} (replaced {len(old_str)} chars with {len(new_str or '')} chars)"


def read_file(task_dir: Path, relative_path: str) -> str:
    target = _safe_path(task_dir, relative_path)
    if not target.exists():
        return f"ERROR: {relative_path} does not exist."
    return target.read_text(encoding="utf-8")


def _is_scratch_file(p: Path) -> bool:
    return p.name.startswith("_scratch.")


def list_files(task_dir: Path) -> str:
    paths = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_dir():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in p.relative_to(task_dir).parts):
            continue
        if p.name == "transcript.jsonl" or _is_scratch_file(p):
            continue
        paths.append(str(p.relative_to(task_dir)))
    return "\n".join(paths) if paths else "(no files yet)"


def search_files(task_dir: Path, pattern: str) -> str:
    """
    Regex search across all text files in the task directory.
    Returns matches as `path:line: matched_line_content`.
    """
    if not pattern:
        raise ToolError("pattern is required for search_files")
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"Invalid regex pattern: {e}")

    hits = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_dir() or p.name == "transcript.jsonl" or _is_scratch_file(p):
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in p.relative_to(task_dir).parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = p.relative_to(task_dir)
                hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                if len(hits) >= config.MAX_SEARCH_RESULTS:
                    hits.append(f"... truncated at {config.MAX_SEARCH_RESULTS} matches")
                    return "\n".join(hits)

    return "\n".join(hits) if hits else "No matches found."


def run_scratch(task_dir: Path, code: str, language: str = config.DEFAULT_LANGUAGE) -> str:
    """
    Runs arbitrary scratch code in the given language, cwd'd to the task dir
    so it can import/require files already written there. For investigation/
    debugging only — not part of the solution and not a substitute for run_tests.

    IMPORTANT: the scratch file lives directly in task_dir (named _scratch.<ext>),
    NOT in a subfolder. Python sets sys.path[0] (and Node resolves relative
    requires) based on the SCRIPT'S OWN directory, not the subprocess cwd — so
    if this lived in a subfolder, `import sibling_module` would fail even
    though cwd is correct. Keeping it at task_dir root makes import/require
    behave exactly like it would for the real solution files.
    """
    if not code:
        raise ToolError("code is required for run_scratch")

    lang_cfg = _lang_config(language)
    scratch_file = task_dir / f"_scratch.{lang_cfg['extension']}"
    scratch_file.write_text(code, encoding="utf-8")

    cmd = [part.format(file=str(scratch_file.resolve())) for part in lang_cfg["run_command"]]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=config.SCRATCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Scratch script timed out after {config.SCRATCH_TIMEOUT_SECONDS}s."
    except FileNotFoundError as e:
        return f"Could not run scratch script — interpreter not found ({e}). Is {language} installed and on PATH?"

    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    output = output[: config.MAX_OUTPUT_CHARS]
    if output.strip():
        return f"Exit code {result.returncode}.\nOutput:\n{output}"
    return f"Exit code {result.returncode}. (no output)"


def run_tests(task_dir: Path, language: str = config.DEFAULT_LANGUAGE) -> tuple[bool, str]:
    """
    Runs the test suite for the given language, scoped to task_dir.
    Returns (passed, combined_output).
    """
    lang_cfg = _lang_config(language)
    try:
        result = subprocess.run(
            lang_cfg["test_command"],
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=config.TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"Tests timed out after {config.TEST_TIMEOUT_SECONDS}s "
            f"(possible infinite loop in the code under test)."
        )
    except FileNotFoundError as e:
        return False, (
            f"Could not run test command for language '{language}' ({e}). "
            f"Is the required tool installed and on PATH?"
        )

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    output = output[: config.MAX_OUTPUT_CHARS]
    passed = result.returncode == 0
    return passed, output


def append_project_memory(memory_path: Path, note: str) -> str:
    """
    Appends a durable, project-level note. Persists across separate
    run_task.py invocations (unlike PLAN.md, which lives inside one task's
    own runs/<timestamp> folder).
    """
    if not note:
        raise ToolError("note is required for update_memory")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(memory_path, "a", encoding="utf-8") as f:
        f.write(f"- {note}\n")
    return f"Saved to project memory ({memory_path}): {note}"


# --- Git tools (real-repo mode) ---

def _run_git(repo_dir: Path, args: list, timeout: int = 20) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git"] + args,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise ToolError("git executable not found. Is Git installed and on PATH?")
    except subprocess.TimeoutExpired:
        raise ToolError(f"git {' '.join(args)} timed out after {timeout}s")


def git_status(repo_dir: Path) -> str:
    result = _run_git(repo_dir, ["status", "--porcelain", "-b"])
    if result.returncode != 0:
        raise ToolError(f"git status failed: {result.stderr.strip()}")
    return result.stdout.strip() or "(clean — no changes)"


def git_diff(repo_dir: Path, relative_path: Optional[str] = None) -> str:
    args = ["diff"]
    if relative_path:
        args.append(relative_path)
    result = _run_git(repo_dir, args)
    if result.returncode != 0:
        raise ToolError(f"git diff failed: {result.stderr.strip()}")
    out = result.stdout
    out = out[: config.MAX_OUTPUT_CHARS]
    return out if out.strip() else "(no uncommitted changes)"


def ensure_clean_working_tree(repo_dir: Path):
    if not (repo_dir / ".git").exists():
        raise ToolError(f"{repo_dir} does not look like a git repo (no .git folder).")
    result = _run_git(repo_dir, ["status", "--porcelain"])
    if result.returncode != 0:
        raise ToolError(f"git status failed: {result.stderr.strip()}")
    if result.stdout.strip():
        raise ToolError(
            "Repo has uncommitted changes. Commit or stash them before running the "
            "agent here, so its changes stay cleanly isolated on their own branch."
        )


def create_and_checkout_branch(repo_dir: Path, branch_name: str) -> str:
    result = _run_git(repo_dir, ["checkout", "-b", branch_name])
    if result.returncode != 0:
        raise ToolError(f"Could not create branch '{branch_name}': {result.stderr.strip()}")
    return f"Created and switched to branch '{branch_name}'"


def git_commit_all(repo_dir: Path, message: str) -> str:
    add_result = _run_git(repo_dir, ["add", "-A"])
    if add_result.returncode != 0:
        raise ToolError(f"git add failed: {add_result.stderr.strip()}")
    commit_result = _run_git(repo_dir, ["commit", "-m", message or "Agent task complete"])
    if commit_result.returncode != 0:
        # Most commonly: nothing to commit. Not fatal — surface as info.
        return f"git commit: {commit_result.stdout.strip()} {commit_result.stderr.strip()}".strip()
    return commit_result.stdout.strip()


# --- Web search / doc lookup ---

def web_search(query: str) -> str:
    """
    DuckDuckGo text search via the 'ddgs' package (no API key required).
    Returns only titles/URLs/snippets — use fetch_url to read full content.
    """
    if not config.ENABLE_WEB_TOOLS:
        raise ToolError("Web tools are disabled (set ENABLE_WEB_TOOLS = True in config.py to enable).")
    if not query:
        raise ToolError("query is required for web_search")
    try:
        from ddgs import DDGS
    except ImportError:
        raise ToolError("The 'ddgs' package is not installed. Run: pip install ddgs")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=config.MAX_WEB_RESULTS))
    except Exception as e:
        raise ToolError(f"Web search failed: {e}")

    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "(no title)")
        href = r.get("href", "")
        body = (r.get("body", "") or "")[:200]
        lines.append(f"- {title}\n  {href}\n  {body}")
    return "\n".join(lines)


def gather_research_sources(queries: list) -> list:
    """
    Runs multiple search query variations, dedupes by URL, and fetches actual
    page content (not just snippets) for the top unique results. Falls back
    to the search snippet if a fetch fails. Returns a list of dicts:
    {query, title, href, content, fetched: bool}. Purely mechanical — no LLM
    involved here; synthesis/reasoning over these sources happens elsewhere.
    """
    if not config.ENABLE_WEB_TOOLS:
        raise ToolError("Web tools are disabled (set ENABLE_WEB_TOOLS = True in config.py to enable).")
    if not queries:
        raise ToolError("queries is required for web_research (provide 2-3 search query variations)")
    try:
        from ddgs import DDGS
    except ImportError:
        raise ToolError("The 'ddgs' package is not installed. Run: pip install ddgs")

    queries = list(queries)[: config.RESEARCH_MAX_QUERIES]
    seen_urls = set()
    found = []

    with DDGS() as ddgs:
        for q in queries:
            try:
                results = list(ddgs.text(q, max_results=config.RESEARCH_RESULTS_PER_QUERY))
            except Exception:
                continue
            for r in results:
                href = r.get("href", "")
                if not href or href in seen_urls:
                    continue
                seen_urls.add(href)
                found.append({
                    "query": q,
                    "title": r.get("title", "(no title)"),
                    "href": href,
                    "snippet": (r.get("body", "") or "")[:300],
                })

    if not found:
        return []

    sources = []
    for i, r in enumerate(found):
        if i < config.RESEARCH_MAX_FETCHES:
            try:
                content = fetch_url(r["href"])
                fetched = True
            except ToolError:
                content = r["snippet"]
                fetched = False
        elif i < config.RESEARCH_MAX_FETCHES + config.RESEARCH_SNIPPET_FALLBACK_COUNT:
            content = r["snippet"]
            fetched = False
        else:
            break
        sources.append({
            "query": r["query"],
            "title": r["title"],
            "href": r["href"],
            "content": content[: config.RESEARCH_SOURCE_CHAR_LIMIT],
            "fetched": fetched,
        })

    return sources


def fetch_url(url: str) -> str:
    """
    Fetches a URL and returns its readable text content (HTML stripped of
    tags/scripts/styles). For reading documentation pages found via web_search.
    """
    if not config.ENABLE_WEB_TOOLS:
        raise ToolError("Web tools are disabled (set ENABLE_WEB_TOOLS = True in config.py to enable).")
    if not url:
        raise ToolError("url is required for fetch_url")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError("url must start with http:// or https://")

    try:
        resp = requests.get(
            url,
            timeout=config.WEB_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0 (compatible; local-coding-agent/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ToolError(f"Failed to fetch {url}: {e}")

    # requests defaults to ISO-8859-1 when a server doesn't declare a charset,
    # which silently mangles UTF-8 pages (e.g. "—" becomes "â"). Let it sniff
    # the actual encoding from content instead of trusting a missing/wrong header.
    if resp.encoding is None or resp.encoding.lower() in ("iso-8859-1", "ascii"):
        resp.encoding = resp.apparent_encoding

    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type.lower() or "<html" in resp.text[:300].lower():
        text = _html_to_text(resp.text)
    else:
        text = resp.text

    text = text[: config.WEB_FETCH_MAX_CHARS]
    return text if text.strip() else "(empty response)"


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ToolError("The 'beautifulsoup4' package is not installed. Run: pip install beautifulsoup4")

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        # Tag-name stripping above misses framework-specific wrappers (Sphinx's
        # #sphinxsidebar, MDN's breadcrumbs, etc.) that aren't <nav>/<footer> but
        # are still just navigation chrome, not content.
        noise_patterns = ("sidebar", "toc", "nav", "breadcrumb", "menu", "footer", "header", "related")
        for tag in soup.find_all(True):
            # Some malformed real-world HTML makes html.parser produce tags
            # with attrs=None instead of {} — guard against that explicitly
            # rather than calling tag.get(), which crashes on attrs=None.
            attrs = getattr(tag, "attrs", None) or {}
            classes = attrs.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            attrs_text = f"{attrs.get('id', '')} {' '.join(classes)}".lower()
            if any(p in attrs_text for p in noise_patterns):
                tag.decompose()

        # Prefer an actual content container over the whole page (which still
        # has page furniture even after the stripping above).
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.find(id="main")
        container = main if main else soup

        text = container.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except ToolError:
        raise
    except Exception as e:
        # A parsing quirk on some arbitrary external page should fail this
        # one tool call, not crash the whole run.
        raise ToolError(f"Could not parse page content ({type(e).__name__}: {e})")
