import argparse
import json
import shutil
import time
from pathlib import Path

import config
from orchestrator import run_task

# Master on/off switch for git integration (--repo mode, branch creation,
# git_status/git_diff actions, auto-commit on success). Set to False to
# disable real-repo mode entirely regardless of CLI flags — e.g. if you
# want to temporarily restrict the agent to seeded toy tasks only.
GIT_INTEGRATION_ENABLED = True


def load_workspace_config(base_dir: Path) -> dict:
    """
    Loads .agent_workspace.json from base_dir if present. Lets a project set
    language/memory/protect defaults once instead of retyping CLI flags every
    run. Returns {} if the file doesn't exist or fails to parse (logged, not fatal).
    """
    path = base_dir / config.WORKSPACE_CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded workspace config: {path}")
        return data
    except Exception as e:
        print(f"Warning: could not parse {path} ({e}) — ignoring it.")
        return {}


def write_workspace_config_template(base_dir: Path):
    path = base_dir / config.WORKSPACE_CONFIG_FILENAME
    if path.exists():
        print(f"{path} already exists — not overwriting. Delete it first if you want a fresh template.")
        return
    template = {
        "_readme": (
            "Per-project defaults for run_task.py. Any CLI flag you pass "
            "explicitly overrides the matching value here. Delete keys you "
            "don't want to set; this file is optional."
        ),
        "language": config.DEFAULT_LANGUAGE,
        "memory": config.PROJECT_MEMORY_FILE,
        "protect": [],
        "no_branch": False,
    }
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Wrote template workspace config to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Path to a .txt task description file")
    parser.add_argument(
        "--init-workspace",
        action="store_true",
        help="Write a template .agent_workspace.json to the cwd (or --repo target) and exit, "
             "without running a task.",
    )
    parser.add_argument(
        "--seed",
        help="Optional directory whose contents get copied into a fresh task dir before starting "
             "(mutually exclusive with --repo)",
    )
    parser.add_argument(
        "--repo",
        help="Path to an EXISTING git repo to operate on directly, in place — no seeding. "
             "Requires a clean working tree; creates a dedicated branch automatically.",
    )
    parser.add_argument(
        "--no-branch",
        action="store_true",
        help="(--repo mode only) skip automatic branch creation — agent edits your "
             "current branch directly. Clean-working-tree check still applies.",
    )
    parser.add_argument(
        "--protect",
        default=None,
        help="Comma-separated relative filenames the model may not overwrite (e.g. test_inventory.py)",
    )
    parser.add_argument(
        "--language",
        default=None,
        choices=list(config.LANGUAGES.keys()),
        help=f"Language for this task (default: {config.DEFAULT_LANGUAGE}, or workspace config)",
    )
    parser.add_argument(
        "--memory",
        default=None,
        help=(
            f"Path to the project memory file (default: {config.PROJECT_MEMORY_FILE} in cwd, "
            f"or workspace config). Use the SAME path across run_task.py calls for notes to persist."
        ),
    )
    args = parser.parse_args()

    workspace_base = Path(args.repo).resolve() if args.repo else Path(".").resolve()

    if args.init_workspace:
        write_workspace_config_template(workspace_base)
        return

    if not args.task:
        parser.error("--task is required (unless using --init-workspace)")

    if args.seed and args.repo:
        parser.error("--seed and --repo are mutually exclusive")

    if args.repo and not GIT_INTEGRATION_ENABLED:
        parser.error(
            "Git integration is disabled (GIT_INTEGRATION_ENABLED = False at the "
            "top of run_task.py). Set it to True to use --repo mode."
        )

    workspace_cfg = load_workspace_config(workspace_base)

    language = args.language or workspace_cfg.get("language") or config.DEFAULT_LANGUAGE
    memory_value = args.memory or workspace_cfg.get("memory")
    no_branch = args.no_branch or bool(workspace_cfg.get("no_branch", False))

    if args.protect:
        protected = [p.strip() for p in args.protect.split(",")]
    else:
        protected = list(workspace_cfg.get("protect", []))

    task_description = Path(args.task).read_text(encoding="utf-8")
    task_id = time.strftime("%Y%m%d-%H%M%S")
    started_epoch = time.time()

    if args.repo:
        task_dir = Path(args.repo).resolve()
        if not (task_dir / ".git").exists():
            parser.error(f"{task_dir} does not look like a git repo (no .git folder found)")
        meta_dir = task_dir / ".agent_meta" / task_id
        git_mode = True
        branch_name = None if no_branch else f"agent/{task_id}"
        print(f"Operating directly on repo: {task_dir}")
        if branch_name:
            print(f"Will create and switch to branch: {branch_name}")
        else:
            print("WARNING: --no-branch set — agent will edit your current branch directly.")
    else:
        task_dir = (Path("runs") / task_id).resolve()
        meta_dir = task_dir
        git_mode = False
        branch_name = None
        if args.seed:
            task_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(args.seed, task_dir, dirs_exist_ok=True)
            print(f"Seeded {args.seed} -> {task_dir}")

    memory_path = Path(memory_value) if memory_value else None

    print(f"Running task in: {task_dir} (language={language}, git_mode={git_mode})")
    result = run_task(
        task_description,
        task_dir,
        protected_files=protected,
        language=language,
        memory_path=memory_path,
        meta_dir=meta_dir,
        git_mode=git_mode,
        branch_name=branch_name,
    )

    finished_epoch = time.time()

    print("\n=== RESULT ===")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Summary: {result.summary}")
    print(f"Files in: {result.task_dir.resolve()}")
    print(f"Transcript: {(meta_dir / 'transcript.jsonl').resolve()}")
    if args.repo and branch_name:
        print(f"\nReview with: git -C \"{task_dir}\" log --oneline -5")
        print(f"Diff vs main: git -C \"{task_dir}\" diff main..{branch_name}")
        print(f"Discard with: git -C \"{task_dir}\" checkout main && git -C \"{task_dir}\" branch -D {branch_name}")

    # Housekeeping: a small summary file makes manage_runs.py possible without
    # re-parsing every transcript.jsonl just to know whether a run succeeded.
    summary_data = {
        "task_file": str(args.task),
        "success": result.success,
        "iterations": result.iterations,
        "summary": result.summary,
        "language": language,
        "git_mode": git_mode,
        "repo": str(args.repo) if args.repo else None,
        "task_dir": str(result.task_dir.resolve()),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_epoch)),
        "started_epoch": started_epoch,
        "finished_epoch": finished_epoch,
        "duration_seconds": round(finished_epoch - started_epoch, 1),
    }
    (meta_dir / "run_summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
