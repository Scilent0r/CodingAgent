"""
Lists and cleans up old run folders under ./runs/ (toy/seeded tasks only —
--repo mode runs live inside the actual repo's .agent_meta/ and aren't
touched by this, since that's your project, not agent scratch space).

Usage:
    python manage_runs.py list
    python manage_runs.py clean --keep-last 10
    python manage_runs.py clean --older-than 14          (days)
    python manage_runs.py clean --failed-only
    python manage_runs.py clean --keep-last 10 --dry-run
    python manage_runs.py clean --keep-last 10 --yes      (skip confirmation)
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def find_run_summaries(runs_dir: Path) -> list:
    summaries = []
    if not runs_dir.exists():
        return summaries
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        summary_file = d / "run_summary.json"
        if not summary_file.exists():
            # Run from before this feature existed, or summary write failed.
            # Still list it (unknown status) so cleanup can still find it.
            summaries.append({
                "_dir": d,
                "started_at": "?",
                "started_epoch": d.stat().st_mtime,
                "success": None,
                "iterations": "?",
                "language": "?",
                "task_file": "?",
            })
            continue
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            data["_dir"] = d
            summaries.append(data)
        except Exception:
            continue
    return summaries


def status_label(success) -> str:
    if success is True:
        return "OK"
    if success is False:
        return "FAIL"
    return "?"


def cmd_list(args):
    summaries = find_run_summaries(Path("runs"))
    if not summaries:
        print("No runs found under ./runs/")
        return
    summaries.sort(key=lambda s: s.get("started_epoch", 0))
    print(f"{'Timestamp':<20} {'Status':<6} {'Iter':<5} {'Lang':<8} {'Task'}")
    print("-" * 80)
    for s in summaries:
        task_name = Path(str(s.get("task_file", "?"))).name
        print(
            f"{str(s.get('started_at', '?')):<20} "
            f"{status_label(s.get('success')):<6} "
            f"{str(s.get('iterations', '?')):<5} "
            f"{str(s.get('language', '?')):<8} "
            f"{task_name}"
        )
    print(f"\n{len(summaries)} run(s) total.")


def cmd_clean(args):
    summaries = find_run_summaries(Path("runs"))
    summaries.sort(key=lambda s: s.get("started_epoch", 0))

    if args.keep_last is not None:
        to_delete = summaries[: max(0, len(summaries) - args.keep_last)]
    elif args.older_than is not None:
        cutoff = time.time() - args.older_than * 86400
        to_delete = [s for s in summaries if s.get("started_epoch", 0) < cutoff]
    elif args.failed_only:
        to_delete = [s for s in summaries if s.get("success") is False]
    else:
        print("Specify one of: --keep-last N, --older-than DAYS, --failed-only")
        return

    if not to_delete:
        print("Nothing to delete.")
        return

    print(f"Will delete {len(to_delete)} run folder(s):")
    for s in to_delete:
        print(f"  {s['_dir'].name}  status={status_label(s.get('success'))}  started={s.get('started_at')}")

    if args.dry_run:
        print("\n(dry run — nothing deleted)")
        return

    if not args.yes:
        confirm = input(f"\nDelete these {len(to_delete)} folder(s)? [y/N]: ")
        if confirm.strip().lower() != "y":
            print("Aborted — nothing deleted.")
            return

    deleted = 0
    for s in to_delete:
        try:
            shutil.rmtree(s["_dir"])
            deleted += 1
        except Exception as e:
            print(f"Could not delete {s['_dir']}: {e}")

    print(f"Deleted {deleted}/{len(to_delete)} run folder(s).")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all runs with status, newest last")

    clean_parser = sub.add_parser("clean", help="Delete old run folders")
    clean_parser.add_argument("--keep-last", type=int, default=None, help="Keep only the N most recent runs")
    clean_parser.add_argument("--older-than", type=int, default=None, help="Delete runs older than N days")
    clean_parser.add_argument("--failed-only", action="store_true", help="Delete only runs that failed")
    clean_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted, don't delete")
    clean_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "clean":
        cmd_clean(args)


if __name__ == "__main__":
    main()
