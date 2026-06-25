"""
Creates a small, disposable git repo with one deliberate bug, so you can test
`--repo` mode without pointing the agent at anything you actually care about.

Usage:
    python setup_test_repo.py
    (creates ./test_repo relative to wherever you run this)
"""
import subprocess
from pathlib import Path

REPO_DIR = Path("test_repo")

CALCULATOR_PY = '''def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def average(numbers):
    """Return the arithmetic mean of a list of numbers."""
    # BUG: integer division truncates instead of giving a true average.
    return sum(numbers) // len(numbers)
'''

TEST_CALCULATOR_PY = '''from calculator import add, subtract, average


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_average_whole_number():
    assert average([1, 2, 3]) == 2


def test_average_needs_decimal():
    assert average([1, 2, 4]) == pytest.approx(2.333, rel=1e-3)
'''

# Fix the import line above — needs pytest imported for pytest.approx
TEST_CALCULATOR_PY = "import pytest\n" + TEST_CALCULATOR_PY

README = "# Test Repo\n\nA tiny disposable repo for testing the local coding agent's --repo mode.\n"


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def main():
    if REPO_DIR.exists():
        print(f"{REPO_DIR} already exists — delete it first if you want a fresh one.")
        return

    REPO_DIR.mkdir(parents=True)
    (REPO_DIR / "calculator.py").write_text(CALCULATOR_PY, encoding="utf-8")
    (REPO_DIR / "test_calculator.py").write_text(TEST_CALCULATOR_PY, encoding="utf-8")
    (REPO_DIR / "README.md").write_text(README, encoding="utf-8")

    run(["git", "init"], cwd=REPO_DIR)
    run(["git", "config", "user.email", "test@example.com"], cwd=REPO_DIR)
    run(["git", "config", "user.name", "Test Setup"], cwd=REPO_DIR)
    run(["git", "checkout", "-b", "main"], cwd=REPO_DIR)
    run(["git", "add", "-A"], cwd=REPO_DIR)
    run(["git", "commit", "-m", "Initial commit (calculator.py has a bug)"], cwd=REPO_DIR)

    print(f"\nCreated {REPO_DIR.resolve()} with an initial commit on 'main'.")
    print("It has one bug: average() uses integer division (//) instead of true division (/).")
    print("\nNow try:")
    print(
        f'  python run_task.py --task example_tasks/repo_average_bug.txt '
        f'--repo {REPO_DIR} --protect test_calculator.py'
    )


if __name__ == "__main__":
    main()
