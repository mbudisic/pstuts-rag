#!/usr/bin/env python3
"""
Script to generate TODO.md by parsing all TODO-style comments using ruff.
Run with: uv run python generate_todo_md.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
TODO_MD_PATH = REPO_ROOT / "TODO.md"


def run_ruff_todos():
    """Run ruff to find TODO/FIXME comments in all .py files."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select=TD",
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running ruff: {e}")
        sys.exit(1)


def parse_ruff_output(output):
    """Parse ruff output into a list of TODOs."""
    todos = []
    for line in output.splitlines():
        # Example: app.py:251:1: TD001 TODO: ...
        parts = line.split(":", 3)
        if len(parts) == 4:
            file_path, line_no, col_no, rest = parts
            rest = rest.strip()
            if rest.startswith("TD"):  # e.g. TD001 TODO: ...
                todos.append((file_path.strip(), line_no.strip(), rest))
    return todos


def write_todo_md(todos):
    with open(TODO_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# 📝 TODOs in Codebase\n\n")
        if not todos:
            f.write("No TODOs found! 🎉\n")
            return
        for file_path, line_no, rest in todos:
            f.write(f"- `{file_path}:{line_no}`: {rest}\n")


def main():
    output = run_ruff_todos()
    todos = parse_ruff_output(output)
    write_todo_md(todos)
    print(f"Wrote {len(todos)} TODOs to {TODO_MD_PATH}")


if __name__ == "__main__":
    main()
