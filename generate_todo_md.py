#!/usr/bin/env python3
"""
Script to generate TODO.md by grepping for TODO/FIXME/HACK comments in Python files.
Run with: uvx python generate_todo_md.py
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
TODO_MD_PATH = REPO_ROOT / "TODO.md"


def run_grep_todos():
    """Run grep to find TODO/FIXME/HACK comments in all .py files using shell=True."""
    try:
        cmd = r'grep -rn "^\s*#\s*\(TODO\|FIXME\|HACK\)" --include="*.py"  --exclude-dir=".venv" --exclude-dir=".git"  .'
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip().splitlines()
    except Exception as e:
        print(f"Error running grep: {e}")
        return []


def write_todo_md(todos):
    with open(TODO_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# 📝 TODOs in Codebase\n\n")
        if not todos or (len(todos) == 1 and todos[0] == ""):
            f.write("No TODOs found! 🎉\n")
            return
        for line in todos:
            f.write(f"- `{line}`\n")
        f.write("\nKeep up the great work! 🚀\n")


def main():
    todos = run_grep_todos()
    write_todo_md(todos)
    print(f"Wrote {len(todos)} TODOs to {TODO_MD_PATH}")


if __name__ == "__main__":
    main()
