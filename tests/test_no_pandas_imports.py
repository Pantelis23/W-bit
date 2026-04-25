import os
import re
import sys

FORBIDDEN = re.compile(r"^\s*(import\s+pandas|from\s+pandas\s+import)\b")
SKIP_DIRS = {"__pycache__", ".git", "results", ".venv", "venv"}


def scan_repo():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    offending = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), repo_root)
            # Skip this test file and other tests to avoid false positives
            if rel_path.startswith("tests/"):
                if rel_path == "tests/test_no_pandas_imports.py":
                    continue
                # still scan other tests if desired? skip to reduce noise
                continue
            try:
                with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        if FORBIDDEN.search(line):
                            offending.append(rel_path)
                            break
            except OSError:
                continue
    return offending


def test_no_pandas_imports():
    offending = scan_repo()
    assert not offending, f"Found pandas imports in: {offending}"


def main():
    offending = scan_repo()
    if offending:
        print(f"Found pandas imports in: {offending}")
        return 1
    print("OK: no pandas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
