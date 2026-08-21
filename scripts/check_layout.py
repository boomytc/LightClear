from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOT_PATHS = ("pyproject.toml", "demo", "third_party", "models")
FORBIDDEN_SUFFIXES = (".pth", ".bk", ".pkf")
SKIP_PARTS = {"third_party", ".venv", "__pycache__", ".git"}


def git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def iter_tracked_files(suffixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for relative in git_ls_files():
        path = REPO_ROOT / relative
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return files


def main() -> None:
    for name in FORBIDDEN_ROOT_PATHS:
        path = REPO_ROOT / name
        if path.exists():
            fail(f"repo-root {name} still exists")

    tracked = git_ls_files()
    for relative in tracked:
        if relative.endswith(FORBIDDEN_SUFFIXES):
            fail(f"tracked weight/backup still present: {relative}")

    repo_third_party = str(REPO_ROOT / "third_party")
    for path in iter_tracked_files((".py",)):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative == "scripts/check_layout.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "parents[3]" in text:
            fail(f"{relative} still uses parents[3] (old repo-root launch)")
        if repo_third_party in text:
            fail(f"{relative} references repo-root third_party")

    for path in iter_tracked_files((".md",)):
        text = path.read_text(encoding="utf-8")
        if "uvicorn products.speech_" in text:
            fail(f"{path.relative_to(REPO_ROOT)} still documents uvicorn products.speech_* launch")
        if 'uv pip install -e ".[web]"' in text or "uv pip install -e '.[web]'" in text:
            fail(f"{path.relative_to(REPO_ROOT)} still documents root extra web install")

    for relative in tracked:
        if relative == "scripts/check_layout.py":
            continue
        if relative.endswith((".py", ".md", ".yaml", ".yml", ".html", ".js", ".toml")):
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            if "clearvoice_samples" in text:
                fail(f"{relative} still mentions retired sample tree")

    print("layout_ok")
    print(f"tracked_files={len(tracked)}")
    print(f"repo_root={REPO_ROOT}")


if __name__ == "__main__":
    main()
