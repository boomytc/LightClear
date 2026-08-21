from __future__ import annotations

import py_compile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TASKS = (
    "speech_enhancement",
    "speech_separation",
    "speech_super_resolution",
    "target_speaker_extraction",
)
IMPORT_LINE = "from clearvoice import ClearVoice"


def load_clearvoice_name() -> str:
    cwd = Path.cwd().resolve()
    if cwd != MODULE_ROOT.resolve():
        raise AssertionError(f"cwd must be {MODULE_ROOT}, got {cwd}")

    from clearvoice import ClearVoice

    if ClearVoice.__name__ != "ClearVoice":
        raise AssertionError(f"expected ClearVoice.__name__ == 'ClearVoice', got {ClearVoice.__name__!r}")
    return ClearVoice.__name__


def compile_demos() -> list[str]:
    demo_dir = MODULE_ROOT / "demo"
    demo_files = sorted(demo_dir.glob("demo_*.py"))
    if not demo_files:
        raise AssertionError(f"no demo scripts in {demo_dir}")
    names: list[str] = []
    sources: list[str] = []
    for path in demo_files:
        py_compile.compile(str(path), doraise=True)
        text = path.read_text(encoding="utf-8")
        sources.append(text)
        names.append(path.name)
        if IMPORT_LINE not in text:
            raise AssertionError(f"{path.name} missing {IMPORT_LINE!r}")
    joined = "\n".join(sources)
    missing = [task for task in REQUIRED_TASKS if task not in joined]
    if missing:
        raise AssertionError(f"demo sources missing tasks: {missing}")
    return names


def main() -> None:
    name = load_clearvoice_name()
    demos = compile_demos()
    print(f"import_name={name}")
    print("demos=" + ",".join(demos))


if __name__ == "__main__":
    main()
