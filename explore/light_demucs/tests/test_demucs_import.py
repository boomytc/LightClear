from __future__ import annotations

import py_compile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_LINE = "from demucs.api import Separator"


def load_separator_name() -> str:
    cwd = Path.cwd().resolve()
    if cwd != MODULE_ROOT.resolve():
        raise AssertionError(f"cwd must be {MODULE_ROOT}, got {cwd}")

    from demucs.api import Separator
    import demucs

    if Separator.__name__ != "Separator":
        raise AssertionError(f"expected Separator.__name__ == 'Separator', got {Separator.__name__!r}")
    if demucs.__version__ != "4.1.0":
        raise AssertionError(f"expected demucs.__version__ == '4.1.0', got {demucs.__version__!r}")
    return Separator.__name__


def compile_demos() -> list[str]:
    demo_dir = MODULE_ROOT / "demo"
    demo_files = sorted(demo_dir.glob("demo_*.py"))
    if not demo_files:
        raise AssertionError(f"no demo scripts in {demo_dir}")
    names: list[str] = []
    for path in demo_files:
        py_compile.compile(str(path), doraise=True)
        text = path.read_text(encoding="utf-8")
        names.append(path.name)
        if IMPORT_LINE not in text:
            raise AssertionError(f"{path.name} missing {IMPORT_LINE!r}")
        if "def " in text:
            raise AssertionError(f"{path.name} must not define functions")
        if path.name == "demo_htdemucs_ft.py" and 'MODEL_NAME = "htdemucs_ft"' not in text:
            raise AssertionError("demo_htdemucs_ft.py must pin MODEL_NAME = \"htdemucs_ft\"")
        if path.name == "demo_htdemucs_6s.py":
            if 'MODEL_NAME = "htdemucs_6s"' not in text:
                raise AssertionError("demo_htdemucs_6s.py must pin MODEL_NAME = \"htdemucs_6s\"")
            if '"guitar"' not in text or '"piano"' not in text:
                raise AssertionError("demo_htdemucs_6s.py must write guitar and piano stems")
    required = {
        "demo_vocals.py",
        "demo_stems.py",
        "demo_htdemucs_ft.py",
        "demo_htdemucs_6s.py",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise AssertionError(f"missing demo scripts: {missing}")
    return names


def compile_installer() -> str:
    path = MODULE_ROOT / "scripts" / "install_model.py"
    stale = MODULE_ROOT / "scripts" / "install_htdemucs.py"
    if stale.exists():
        raise AssertionError("stale scripts/install_htdemucs.py still present")
    if not path.is_file():
        raise AssertionError(f"missing {path}")
    py_compile.compile(str(path), doraise=True)
    text = path.read_text(encoding="utf-8")
    for model_name in ("htdemucs", "htdemucs_ft", "htdemucs_6s"):
        if f'"{model_name}"' not in text:
            raise AssertionError(f"install_model.py missing {model_name!r}")
    return path.name


def main() -> None:
    name = load_separator_name()
    demos = compile_demos()
    installer = compile_installer()
    print(f"import_name={name}")
    print("demos=" + ",".join(demos))
    print(f"installer={installer}")


if __name__ == "__main__":
    main()
