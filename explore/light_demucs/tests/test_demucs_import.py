from __future__ import annotations

import ast
import py_compile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_LINE = "from demucs.api import Separator"
REQUIRED_BACKENDS = ("cpu", "mps", "cuda")
REQUIRED_DEMOS = {
    "demo_vocals.py",
    "demo_stems.py",
    "demo_htdemucs_ft.py",
    "demo_htdemucs_6s.py",
}
EXPECTED_DEVICE_LITERALS = {
    "cpu": 'DEVICE = "cpu"',
    "mps": 'DEVICE = "mps"',
    "cuda": 'DEVICE = "cuda:0"',
}


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
    stray_top = sorted(demo_dir.glob("demo_*.py"))
    if stray_top:
        raise AssertionError(f"unexpected top-level demos in {demo_dir}: {[p.name for p in stray_top]}")

    names: list[str] = []
    for backend in REQUIRED_BACKENDS:
        backend_dir = demo_dir / backend
        demo_files = sorted(backend_dir.glob("demo_*.py"))
        if not demo_files:
            raise AssertionError(f"no demo scripts in {backend_dir}")
        backend_names: set[str] = set()
        for path in demo_files:
            py_compile.compile(str(path), doraise=True)
            text = path.read_text(encoding="utf-8")
            rel_name = f"{backend}/{path.name}"
            names.append(rel_name)
            backend_names.add(path.name)
            if IMPORT_LINE not in text:
                raise AssertionError(f"{rel_name} missing {IMPORT_LINE!r}")
            if EXPECTED_DEVICE_LITERALS[backend] not in text:
                raise AssertionError(f"{rel_name} must pin {EXPECTED_DEVICE_LITERALS[backend]!r}")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    raise AssertionError(f"{rel_name} must not define functions (found {node.name!r})")
            if path.name == "demo_htdemucs_ft.py" and 'MODEL_NAME = "htdemucs_ft"' not in text:
                raise AssertionError(f"{rel_name} must pin MODEL_NAME = \"htdemucs_ft\"")
            if path.name == "demo_htdemucs_6s.py":
                if 'MODEL_NAME = "htdemucs_6s"' not in text:
                    raise AssertionError(f"{rel_name} must pin MODEL_NAME = \"htdemucs_6s\"")
                if '"guitar"' not in text or '"piano"' not in text:
                    raise AssertionError(f"{rel_name} must write guitar and piano stems")
        missing = sorted(REQUIRED_DEMOS.difference(backend_names))
        if missing:
            raise AssertionError(f"missing {backend} demo scripts: {missing}")
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
    print(f"demo_count={len(demos)}")
    print("demos=" + ",".join(demos))
    print(f"installer={installer}")


if __name__ == "__main__":
    main()
