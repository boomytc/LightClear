from __future__ import annotations

import ast
import py_compile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_LINE = "from inworld import voice_isolation"
DEVICE_LINE = 'DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")'
REQUIRED_DEMOS = {
    "demo_offline_single.py",
    "demo_offline_options.py",
    "demo_batch_many.py",
    "demo_streaming_realtime.py",
    "demo_streaming_knobs.py",
    "demo_lowlevel_network.py",
}


def check_public_import() -> str:
    cwd = Path.cwd().resolve()
    if cwd != MODULE_ROOT.resolve():
        raise AssertionError(f"cwd must be {MODULE_ROOT}, got {cwd}")

    from inworld import voice_isolation

    if voice_isolation.VoiceIsolator.__name__ != "VoiceIsolator":
        raise AssertionError(
            f"expected VoiceIsolator, got {voice_isolation.VoiceIsolator.__name__!r}"
        )
    if voice_isolation.__version__ != "1.0.0":
        raise AssertionError(
            f"expected voice_isolation.__version__ == '1.0.0', got {voice_isolation.__version__!r}"
        )
    expected_path = (MODULE_ROOT / "third_party" / "inworld" / "voice_isolation" / "__init__.py").resolve()
    actual_path = Path(voice_isolation.__file__).resolve()
    if actual_path != expected_path:
        raise AssertionError(f"expected vendored package at {expected_path}, got {actual_path}")
    return voice_isolation.VoiceIsolator.__name__


def check_missing_checkpoint_fails_fast() -> None:
    from inworld import voice_isolation

    missing = MODULE_ROOT / "nonexistent_checkpoint.safetensors"
    try:
        voice_isolation.VoiceIsolator(str(missing), device="cpu")
    except ValueError:
        return
    raise AssertionError("missing local checkpoint must raise ValueError immediately")


def compile_demos() -> list[str]:
    demo_dir = MODULE_ROOT / "demo"
    for stale_dir in ("cpu", "mps", "cuda"):
        if (demo_dir / stale_dir).exists():
            raise AssertionError(f"stale hardware demo directory still exists: {demo_dir / stale_dir}")

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
        if DEVICE_LINE not in text:
            raise AssertionError(f"{path.name} missing {DEVICE_LINE!r}")
        if "torch.backends.mps.is_available()" not in text:
            raise AssertionError(f"{path.name} must support MPS in auto/mps selection")
        if "LIGHTCLEAR_MODEL_ROOT" not in text or "inworld-voice-isolator" not in text:
            raise AssertionError(f"{path.name} must resolve central weights via LIGHTCLEAR_MODEL_ROOT")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise AssertionError(
                    f"{path.name} must not define functions (found {node.name!r})"
                )

    found_names = set(names)
    missing = sorted(REQUIRED_DEMOS.difference(found_names))
    if missing:
        raise AssertionError(f"missing demo scripts: {missing}")
    extra = sorted(found_names.difference(REQUIRED_DEMOS))
    if extra:
        raise AssertionError(f"unexpected demo scripts: {extra}")
    return names


def main() -> None:
    name = check_public_import()
    check_missing_checkpoint_fails_fast()
    demos = compile_demos()
    print(f"import_name={name}")
    print(f"demo_count={len(demos)}")
    print("demos=" + ",".join(demos))


if __name__ == "__main__":
    main()
