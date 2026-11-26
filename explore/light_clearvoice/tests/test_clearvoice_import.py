from __future__ import annotations

import ast
import py_compile
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TASKS = (
    "speech_enhancement",
    "speech_separation",
    "speech_super_resolution",
    "target_speaker_extraction",
)
REQUIRED_DEMOS_BY_BACKEND = {
    "cpu": {
        "demo_se.py",
        "demo_frcrn_se.py",
        "demo_sr.py",
        "demo_Numpy2Numpy.py",
        "demo_tensor2tensor.py",
    },
    "mps": {
        "demo_se.py",
        "demo_ss.py",
        "demo_sr.py",
        "demo_Numpy2Numpy.py",
    },
    "cuda": {
        "demo_se.py",
        "demo_frcrn_se.py",
        "demo_mossformergan_se.py",
        "demo_ss.py",
        "demo_sr.py",
        "demo_tse.py",
        "demo_Numpy2Numpy.py",
        "demo_tensor2tensor.py",
    },
}
EXPECTED_DEVICE_LITERALS = {
    "cpu": 'DEVICE = "cpu"',
    "mps": 'DEVICE = "mps"',
    "cuda": 'DEVICE = "cuda:0"',
}
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
    stray_top = sorted(demo_dir.glob("demo_*.py"))
    if stray_top:
        raise AssertionError(f"unexpected top-level demos in {demo_dir}: {[p.name for p in stray_top]}")

    names: list[str] = []
    sources: list[str] = []
    for backend, required_demos in REQUIRED_DEMOS_BY_BACKEND.items():
        backend_dir = demo_dir / backend
        demo_files = sorted(backend_dir.glob("demo_*.py"))
        if not demo_files:
            raise AssertionError(f"no demo scripts in {backend_dir}")
        backend_names: set[str] = set()
        for path in demo_files:
            py_compile.compile(str(path), doraise=True)
            text = path.read_text(encoding="utf-8")
            sources.append(text)
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
        missing = sorted(required_demos.difference(backend_names))
        if missing:
            raise AssertionError(f"missing {backend} demo scripts: {missing}")
        extra = sorted(backend_names.difference(required_demos))
        if extra:
            raise AssertionError(f"unexpected {backend} demo scripts: {extra}")

    joined = "\n".join(sources)
    missing_tasks = [task for task in REQUIRED_TASKS if task not in joined]
    if missing_tasks:
        raise AssertionError(f"demo sources missing tasks: {missing_tasks}")
    return names


def check_sfd_face_loader() -> str:
    path = (
        MODULE_ROOT
        / "third_party"
        / "clearvoice"
        / "models"
        / "av_mossformer2_tse"
        / "faceDetector"
        / "s3fd"
        / "__init__.py"
    )
    if not path.is_file():
        raise AssertionError(f"missing {path}")
    text = path.read_text(encoding="utf-8")
    if "gdown" in text or "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt" in text:
        raise AssertionError("s3fd loader must not download sfd_face.pth")
    if "FileNotFoundError" not in text:
        raise AssertionError("s3fd loader must fail locally when sfd_face.pth is missing")
    weight = path.with_name("sfd_face.pth")
    return "present" if weight.is_file() else "missing"


def main() -> None:
    name = load_clearvoice_name()
    demos = compile_demos()
    sfd_face = check_sfd_face_loader()
    print(f"import_name={name}")
    print(f"demo_count={len(demos)}")
    print("demos=" + ",".join(demos))
    print(f"sfd_face={sfd_face}")


if __name__ == "__main__":
    main()
