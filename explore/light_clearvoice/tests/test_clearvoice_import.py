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
MPS_SUPPORTED_DEMOS = {
    "demo_se.py",
    "demo_ss.py",
    "demo_sr.py",
    "demo_Numpy2Numpy.py",
}
CUDA_CPU_ONLY_DEMOS = {
    "demo_frcrn_se.py",
    "demo_mossformergan_se.py",
    "demo_tse.py",
    "demo_tensor2tensor.py",
}
REQUIRED_DEMOS = MPS_SUPPORTED_DEMOS | CUDA_CPU_ONLY_DEMOS
IMPORT_LINE = "from clearvoice import ClearVoice"
DEVICE_LINE = 'DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")'


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
    for stale_dir in ("cpu", "mps", "cuda"):
        if (demo_dir / stale_dir).exists():
            raise AssertionError(f"stale hardware demo directory still exists: {demo_dir / stale_dir}")

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
        if DEVICE_LINE not in text:
            raise AssertionError(f"{path.name} missing {DEVICE_LINE!r}")
        if path.name in MPS_SUPPORTED_DEMOS and "torch.backends.mps.is_available()" not in text:
            raise AssertionError(f"{path.name} must support MPS in auto/mps selection")
        if path.name in CUDA_CPU_ONLY_DEMOS and "torch.mps.synchronize()" in text:
            raise AssertionError(f"{path.name} must not enable MPS execution path")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise AssertionError(f"{path.name} must not define functions (found {node.name!r})")

    found_names = set(names)
    missing = sorted(REQUIRED_DEMOS.difference(found_names))
    if missing:
        raise AssertionError(f"missing demo scripts: {missing}")
    extra = sorted(found_names.difference(REQUIRED_DEMOS))
    if extra:
        raise AssertionError(f"unexpected demo scripts: {extra}")

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
