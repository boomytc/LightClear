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
    print("demos=" + ",".join(demos))
    print(f"sfd_face={sfd_face}")


if __name__ == "__main__":
    main()
