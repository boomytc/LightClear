"""Install htdemucs into a Demucs local repo.

Default destination is this module's models/htdemucs.
Pass another directory to install a second copy (for example a product).
Scratch downloads stay under that destination's .download and are deleted afterwards.
This script does not write ~/.cache/huggingface and does not copy into other modules.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

DEFAULT_DIR = MODULE_ROOT / "models" / "htdemucs"
HF_REPO_ID = "adefossez/HTDemucs"
MODEL_NAME = "htdemucs"


def export_th(safetensors_path: Path, dest_th: Path) -> None:
    import importlib

    import torch
    from safetensors import safe_open

    from demucs.hf import _decode_json, _unflatten_state

    with safe_open(str(safetensors_path), framework="pt") as file:
        metadata = file.metadata()
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    if "structure" in metadata:
        state = _unflatten_state(tensors, json.loads(metadata["structure"]))
    else:
        state = tensors
    module, name = metadata["klass"].rsplit(".", 1)
    klass = getattr(importlib.import_module(module), name)
    args = _decode_json(json.loads(metadata["args"]))
    kwargs = _decode_json(json.loads(metadata["kwargs"]))
    dest_th.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"klass": klass, "args": args, "kwargs": kwargs, "state": state},
        dest_th,
    )


def download_snapshot(scratch: Path) -> Path:
    from huggingface_hub import snapshot_download

    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(scratch / "hf_home")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(scratch / "hf_home" / "hub")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(scratch / "snapshot"))
    return scratch / "snapshot"


def repo_is_ready(dest: Path) -> bool:
    yaml_path = dest / f"{MODEL_NAME}.yaml"
    return yaml_path.is_file() and any(dest.glob("*.th"))


def install(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    yaml_dest = dest / f"{MODEL_NAME}.yaml"
    scratch = dest / ".download"
    source = download_snapshot(scratch)

    shutil.copy2(source / f"{MODEL_NAME}.yaml", yaml_dest)
    bag = yaml.safe_load(yaml_dest.read_text(encoding="utf-8"))
    for sig in bag["models"]:
        safetensors_path = source / f"{sig}.safetensors"
        if not safetensors_path.is_file():
            raise FileNotFoundError(f"missing {safetensors_path}")
        export_th(safetensors_path, dest / f"{sig}.th")

    if scratch.exists():
        shutil.rmtree(scratch)
    return dest


if __name__ == "__main__":
    dest = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_DIR.resolve()
    path = dest if repo_is_ready(dest) else install(dest)
    print(f"installed {MODEL_NAME} -> {path}")
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        print(f"  {child.name} {child.stat().st_size}")
