"""Install htdemucs into /Users/boom/Model/MSS/htdemucs as a Demucs local repo.

Downloads only into a scratch folder under the center dir, converts to .th + yaml,
then deletes the scratch so weights never land in ~/.cache/huggingface.
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

CENTER_DIR = Path("/Users/boom/Model/MSS/htdemucs")
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


def find_snapshot() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--adefossez--HTDemucs" / "snapshots"
    if not hub.is_dir():
        return None
    snapshots = sorted(path for path in hub.iterdir() if path.is_dir())
    for snapshot in reversed(snapshots):
        yaml_path = snapshot / f"{MODEL_NAME}.yaml"
        if yaml_path.is_file():
            return snapshot
    return None


def download_snapshot(scratch: Path) -> Path:
    from huggingface_hub import snapshot_download

    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(scratch / "hf_home")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(scratch / "hf_home" / "hub")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(scratch / "snapshot"))
    return scratch / "snapshot"


def install() -> Path:
    CENTER_DIR.mkdir(parents=True, exist_ok=True)
    yaml_dest = CENTER_DIR / f"{MODEL_NAME}.yaml"
    source = find_snapshot()
    scratch = CENTER_DIR / ".download"
    created_scratch = False
    if source is None:
        source = download_snapshot(scratch)
        created_scratch = True

    shutil.copy2(source / f"{MODEL_NAME}.yaml", yaml_dest)
    bag = yaml.safe_load(yaml_dest.read_text(encoding="utf-8"))
    for sig in bag["models"]:
        safetensors_path = source / f"{sig}.safetensors"
        if not safetensors_path.is_file():
            raise FileNotFoundError(f"missing {safetensors_path}")
        export_th(safetensors_path, CENTER_DIR / f"{sig}.th")

    if created_scratch and scratch.exists():
        shutil.rmtree(scratch)
    return CENTER_DIR


if __name__ == "__main__":
    path = install()
    print(f"installed {MODEL_NAME} -> {path}")
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        print(f"  {child.name} {child.stat().st_size}")
