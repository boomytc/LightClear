"""Install one HTDemucs bag into a Demucs local repo.

Usage, from this module directory:

    .venv/bin/python scripts/install_model.py
    .venv/bin/python scripts/install_model.py htdemucs_ft
    .venv/bin/python scripts/install_model.py htdemucs_6s
    .venv/bin/python scripts/install_model.py htdemucs /path/to/dest

Default destination is this module's models/<model>.
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

from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name

KNOWN_MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s")


def hf_repo_id(model_name: str) -> str:
    return f"{DEFAULT_NAMESPACE}/{hf_repo_name(model_name)}"


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


def download_snapshot(scratch: Path, repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(scratch / "hf_home")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(scratch / "hf_home" / "hub")
    snapshot_download(repo_id=repo_id, local_dir=str(scratch / "snapshot"))
    return scratch / "snapshot"


def bag_signatures(yaml_path: Path) -> list[str]:
    bag = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    signatures = bag.get("models") or []
    if not signatures:
        raise ValueError(f"{yaml_path} has no models list")
    return [str(sig) for sig in signatures]


def repo_is_ready(dest: Path, model_name: str) -> bool:
    yaml_path = dest / f"{model_name}.yaml"
    if not yaml_path.is_file():
        return False
    try:
        signatures = bag_signatures(yaml_path)
    except ValueError:
        return False
    return all((dest / f"{sig}.th").is_file() for sig in signatures)


def install(dest: Path, model_name: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    yaml_dest = dest / f"{model_name}.yaml"
    scratch = dest / ".download"
    source = download_snapshot(scratch, hf_repo_id(model_name))

    yaml_source = source / f"{model_name}.yaml"
    if not yaml_source.is_file():
        raise FileNotFoundError(f"missing {yaml_source}")
    shutil.copy2(yaml_source, yaml_dest)
    for sig in bag_signatures(yaml_dest):
        safetensors_path = source / f"{sig}.safetensors"
        if not safetensors_path.is_file():
            raise FileNotFoundError(f"missing {safetensors_path}")
        export_th(safetensors_path, dest / f"{sig}.th")

    if scratch.exists():
        shutil.rmtree(scratch)
    return dest


def parse_args(argv: list[str]) -> tuple[str, Path]:
    if len(argv) >= 2:
        model_name = argv[1]
    else:
        model_name = "htdemucs"
    if model_name not in KNOWN_MODELS:
        known = ", ".join(KNOWN_MODELS)
        raise SystemExit(f"unknown model {model_name!r}, expected one of: {known}")
    if len(argv) >= 3:
        dest = Path(argv[2]).expanduser().resolve()
    else:
        dest = (MODULE_ROOT / "models" / model_name).resolve()
    if len(argv) >= 4:
        raise SystemExit("usage: install_model.py [htdemucs|htdemucs_ft|htdemucs_6s] [dest]")
    return model_name, dest


if __name__ == "__main__":
    model_name, dest = parse_args(sys.argv)
    path = dest if repo_is_ready(dest, model_name) else install(dest, model_name)
    print(f"installed {model_name} -> {path}")
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        print(f"  {child.name} {child.stat().st_size}")
