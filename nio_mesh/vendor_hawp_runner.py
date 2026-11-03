from __future__ import annotations

import sys
from pathlib import Path


def resolve_bundle_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_bundle_hawpv3(
    *,
    image_path: Path,
    checkpoint_path: Path,
    threshold: float,
    width: int,
    height: int,
    output_dir: Path,
    device: str,
) -> dict:
    vendor_root = Path(__file__).resolve().parent / "vendor_hawp"
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)

    from hawp.ssl.predict import run_prediction

    generated = run_prediction(
        images=[str(image_path)],
        ckpt=str(checkpoint_path),
        threshold=float(threshold),
        width=int(width),
        height=int(height),
        saveto=str(output_dir),
        device=device,
        ext="json",
        disable_show=True,
    )
    raw_json_path = generated[0] if generated else output_dir / f"{image_path.stem}.json"
    return {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "raw_json_path": str(raw_json_path),
    }
