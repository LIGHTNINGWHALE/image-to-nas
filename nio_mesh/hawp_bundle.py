from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import hawp as _base
from .vendor_hawp_runner import resolve_bundle_device, run_bundle_hawpv3


HawpConfig = _base.HawpConfig


def _ready_paths(hawp_python: str | None, hawp_repo: str | None, checkpoints_dir: str | None) -> tuple[bool, str, Path | None, Path | None, Path | None]:
    resolved_repo = Path(hawp_repo).resolve() if hawp_repo else None
    resolved_python = Path(hawp_python).resolve() if hawp_python else None
    resolved_checkpoints = _base.discover_checkpoints_dir(checkpoints_dir, resolved_repo)
    if resolved_python is None or not resolved_python.exists():
        return False, "missing --hawp-python", resolved_repo, resolved_python, resolved_checkpoints
    if resolved_repo is None or not resolved_repo.exists():
        return False, "missing --hawp-repo", resolved_repo, resolved_python, resolved_checkpoints
    if resolved_checkpoints is None:
        return False, "missing --hawp-checkpoints-dir", resolved_repo, resolved_python, resolved_checkpoints
    return True, "ok", resolved_repo, resolved_python, resolved_checkpoints


def run_hawp_sweep(
    image_name: str,
    mesh_input_png: Path,
    mesh_input_rgb: np.ndarray,
    output_dir: Path,
    hawp_python: str | None,
    hawp_repo: str | None,
    checkpoints_dir: str | None,
    device: str = "auto",
    profile: str = "extreme",
    runner: str = "all",
    thresholds: tuple[float, ...] | None = None,
    min_width: int | None = None,
    checkpoint_names: tuple[str, ...] | None = None,
    input_variants: tuple[str, ...] | None = None,
    widths: tuple[int, ...] | None = None,
) -> list[dict]:
    mesh_input_png = Path(mesh_input_png).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _base.default_hawp_configs_for_profile(profile)
    if runner in {"hawpv2", "hawpv3"}:
        configs = [config for config in configs if config.runner == runner]
    if thresholds:
        allowed_thresholds = {round(float(value), 6) for value in thresholds}
        configs = [config for config in configs if round(float(config.threshold), 6) in allowed_thresholds]
    if min_width is not None:
        configs = [config for config in configs if int(config.width) >= int(min_width)]
    if checkpoint_names:
        allowed_checkpoints = {str(value).strip() for value in checkpoint_names}
        configs = [config for config in configs if config.checkpoint_name in allowed_checkpoints]
    if input_variants:
        allowed_variants = {str(value).strip() for value in input_variants}
        configs = [config for config in configs if config.input_variant in allowed_variants]
    if widths:
        allowed_widths = {int(value) for value in widths}
        configs = [config for config in configs if int(config.width) in allowed_widths]

    available_input_variants = _base._build_hawp_input_variants(
        mesh_input_png,
        mesh_input_rgb,
        output_dir / "_inputs",
        profile=profile,
    )
    ready, reason, resolved_repo, resolved_python, resolved_checkpoints = _ready_paths(
        hawp_python,
        hawp_repo,
        checkpoints_dir,
    )
    resolved_device = resolve_bundle_device(device)
    results: list[dict] = []

    for config in configs:
        config_dir = output_dir / config.slug
        config_dir.mkdir(parents=True, exist_ok=True)
        summary_path = config_dir / "summary.json"
        variant = available_input_variants[config.input_variant]
        variant_input_path = Path(variant["image_path"]).resolve()
        checkpoint_path = (
            resolved_checkpoints / config.checkpoint_name if resolved_checkpoints is not None else Path(config.checkpoint_name)
        )
        result = {
            "image_name": image_name,
            "config": asdict(config),
            "mesh_input_png": str(mesh_input_png),
            "status": "skipped",
            "reason": reason,
            "summary_path": str(summary_path),
            "resolved_hawp_python": str(resolved_python) if resolved_python else None,
            "resolved_hawp_repo": str(resolved_repo) if resolved_repo else None,
            "resolved_checkpoints_dir": str(resolved_checkpoints) if resolved_checkpoints else None,
            "resolved_device": resolved_device,
            "input_variant": config.input_variant,
            "input_variant_path": str(variant_input_path),
            "input_transform": variant["transform"],
            "runner": config.runner,
            "requested_runner": runner,
            "requested_thresholds": list(thresholds) if thresholds else None,
            "requested_min_width": int(min_width) if min_width is not None else None,
            "requested_checkpoint_names": list(checkpoint_names) if checkpoint_names else None,
            "requested_input_variants": list(input_variants) if input_variants else None,
            "requested_widths": list(widths) if widths else None,
            "profile": profile,
        }

        if ready and checkpoint_path.exists():
            if config.runner != "hawpv3":
                result["status"] = "skipped"
                result["reason"] = "unsupported runner in submission bundle"
            else:
                try:
                    prediction = run_bundle_hawpv3(
                        image_path=variant_input_path,
                        checkpoint_path=checkpoint_path,
                        threshold=config.threshold,
                        width=config.width,
                        height=config.height,
                        output_dir=config_dir,
                        device=resolved_device,
                    )
                    result["returncode"] = prediction["returncode"]
                    result["stdout"] = prediction["stdout"]
                    result["stderr"] = prediction["stderr"]
                    raw_json_path = Path(prediction["raw_json_path"])
                    output_ready = raw_json_path.exists()
                    if output_ready:
                        _base._remap_graph_to_full_image(raw_json_path, variant["transform"])
                        segments_path = config_dir / "filtered_segments.json"
                        overlay_path = config_dir / "overlay.png"
                        segment_count = _base._write_segments_json(raw_json_path, config.threshold, segments_path)
                        _base._render_segments_overlay(mesh_input_rgb, raw_json_path, config.threshold, overlay_path)
                        result["status"] = "ok"
                        result["reason"] = "ok"
                        result["raw_json_path"] = str(raw_json_path)
                        result["filtered_segments_path"] = str(segments_path)
                        result["overlay_path"] = str(overlay_path)
                        result["segment_count"] = segment_count
                    else:
                        result["status"] = "failed"
                        result["reason"] = "bundle hawp prediction did not emit json"
                except Exception as exc:
                    result["status"] = "failed"
                    result["reason"] = str(exc)
        elif ready:
            result["status"] = "skipped"
            result["reason"] = f"checkpoint not found: {checkpoint_path}"

        if result["status"] != "ok":
            cached_result = _base._reuse_cached_hawp_result(
                image_name=image_name,
                config_dir=config_dir,
                config=config,
                summary_path=summary_path,
                mesh_input_rgb=mesh_input_rgb,
                input_transform=variant["transform"],
                profile=profile,
                expected_device=resolved_device,
            )
            if cached_result is not None:
                result.update(cached_result)

        summary_path.write_text(json.dumps(result, indent=2))
        results.append(result)

    return results
