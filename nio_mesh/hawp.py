from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


@dataclass(slots=True)
class HawpConfig:
    checkpoint_name: str
    threshold: float
    width: int
    height: int
    input_variant: str = "full_gray"
    runner: str = "hawpv3"

    @property
    def slug(self) -> str:
        stem = Path(self.checkpoint_name).stem
        return f"{stem}_{self.input_variant}_thr_{self.threshold:.2f}_{self.width}x{self.height}"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_hawp_repo(explicit_repo: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_repo:
        candidates.append(Path(explicit_repo))

    env_repo = Path.cwd() / "hawp-main"
    sibling_repo = _workspace_root() / "hawp-main"
    candidates.extend([env_repo, sibling_repo])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "hawp" / "ssl" / "predict.py").exists():
            return candidate
    return None


def discover_hawp_python(explicit_python: str | None = None, hawp_repo: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_python:
        candidates.append(Path(explicit_python))
    if hawp_repo is not None:
        candidates.append(hawp_repo / ".venv_linux" / "bin" / "python")
        candidates.append(hawp_repo / ".venv" / "bin" / "python")
        candidates.append(hawp_repo / "venv" / "bin" / "python")
        candidates.append(hawp_repo / ".venv" / "Scripts" / "python.exe")
        candidates.append(hawp_repo / "venv" / "Scripts" / "python.exe")

    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix.lower() == ".exe" and os.name != "nt":
                continue
            return candidate
    return None


def discover_checkpoints_dir(explicit_dir: str | None = None, hawp_repo: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    if hawp_repo is not None:
        candidates.append(hawp_repo / "checkpoints")
        candidates.append(hawp_repo)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        if list(candidate.glob("hawpv3-*.pth")) or list(candidate.glob("hawpv2-*.pth")):
            return candidate
    return None


def resolve_hawp_device(requested_device: str, hawp_python: Path | None) -> str:
    if requested_device != "auto" or hawp_python is None:
        return requested_device

    try:
        completed = subprocess.run(
            [
                str(hawp_python),
                "-c",
                "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "cpu"

    device = completed.stdout.strip().lower()
    return device if device in {"cuda", "mps", "cpu"} else "cpu"


def default_hawp_configs() -> list[HawpConfig]:
    return default_hawp_configs_for_profile("extreme")


def default_hawp_configs_for_profile(profile: str = "extreme") -> list[HawpConfig]:
    configs: list[HawpConfig] = []
    if profile == "baseline":
        for checkpoint_name in ("hawpv3-imagenet-03a84.pth", "hawpv3-fdc5487a.pth"):
            for threshold in (0.02, 0.03, 0.05):
                for input_variant, width, height in (
                    ("roi_sharp_context", 1792, 1024),
                    ("roi_sharp", 1792, 1024),
                    ("roi_sharp", 2048, 1152),
                    ("roi_sharp_tight", 2048, 1152),
                    ("roi_sharp_tight", 2304, 1280),
                ):
                    configs.append(HawpConfig(checkpoint_name, threshold, width, height, input_variant, "hawpv3"))
        for threshold in (0.02, 0.03, 0.05):
            for width, height in ((1536, 896), (1792, 1024), (2048, 1152)):
                configs.append(HawpConfig("hawpv2-edb9b23f.pth", threshold, width, height, "full_color", "hawpv2"))
            configs.append(HawpConfig("hawpv2-edb9b23f.pth", threshold, 1792, 1024, "roi_color_loose", "hawpv2"))
        return configs

    for threshold in (0.02, 0.03, 0.05):
        for width, height in ((1536, 896), (1792, 1024), (2048, 1152)):
            configs.append(HawpConfig("hawpv2-edb9b23f.pth", threshold, width, height, "full_color", "hawpv2"))
        configs.append(HawpConfig("hawpv2-edb9b23f.pth", threshold, 1792, 1024, "roi_color_loose", "hawpv2"))
        configs.append(HawpConfig("hawpv2-edb9b23f.pth", threshold, 2048, 1152, "roi_color_tight", "hawpv2"))
    return configs


def hawp_ready(hawp_python: str | None, hawp_repo: str | None, checkpoints_dir: str | None) -> tuple[bool, str]:
    repo_path = discover_hawp_repo(hawp_repo)
    python_path = discover_hawp_python(hawp_python, repo_path)
    checkpoints_path = discover_checkpoints_dir(checkpoints_dir, repo_path)

    if python_path is None:
        return False, "missing --hawp-python"
    if repo_path is None:
        return False, "missing --hawp-repo"
    if checkpoints_path is None:
        return False, "missing --hawp-checkpoints-dir"
    return True, "ok"


def _write_segments_json(raw_json_path: Path, threshold: float, save_path: Path) -> int:
    graph = json.loads(raw_json_path.read_text())
    vertices = np.array(graph["vertices"], dtype=float)
    edges = np.array(graph["edges"], dtype=int)
    weights = np.array(graph["edges-weights"], dtype=float)
    segments: list[dict] = []
    for edge, weight in zip(edges, weights):
        if weight < threshold:
            continue
        left = vertices[edge[0]].tolist()
        right = vertices[edge[1]].tolist()
        segments.append(
            {
                "x1": left[0],
                "y1": left[1],
                "x2": right[0],
                "y2": right[1],
                "score": float(weight),
            }
        )
    save_path.write_text(json.dumps(segments, indent=2))
    return len(segments)


def _identity_transform(width: int, height: int) -> dict:
    return {
        "offset_x": 0.0,
        "offset_y": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "full_width": int(width),
        "full_height": int(height),
    }


def _mesh_bbox(mesh_input_rgb: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.any(np.asarray(mesh_input_rgb) > 0, axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        height, width = mesh_input_rgb.shape[:2]
        return 0, 0, width, height

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    pad = max(24, int(round(max(x1 - x0, y1 - y0) * 0.12)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mesh_input_rgb.shape[1], x1 + pad)
    y1 = min(mesh_input_rgb.shape[0], y1 + pad)
    return x0, y0, x1, y1


def _enhance_mesh_gray(mesh_input_rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(mesh_input_rgb).astype(np.uint8), mode="RGB").convert("L")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.8, percent=240, threshold=2))
    image = image.point(lambda value: 0 if value < 10 else value)
    return image


def _enhance_mesh_rgb(mesh_input_rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(mesh_input_rgb).astype(np.uint8), mode="RGB")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.8, percent=240, threshold=2))
    return image


def _build_roi_variant(
    mesh_input_rgb: np.ndarray,
    variants: dict[str, dict],
    variants_dir: Path,
    *,
    name: str,
    pad_ratio: float,
    min_pad_px: int,
    target_long_side: int,
    grayscale: bool,
) -> None:
    height, width = mesh_input_rgb.shape[:2]
    mesh_mask = np.any(np.asarray(mesh_input_rgb) > 0, axis=2)
    ys, xs = np.where(mesh_mask)
    if xs.size == 0 or ys.size == 0:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        pad = max(min_pad_px, int(round(max(x1 - x0, y1 - y0) * pad_ratio)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(width, x1 + pad)
        y1 = min(height, y1 + pad)

    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)
    scale = max(1.0, target_long_side / max(crop_width, crop_height))
    scaled_width = max(1, int(round(crop_width * scale)))
    scaled_height = max(1, int(round(crop_height * scale)))
    crop_rgb = mesh_input_rgb[y0:y1, x0:x1]
    image = _enhance_mesh_gray(crop_rgb) if grayscale else _enhance_mesh_rgb(crop_rgb)
    image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    save_path = variants_dir / f"{name}.png"
    image.save(save_path)
    variants[name] = {
        "name": name,
        "image_path": save_path,
        "transform": {
            "offset_x": float(x0),
            "offset_y": float(y0),
            "scale_x": float(crop_width / scaled_width),
            "scale_y": float(crop_height / scaled_height),
            "full_width": int(width),
            "full_height": int(height),
            "crop_bbox": [int(x0), int(y0), int(x1), int(y1)],
            "upsampled_size": [int(scaled_width), int(scaled_height)],
        },
    }


def _build_hawp_input_variants(
    mesh_input_png: Path,
    mesh_input_rgb: np.ndarray,
    variants_dir: Path,
    profile: str = "extreme",
) -> dict[str, dict]:
    variants_dir.mkdir(parents=True, exist_ok=True)
    height, width = mesh_input_rgb.shape[:2]
    full_color_path = variants_dir / "full_color.png"
    Image.fromarray(np.asarray(mesh_input_rgb).astype(np.uint8), mode="RGB").save(full_color_path)
    variants: dict[str, dict] = {
        "full_gray": {
            "name": "full_gray",
            "image_path": mesh_input_png.resolve(),
            "transform": _identity_transform(width, height),
        },
        "full_color": {
            "name": "full_color",
            "image_path": full_color_path.resolve(),
            "transform": _identity_transform(width, height),
        }
    }
    if profile == "baseline":
        _build_roi_variant(
            mesh_input_rgb,
            variants,
            variants_dir,
            name="roi_sharp_context",
            pad_ratio=0.32,
            min_pad_px=160,
            target_long_side=1600,
            grayscale=True,
        )
        _build_roi_variant(
            mesh_input_rgb,
            variants,
            variants_dir,
            name="roi_sharp",
            pad_ratio=0.12,
            min_pad_px=24,
            target_long_side=1600,
            grayscale=True,
        )
        _build_roi_variant(
            mesh_input_rgb,
            variants,
            variants_dir,
            name="roi_sharp_tight",
            pad_ratio=0.05,
            min_pad_px=10,
            target_long_side=2560,
            grayscale=True,
        )
    _build_roi_variant(
        mesh_input_rgb,
        variants,
        variants_dir,
        name="roi_color_loose",
        pad_ratio=0.12,
        min_pad_px=24,
        target_long_side=1600,
        grayscale=False,
    )
    _build_roi_variant(
        mesh_input_rgb,
        variants,
        variants_dir,
        name="roi_color_tight",
        pad_ratio=0.04,
        min_pad_px=8,
        target_long_side=2200,
        grayscale=False,
    )
    return variants


def _remap_graph_to_full_image(raw_json_path: Path, transform: dict) -> None:
    if (
        float(transform.get("offset_x", 0.0)) == 0.0
        and float(transform.get("offset_y", 0.0)) == 0.0
        and float(transform.get("scale_x", 1.0)) == 1.0
        and float(transform.get("scale_y", 1.0)) == 1.0
    ):
        return

    graph = json.loads(raw_json_path.read_text())
    vertices = np.array(graph["vertices"], dtype=float)
    if len(vertices) == 0:
        graph["width"] = int(transform.get("full_width", graph.get("width", 0)))
        graph["height"] = int(transform.get("full_height", graph.get("height", 0)))
        raw_json_path.write_text(json.dumps(graph))
        return

    vertices[:, 0] = float(transform["offset_x"]) + vertices[:, 0] * float(transform["scale_x"])
    vertices[:, 1] = float(transform["offset_y"]) + vertices[:, 1] * float(transform["scale_y"])
    graph["vertices"] = vertices.tolist()
    graph["width"] = int(transform.get("full_width", graph.get("width", 0)))
    graph["height"] = int(transform.get("full_height", graph.get("height", 0)))
    raw_json_path.write_text(json.dumps(graph))


def _write_hawpv2_runtime_config(
    base_config_path: Path,
    save_path: Path,
    *,
    device: str,
    width: int,
    height: int,
) -> None:
    lines = base_config_path.read_text().splitlines()
    top_section = ""
    sub_section = ""
    patched: list[str] = []

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            top_section = stripped[:-1]
            sub_section = ""
        elif indent == 2 and stripped.endswith(":") and top_section == "DATASETS":
            sub_section = stripped[:-1]
        elif indent == 2 and stripped.endswith(":"):
            sub_section = ""

        if top_section == "MODEL" and stripped.startswith("DEVICE:"):
            line = f"{' ' * indent}DEVICE: {device}"
        elif top_section == "DATASETS" and sub_section == "IMAGE" and stripped.startswith("HEIGHT:"):
            line = f"{' ' * indent}HEIGHT: {height}"
        elif top_section == "DATASETS" and sub_section == "IMAGE" and stripped.startswith("WIDTH:"):
            line = f"{' ' * indent}WIDTH: {width}"
        patched.append(line)

    save_path.write_text("\n".join(patched) + "\n")


def _extract_hawpv2_graph(hawp_json_path: Path, raw_json_path: Path) -> bool:
    if not hawp_json_path.exists():
        return False

    payload = json.loads(hawp_json_path.read_text())
    if isinstance(payload, list):
        if not payload:
            return False
        graph = payload[0]
    elif isinstance(payload, dict):
        graph = payload
    else:
        return False

    raw_json_path.write_text(json.dumps(graph))
    return True


def _render_segments_overlay(
    image_rgb: np.ndarray,
    raw_json_path: Path,
    threshold: float,
    save_path: Path,
) -> None:
    graph = json.loads(raw_json_path.read_text())
    vertices = np.array(graph["vertices"], dtype=float)
    edges = np.array(graph["edges"], dtype=int)
    weights = np.array(graph["edges-weights"], dtype=float)

    image = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for edge, weight in zip(edges, weights):
        if weight < threshold:
            continue
        left = tuple(vertices[edge[0]].tolist())
        right = tuple(vertices[edge[1]].tolist())
        draw.line([left, right], fill=(0, 255, 255), width=3)
    image.save(save_path)


def _find_cached_hawp_dir(image_name: str, slug: str, current_output_dir: Path) -> Path | None:
    image_stem = Path(image_name).stem
    outputs_root = _project_root() / "outputs"
    if not outputs_root.exists():
        return None
    for summary_path in sorted(outputs_root.glob(f"*/{image_stem}/hawp/{slug}/summary.json")):
        if current_output_dir.resolve() in summary_path.parents:
            continue
        return summary_path.parent
    return None


def _reuse_cached_hawp_result(
    *,
    image_name: str,
    config_dir: Path,
    config: HawpConfig,
    summary_path: Path,
    mesh_input_rgb: np.ndarray,
    input_transform: dict,
    profile: str,
    expected_device: str | None = None,
) -> dict | None:
    cached_dir = _find_cached_hawp_dir(image_name, config.slug, config_dir.parent)
    if cached_dir is None:
        return None
    cached_summary_path = cached_dir / "summary.json"
    if not cached_summary_path.exists():
        return None
    try:
        cached_summary = json.loads(cached_summary_path.read_text())
    except json.JSONDecodeError:
        return None
    if cached_summary.get("config") != asdict(config):
        return None
    if cached_summary.get("profile") != profile:
        return None
    if expected_device is not None and cached_summary.get("resolved_device") != expected_device:
        return None
    if cached_summary.get("input_transform") != input_transform:
        return None
    raw_json_candidates = [
        path
        for path in cached_dir.glob("*.json")
        if path.name not in {"summary.json", "filtered_segments.json", "hawp.json"}
    ]
    if not raw_json_candidates:
        return None
    cached_raw_json = raw_json_candidates[0]
    raw_json_path = config_dir / cached_raw_json.name
    shutil.copy2(cached_raw_json, raw_json_path)
    segments_path = config_dir / "filtered_segments.json"
    overlay_path = config_dir / "overlay.png"
    segment_count = _write_segments_json(raw_json_path, config.threshold, segments_path)
    _render_segments_overlay(mesh_input_rgb, raw_json_path, config.threshold, overlay_path)
    result = {
        "status": "ok",
        "reason": "cached_existing_output",
        "summary_path": str(summary_path),
        "raw_json_path": str(raw_json_path),
        "filtered_segments_path": str(segments_path),
        "overlay_path": str(overlay_path),
        "segment_count": segment_count,
    }
    return result


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
    mesh_input_png = mesh_input_png.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = default_hawp_configs_for_profile(profile)
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
    available_input_variants = _build_hawp_input_variants(
        mesh_input_png,
        mesh_input_rgb,
        output_dir / "_inputs",
        profile=profile,
    )
    resolved_repo = discover_hawp_repo(hawp_repo)
    resolved_python = discover_hawp_python(hawp_python, resolved_repo)
    resolved_checkpoints = discover_checkpoints_dir(checkpoints_dir, resolved_repo)
    resolved_device = resolve_hawp_device(device, resolved_python)
    resolved_hawpv2_device = resolved_device if resolved_device in {"cuda", "cpu"} else "cpu"
    hawpv2_base_config = resolved_repo / "configs" / "hawpv2.yaml" if resolved_repo is not None else None
    runtime_configs_dir = output_dir / "_runtime_configs"
    runtime_configs_dir.mkdir(parents=True, exist_ok=True)
    ready, reason = hawp_ready(
        str(resolved_python) if resolved_python else None,
        str(resolved_repo) if resolved_repo else None,
        str(resolved_checkpoints) if resolved_checkpoints else None,
    )
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
            if config.runner == "hawpv2":
                if hawpv2_base_config is None or not hawpv2_base_config.exists():
                    result["status"] = "skipped"
                    result["reason"] = "missing hawpv2 config"
                    summary_path.write_text(json.dumps(result, indent=2))
                    results.append(result)
                    continue
                runtime_config_path = runtime_configs_dir / f"{config.slug}.yaml"
                _write_hawpv2_runtime_config(
                    hawpv2_base_config,
                    runtime_config_path,
                    device=resolved_hawpv2_device,
                    width=config.width,
                    height=config.height,
                )
                command = [
                    str(resolved_python),
                    "-m",
                    "hawp.fsl.predict",
                    str(runtime_config_path),
                    str(variant_input_path),
                    "--ckpt",
                    str(checkpoint_path),
                    "--output",
                    str(config_dir),
                ]
                result["resolved_device"] = resolved_hawpv2_device
                result["runtime_config_path"] = str(runtime_config_path)
            else:
                command = [
                    str(resolved_python),
                    "-m",
                    "hawp.ssl.predict",
                    "--ckpt",
                    str(checkpoint_path),
                    "--threshold",
                    f"{config.threshold:.2f}",
                    "--width",
                    str(config.width),
                    "--height",
                    str(config.height),
                    "--img",
                    str(variant_input_path),
                    "--saveto",
                    str(config_dir),
                    "--ext",
                    "json",
                    "--device",
                    resolved_device,
                    "--disable-show",
                ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=resolved_repo,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                result["returncode"] = completed.returncode
                result["stdout"] = completed.stdout[-4000:]
                result["stderr"] = completed.stderr[-4000:]
                raw_json_path = config_dir / f"{variant_input_path.stem}.json"
                if config.runner == "hawpv2":
                    hawpv2_json_path = config_dir / "hawp.json"
                    output_ready = completed.returncode == 0 and _extract_hawpv2_graph(hawpv2_json_path, raw_json_path)
                else:
                    output_ready = completed.returncode == 0 and raw_json_path.exists()

                if output_ready and raw_json_path.exists():
                    _remap_graph_to_full_image(raw_json_path, variant["transform"])
                    segments_path = config_dir / "filtered_segments.json"
                    overlay_path = config_dir / "overlay.png"
                    segment_count = _write_segments_json(raw_json_path, config.threshold, segments_path)
                    _render_segments_overlay(mesh_input_rgb, raw_json_path, config.threshold, overlay_path)
                    result["status"] = "ok"
                    result["reason"] = "ok"
                    result["raw_json_path"] = str(raw_json_path)
                    result["filtered_segments_path"] = str(segments_path)
                    result["overlay_path"] = str(overlay_path)
                    result["segment_count"] = segment_count
                else:
                    result["status"] = "failed"
                    result["reason"] = "hawp subprocess failed"
            except OSError as exc:
                result["status"] = "failed"
                result["reason"] = str(exc)
        elif ready:
            result["status"] = "skipped"
            result["reason"] = f"checkpoint not found: {checkpoint_path}"

        if result["status"] != "ok":
            cached_result = _reuse_cached_hawp_result(
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
