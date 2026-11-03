from __future__ import annotations

import json
import os
import subprocess
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .geometry import cluster_points


@dataclass(slots=True)
class DeepVecConfig:
    input_variant: str
    min_length_px: float
    threshold: float
    runner: str = "deepvec_line"

    @property
    def slug(self) -> str:
        return f"{self.runner}_{self.input_variant}_thr_{self.threshold:.2f}_len_{self.min_length_px:.0f}"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _deepvec_adapter_path() -> Path:
    return _project_root() / "scripts" / "deepvec_line_adapter.py"


def discover_deepvec_repo(explicit_repo: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_repo:
        candidates.append(Path(explicit_repo))
    env_repo = Path.cwd() / "Deep-Vectorization-of-Technical-Drawings-master"
    sibling_repo = _workspace_root() / "Deep-Vectorization-of-Technical-Drawings-master"
    candidates.extend([env_repo, sibling_repo])
    for candidate in candidates:
        if (candidate / "run_pipeline.py").exists() and (candidate / "vectorization").exists():
            return candidate
    return None


def discover_deepvec_python(explicit_python: str | None = None, repo: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_python:
        candidates.append(Path(explicit_python))
    if repo is not None:
        candidates.extend(
            [
                repo / ".venv_linux" / "bin" / "python",
                repo / ".venv" / "bin" / "python",
                repo / "venv" / "bin" / "python",
                repo / ".venv_linux" / "Scripts" / "python.exe",
                repo / ".venv" / "Scripts" / "python.exe",
                repo / "venv" / "Scripts" / "python.exe",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix.lower() == ".exe" and os.name != "nt":
                continue
            return candidate
    return None


def discover_deepvec_model(explicit_dir: str | None = None, repo: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    if repo is not None:
        candidates.extend([repo / "models", repo / "checkpoints", repo])
    for base in candidates:
        if not base.exists():
            continue
        for pattern in ("model_lines.weights", "*.weights", "*.pth", "*.pt"):
            matches = sorted(base.rglob(pattern))
            if matches:
                return matches[0]
    return None


def _python_supports_torch(python_path: Path | None) -> bool:
    if python_path is None:
        return False
    completed = subprocess.run(
        [str(python_path), "-c", "import torch, torchvision; print('ok')"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "ok"


def _enhance_rgb_clean(mesh_input_rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(mesh_input_rgb).astype(np.uint8), mode="RGB")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.6, percent=220, threshold=2))
    return image


def _enhance_contrast_clean(mesh_input_rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(mesh_input_rgb).astype(np.uint8), mode="RGB")
    image = ImageOps.autocontrast(image, cutoff=0)
    image = image.filter(ImageFilter.UnsharpMask(radius=2.0, percent=260, threshold=1))
    return image


def _build_input_variants(mesh_input_rgb: np.ndarray, variants_dir: Path) -> dict[str, Path]:
    variants_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "deep_rgb_clean": variants_dir / "deep_rgb_clean.png",
        "deep_contrast_clean": variants_dir / "deep_contrast_clean.png",
    }
    _enhance_rgb_clean(mesh_input_rgb).save(paths["deep_rgb_clean"])
    _enhance_contrast_clean(mesh_input_rgb).save(paths["deep_contrast_clean"])
    return paths


def default_deepvec_configs(profile: str = "extreme") -> list[DeepVecConfig]:
    min_lengths = (8.0, 14.0) if profile == "extreme" else (14.0,)
    thresholds = (0.05, 0.10) if profile == "extreme" else (0.10,)
    configs: list[DeepVecConfig] = []
    for variant in ("deep_rgb_clean", "deep_contrast_clean"):
        for min_length_px in min_lengths:
            for threshold in thresholds:
                configs.append(DeepVecConfig(variant, min_length_px, threshold))
    return configs


def _default_runtime_settings(
    profile: str,
    *,
    diff_render_it: int | None,
    model_output_count: int | None,
    overlap: int | None,
) -> dict[str, int]:
    return {
        "diff_render_it": int(diff_render_it if diff_render_it is not None else (20 if profile == "extreme" else 12)),
        "model_output_count": int(model_output_count if model_output_count is not None else (10 if profile == "extreme" else 8)),
        "overlap": int(overlap if overlap is not None else (8 if profile == "extreme" else 0)),
    }


def _wireframe_from_segments(
    segments: list[dict],
    image_shape: tuple[int, int],
    snap_threshold_px: float = 6.0,
) -> dict:
    if not segments:
        return {
            "vertices": [],
            "vertices-score": [],
            "edges": [],
            "edges-weights": [],
            "height": int(image_shape[0]),
            "width": int(image_shape[1]),
        }

    points: list[list[float]] = []
    edge_scores: list[float] = []
    for segment in segments:
        points.append([segment["x1"], segment["y1"]])
        points.append([segment["x2"], segment["y2"]])
        edge_scores.append(float(segment["score"]))
    points_array = np.asarray(points, dtype=float)
    centers, cluster_ids = cluster_points(points_array, snap_threshold_px)
    deduped_edges: dict[tuple[int, int], float] = {}
    for edge_idx, score in enumerate(edge_scores):
        left = int(cluster_ids[edge_idx * 2])
        right = int(cluster_ids[edge_idx * 2 + 1])
        if left == right:
            continue
        edge_key = (left, right) if left < right else (right, left)
        deduped_edges[edge_key] = max(float(score), deduped_edges.get(edge_key, 0.0))
    edges = [[left, right] for left, right in deduped_edges]
    weights = [deduped_edges[(left, right)] for left, right in deduped_edges]
    return {
        "vertices": centers.tolist(),
        "vertices-score": [1.0 for _ in range(len(centers))],
        "edges": edges,
        "edges-weights": weights,
        "height": int(image_shape[0]),
        "width": int(image_shape[1]),
    }


def _detect_segments_with_lsd(image_path: Path) -> list[dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(gray)[0]
    if detected is None:
        return []
    image_diag = float(np.hypot(gray.shape[0], gray.shape[1]))
    segments: list[dict] = []
    for line in detected[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        length_score = min(1.0, length / max(image_diag * 0.35, 1.0))
        segments.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": length_score,
                "width": 1.0,
                "length_px": length,
                "score": length_score,
            }
        )
    segments.sort(key=lambda item: (-item["score"], -item["length_px"]))
    return segments


def _render_overlay(mesh_input_rgb: np.ndarray, segments: list[dict], save_path: Path) -> None:
    image = Image.fromarray(np.asarray(mesh_input_rgb).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for segment in segments:
        draw.line(
            [(segment["x1"], segment["y1"]), (segment["x2"], segment["y2"])],
            fill=(0, 255, 255),
            width=3,
        )
    image.save(save_path)


def _filter_segments(
    segments: list[dict],
    *,
    min_length_px: float,
    threshold: float,
) -> list[dict]:
    filtered = [
        segment
        for segment in segments
        if float(segment.get("length_px", 0.0)) >= min_length_px and float(segment.get("score", 0.0)) >= threshold
    ]
    filtered.sort(key=lambda item: (-float(item["score"]), -float(item.get("length_px", 0.0))))
    return filtered


def _segment_signature(segments: list[dict]) -> str:
    payload: list[tuple[float, float, float, float, float]] = []
    for segment in segments:
        left = (round(float(segment["x1"]), 1), round(float(segment["y1"]), 1))
        right = (round(float(segment["x2"]), 1), round(float(segment["y2"]), 1))
        if left > right:
            left, right = right, left
        payload.append(
            (
                left[0],
                left[1],
                right[0],
                right[1],
                round(float(segment.get("score", 0.0)), 4),
            )
        )
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _run_deepvec_variant(
    *,
    variant_path: Path,
    variant_output_dir: Path,
    deepvec_python: Path,
    deepvec_repo: Path,
    model_path: Path,
    runtime_settings: dict[str, int],
) -> tuple[list[dict], dict]:
    raw_segments_path = variant_output_dir / "raw_segments.json"
    log_path = variant_output_dir / "run.log"
    command = [
        str(deepvec_python),
        str(_deepvec_adapter_path()),
        "--repo",
        str(deepvec_repo),
        "--image",
        str(variant_path),
        "--model",
        str(model_path),
        "--output-json",
        str(raw_segments_path),
        "--diff-render-it",
        str(runtime_settings["diff_render_it"]),
        "--model-output-count",
        str(runtime_settings["model_output_count"]),
        "--overlap",
        str(runtime_settings["overlap"]),
        "--device",
        "auto",
    ]
    completed = subprocess.run(
        command,
        cwd=str(_project_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text((completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or ""))
    if completed.returncode != 0:
        return [], {
            "status": "failed",
            "reason": f"deepvec adapter exited with {completed.returncode}",
            "log_path": str(log_path),
        }
    if not raw_segments_path.exists():
        return [], {
            "status": "failed",
            "reason": "deepvec adapter completed without writing raw_segments.json",
            "log_path": str(log_path),
        }
    payload = json.loads(raw_segments_path.read_text())
    return payload.get("segments", []), {
        "status": "ok",
        "reason": "ok",
        "log_path": str(log_path),
        "raw_segments_path": str(raw_segments_path),
        "payload": payload,
    }


def run_deepvec_sweep(
    image_name: str,
    mesh_input_rgb: np.ndarray,
    output_dir: Path,
    deepvec_python: str | None,
    deepvec_repo: str | None,
    model_dir: str | None,
    profile: str = "extreme",
    *,
    diff_render_it: int | None = None,
    model_output_count: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_paths = _build_input_variants(mesh_input_rgb, output_dir / "_inputs")
    configs = default_deepvec_configs(profile)
    runtime_settings = _default_runtime_settings(
        profile,
        diff_render_it=diff_render_it,
        model_output_count=model_output_count,
        overlap=overlap,
    )
    resolved_repo = discover_deepvec_repo(deepvec_repo)
    resolved_python = discover_deepvec_python(deepvec_python, resolved_repo)
    resolved_model = discover_deepvec_model(model_dir, resolved_repo)
    torch_ready = _python_supports_torch(resolved_python)
    deepvec_ready = bool(resolved_repo and resolved_python and resolved_model and torch_ready and _deepvec_adapter_path().exists())
    variant_cache: dict[str, dict] = {}

    for variant_name, variant_path in variant_paths.items():
        variant_output_dir = output_dir / f"_variant_{variant_name}"
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        if deepvec_ready:
            segments, details = _run_deepvec_variant(
                variant_path=variant_path,
                variant_output_dir=variant_output_dir,
                deepvec_python=resolved_python,
                deepvec_repo=resolved_repo,
                model_path=resolved_model,
                runtime_settings=runtime_settings,
            )
            if details["status"] == "ok":
                variant_cache[variant_name] = {
                    "segments": segments,
                    "runner": "deepvec_line_pretrained",
                    "status": "ok",
                    "reason": "ok",
                    "details": details,
                }
                continue
        fallback_segments = _detect_segments_with_lsd(variant_path)
        fallback_reason = "missing DeepVec environment"
        if deepvec_ready:
            fallback_reason = "DeepVec adapter failed; fell back to LSD"
        variant_cache[variant_name] = {
            "segments": fallback_segments,
            "runner": "deepvec_lsd_fallback",
            "status": "ok" if fallback_segments else "failed",
            "reason": fallback_reason if fallback_segments else "no segments detected",
            "details": {
                "status": "ok" if fallback_segments else "failed",
                "reason": fallback_reason if fallback_segments else "no segments detected",
                "raw_segments_path": None,
                "payload": None,
            },
        }

    results: list[dict] = []
    image_shape = mesh_input_rgb.shape[:2]
    seen_signatures: dict[str, str] = {}
    for config in configs:
        config_dir = output_dir / config.slug
        config_dir.mkdir(parents=True, exist_ok=True)
        variant_state = variant_cache[config.input_variant]
        filtered_segments = _filter_segments(
            variant_state["segments"],
            min_length_px=config.min_length_px,
            threshold=config.threshold,
        )
        signature = _segment_signature(filtered_segments) if filtered_segments else ""
        duplicate_of = seen_signatures.get(signature) if signature else None
        if signature and duplicate_of is None:
            seen_signatures[signature] = config.slug
        if duplicate_of is not None:
            result = {
                "image_name": image_name,
                "config": {**asdict(config), **runtime_settings},
                "status": "duplicate",
                "reason": f"duplicate of {duplicate_of}",
                "summary_path": str(config_dir / "summary.json"),
                "raw_json_path": None,
                "filtered_segments_path": None,
                "overlay_path": None,
                "segment_count": len(filtered_segments),
                "base_segment_count": len(variant_state["segments"]),
                "runner": variant_state["runner"],
                "profile": profile,
                "input_variant_path": str(variant_paths[config.input_variant]),
                "resolved_deepvec_repo": str(resolved_repo) if resolved_repo else None,
                "resolved_deepvec_python": str(resolved_python) if resolved_python else None,
                "resolved_deepvec_model": str(resolved_model) if resolved_model else None,
                "torch_ready": torch_ready,
                "deepvec_ready": deepvec_ready,
                "variant_status": variant_state["status"],
                "variant_reason": variant_state["reason"],
                "variant_log_path": variant_state["details"].get("log_path"),
                "raw_segments_path": variant_state["details"].get("raw_segments_path"),
                "duplicate_of": duplicate_of,
                "segment_signature": signature,
            }
            Path(result["summary_path"]).write_text(json.dumps(result, indent=2))
            results.append(result)
            continue
        wireframe = _wireframe_from_segments(filtered_segments, image_shape)
        raw_json_path = config_dir / f"{Path(variant_paths[config.input_variant]).stem}.json"
        raw_json_path.write_text(json.dumps(wireframe, indent=2))
        segments_path = config_dir / "filtered_segments.json"
        segments_path.write_text(json.dumps(filtered_segments, indent=2))
        overlay_path = config_dir / "overlay.png"
        _render_overlay(mesh_input_rgb, filtered_segments, overlay_path)
        result = {
            "image_name": image_name,
            "config": {**asdict(config), **runtime_settings},
            "status": "ok" if filtered_segments else "failed",
            "reason": "ok" if filtered_segments else "no segments survived filtering",
            "summary_path": str(config_dir / "summary.json"),
            "raw_json_path": str(raw_json_path),
            "filtered_segments_path": str(segments_path),
            "overlay_path": str(overlay_path),
            "segment_count": len(filtered_segments),
            "base_segment_count": len(variant_state["segments"]),
            "runner": variant_state["runner"],
            "profile": profile,
            "input_variant_path": str(variant_paths[config.input_variant]),
            "resolved_deepvec_repo": str(resolved_repo) if resolved_repo else None,
            "resolved_deepvec_python": str(resolved_python) if resolved_python else None,
            "resolved_deepvec_model": str(resolved_model) if resolved_model else None,
            "torch_ready": torch_ready,
            "deepvec_ready": deepvec_ready,
            "variant_status": variant_state["status"],
            "variant_reason": variant_state["reason"],
            "variant_log_path": variant_state["details"].get("log_path"),
            "raw_segments_path": variant_state["details"].get("raw_segments_path"),
            "duplicate_of": None,
            "segment_signature": signature or None,
        }
        Path(result["summary_path"]).write_text(json.dumps(result, indent=2))
        results.append(result)

    return results
