
#@file main.py
#@brief For NIO Student Hackathon 
#@team Zhengzhe Liu, Xinyan Wang, Ziyi Jiang
#@github LIGHTNINGWHALE(Zhengzhe Liu)
#@date 2026-04-14


from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent
SUBMISSION_NAS_DIR = BUNDLE_ROOT / "submission_nas"
DEBUG_OUTPUT_DIR = BUNDLE_ROOT / "debug_output"
BENCHMARK_OUTPUT_DIR = BUNDLE_ROOT / "benchmark_output"
TEMP_WORK_DIR = BUNDLE_ROOT / ".bundle_work"


def _check_runtime_dependencies() -> None:
    required = [
        "numpy",
        "scipy",
        "PIL",
        "matplotlib",
        "cv2",
        "ortools",
        "shapely",
        "torch",
        "torchvision",
        "yacs",
        "tqdm",
        "easydict",
        "yaml",
    ]
    missing: list[str] = []
    for module_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        joined = ", ".join(sorted(missing))
        raise SystemExit(
            "Missing runtime dependencies: "
            f"{joined}. Install them first with `pip install -r requirements_submission.txt`."
        )


def _device_candidates(requested_device: str) -> list[str]:
    if requested_device != "auto":
        return [requested_device]

    try:
        import torch
    except Exception:
        return ["cpu"]

    ordered: list[str] = []
    if torch.cuda.is_available():
        ordered.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ordered.append("mps")
    ordered.append("cpu")

    deduped: list[str] = []
    for candidate in ordered:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _run_full_with_fallback(
    image_paths: list[Path],
    *,
    output_dir: Path,
    truth_dir: str | None = None,
    prop_id: int = 0,
    requested_device: str = "auto",
) -> dict:
    last_exc: Exception | None = None
    candidates = _device_candidates(requested_device)
    for index, device_name in enumerate(candidates):
        if output_dir.exists():
            _prepare_dir(output_dir)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if index > 0:
                print(f"[bundle] retrying with device={device_name}")
            return _run_full(
                image_paths,
                output_dir=output_dir,
                truth_dir=truth_dir,
                prop_id=prop_id,
                device=device_name,
            )
        except Exception as exc:
            last_exc = exc
            if index == len(candidates) - 1:
                raise
            print(f"[bundle] device={device_name} failed: {exc}")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No available device candidates")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIO submission bundle runner")
    parser.add_argument("--debug", action="store_true", help="Keep the full debug artifact tree in debug_output/.")
    parser.add_argument(
        "--benchmark-examples",
        action="store_true",
        help="Run the two example benchmark images with truth comparisons into benchmark_output/.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(BUNDLE_ROOT / "jpeg images"),
        help="Directory containing hole_*_normalised.jpg files. Defaults to ./jpeg images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Visible output directory. In normal mode this is the final .nas directory. "
            "In debug mode it is the debug artifact root. In benchmark mode it is the benchmark root."
        ),
    )
    parser.add_argument("--images", nargs="+", default=None, help="Explicit image paths to run.")
    parser.add_argument(
        "--image-names",
        nargs="+",
        default=None,
        help="Image filenames under --input-dir to run, for example hole_001_normalised.jpg.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps"))
    return parser.parse_args()


def _collect_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        return [Path(item).resolve() for item in args.images]

    input_dir = Path(args.input_dir).resolve()
    if args.image_names:
        return [(input_dir / name).resolve() for name in args.image_names]

    return sorted(input_dir.glob("hole_*_normalised.jpg"))


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            child.chmod(0o777)
        except OSError:
            pass
        if child.is_file() or child.is_symlink():
            try:
                child.unlink()
            except OSError:
                pass
        elif child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def _prepare_dir(path: Path) -> Path:
    if path.exists():
        _remove_tree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_submission_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*.nas"):
        item.unlink()
    keep = path / ".gitkeep"
    keep.write_text("", encoding="utf-8")
    return path


def _fixed_options_dict(image_paths: list[Path], output_dir: Path, truth_dir: str | None, prop_id: int, device: str) -> dict:
    return {
        "mode": "full",
        "profile": "baseline",
        "images": [str(path) for path in image_paths],
        "truth_dir": truth_dir,
        "output_dir": str(output_dir),
        "save_intermediates": False,
        "hawp_python": sys.executable,
        "hawp_repo": str(BUNDLE_ROOT / "nio_mesh" / "vendor_hawp"),
        "hawp_checkpoints_dir": str(BUNDLE_ROOT / "models" / "checkpoints"),
        "hawp_device": device,
        "hawp_runner": "hawpv3",
        "hawp_thresholds": [0.05],
        "hawp_min_width": None,
        "hawp_checkpoint_names": ["hawpv3-imagenet-03a84.pth"],
        "hawp_input_variants": ["roi_sharp", "roi_sharp_context"],
        "hawp_widths": [1792],
        "candidate_pool": "hawp_faces_only",
        "deepvec_python": None,
        "deepvec_repo": None,
        "deepvec_model_dir": None,
        "deepvec_diff_render_it": None,
        "deepvec_model_output_count": None,
        "deepvec_overlap": None,
        "filter_black_elements": True,
        "black_pixel_threshold": 24,
        "black_patch_radius": 2,
        "black_patch_min_fraction": 0.6,
        "prop_id": prop_id,
    }


def _run_full(
    image_paths: list[Path], *, output_dir: Path, truth_dir: str | None = None, prop_id: int = 0, device: str = "auto"
) -> dict:
    from nio_mesh.pipeline import PipelineOptions, run_full_mode

    options = PipelineOptions(**_fixed_options_dict(image_paths, output_dir, truth_dir, prop_id, device))
    return run_full_mode(options)


def _copy_final_nas(image_paths: list[Path], work_root: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        stem = image_path.stem
        src = work_root / stem / f"{stem}.nas"
        dst = final_dir / f"{stem}.nas"
        if not src.exists():
            raise FileNotFoundError(f"Missing generated NAS file: {src}")
        shutil.copy2(src, dst)


def _run_default(args: argparse.Namespace) -> None:
    image_paths = _collect_images(args)
    if not image_paths:
        raise SystemExit("No input images found.")

    final_dir = Path(args.output_dir).resolve() if args.output_dir else SUBMISSION_NAS_DIR
    _prepare_submission_dir(final_dir)

    if args.debug:
        _prepare_submission_dir(SUBMISSION_NAS_DIR)
        debug_root = Path(args.output_dir).resolve() if args.output_dir else DEBUG_OUTPUT_DIR
        _run_full_with_fallback(image_paths, output_dir=debug_root, requested_device=args.device)
        _copy_final_nas(image_paths, debug_root, SUBMISSION_NAS_DIR)
        return

    work_root = _prepare_dir(TEMP_WORK_DIR)
    try:
        _run_full_with_fallback(image_paths, output_dir=work_root, requested_device=args.device)
        _copy_final_nas(image_paths, work_root, final_dir)
    finally:
        if work_root.exists():
            _remove_tree(work_root)


def _run_benchmark(args: argparse.Namespace) -> None:
    benchmark_root = Path(args.output_dir).resolve() if args.output_dir else BENCHMARK_OUTPUT_DIR
    _prepare_dir(benchmark_root)
    truth_dir = BUNDLE_ROOT / "Example .nas Format"
    benchmark_specs = [
        ("hole_115_normalised.jpg", 115, benchmark_root / "hole_115_run"),
        ("hole_122_normalised.jpg", 122, benchmark_root / "hole_122_run"),
    ]
    for image_name, prop_id, run_dir in benchmark_specs:
        _run_full_with_fallback(
            [(truth_dir / image_name).resolve()],
            output_dir=run_dir,
            truth_dir=str(truth_dir),
            prop_id=prop_id,
            requested_device=args.device,
        )


def main() -> None:
    _check_runtime_dependencies()
    args = _parse_args()
    os.chdir(BUNDLE_ROOT)
    if args.benchmark_examples:
        _run_benchmark(args)
    else:
        _run_default(args)


if __name__ == "__main__":
    main()
