from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .axis import (
    detect_axis_calibration,
    detect_axis_calibration_details,
    focus_mesh_region,
    save_axis_debug,
    strip_axis_artifacts,
)
from .hawp_bundle import run_hawp_sweep


@dataclass(slots=True)
class PipelineOptions:
    mode: str
    profile: str
    images: list[str]
    truth_dir: str | None
    output_dir: str
    save_intermediates: bool
    hawp_python: str | None
    hawp_repo: str | None
    hawp_checkpoints_dir: str | None
    hawp_device: str
    hawp_runner: str
    hawp_thresholds: list[float] | None
    hawp_min_width: int | None
    hawp_checkpoint_names: list[str] | None
    hawp_input_variants: list[str] | None
    hawp_widths: list[int] | None
    candidate_pool: str
    deepvec_python: str | None
    deepvec_repo: str | None
    deepvec_model_dir: str | None
    deepvec_diff_render_it: int | None
    deepvec_model_output_count: int | None
    deepvec_overlap: int | None
    filter_black_elements: bool
    black_pixel_threshold: int
    black_patch_radius: int
    black_patch_min_fraction: float
    prop_id: int


def _load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _save_image(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB").save(path)


def _save_gray(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB").convert("L").save(path)


def _image_prop_id(image_path: Path, default_prop_id: int) -> int:
    if int(default_prop_id) > 0:
        return int(default_prop_id)
    match = re.search(r"hole_(\d+)_", image_path.stem)
    if not match:
        raise ValueError(f"Cannot infer prop_id/file_id from image name: {image_path.name}")
    return int(match.group(1))


def _truth_path(image_path: Path, truth_dir: str | None) -> Path | None:
    if not truth_dir:
        return None
    candidate = Path(truth_dir) / f"{image_path.stem}.nas"
    return candidate if candidate.exists() else None


def _validate_candidate(candidate, prop_id: int):
    from .mesh_validator import validate_mesh
    from .nas import compact_mesh

    candidate.mesh = compact_mesh(candidate.mesh, drop_orphan_nodes=True, renumber_nodes=True)
    candidate.validation_report = validate_mesh(candidate.mesh, expected_prop_id=prop_id)
    candidate.structural_score = float(candidate.validation_report.get("structural_score", 0.0))
    if candidate.solver_status is None:
        candidate.solver_status = candidate.mesh.metadata.get(
            "solver_status",
            "valid" if candidate.validation_report.get("valid") else "invalid",
        )
    return candidate


def _candidate_leaderboard_entry(candidate, report: dict | None = None) -> dict:
    metadata = candidate.mesh.metadata or {}
    return {
        "candidate": candidate.name,
        "source": candidate.source,
        "proposal_family": candidate.proposal_family,
        "structural_score": candidate.structural_score,
        "solver_status": candidate.solver_status,
        "wireframe_leaf_vertex_count": metadata.get("wireframe_leaf_vertex_count"),
        "wireframe_long_leaf_vertex_count": metadata.get("wireframe_long_leaf_vertex_count"),
        "variant_fusion_base": metadata.get("variant_fusion_base"),
        "variant_fusion_guide": metadata.get("variant_fusion_guide"),
        "variant_fusion_snap_count": metadata.get("variant_fusion_snap_count"),
        "variant_fusion_marker_snap_count": metadata.get("variant_fusion_marker_snap_count"),
        "variant_fusion_marker_preserve_count": metadata.get("variant_fusion_marker_preserve_count"),
        "variant_fusion_marker_refine_count": metadata.get("variant_fusion_marker_refine_count"),
        "validation_report": candidate.validation_report,
        "report": report,
    }


def _no_truth_sort_key(candidate) -> tuple:
    validation = candidate.validation_report or {}
    counts = candidate.mesh.counts()
    metadata = candidate.mesh.metadata or {}
    edge_coverage = float(metadata.get("wireframe_edge_coverage", 0.0))
    element_area_px_sum = float(metadata.get("element_area_px_sum", 0.0))
    unused_edge_count = int(metadata.get("unused_wireframe_edge_count", 10**9))
    long_leaf_count = int(metadata.get("wireframe_long_leaf_vertex_count", 0))
    leaf_count = int(metadata.get("wireframe_leaf_vertex_count", 0))
    return (
        0 if validation.get("valid") else 1,
        long_leaf_count,
        leaf_count,
        -edge_coverage,
        unused_edge_count,
        -element_area_px_sum,
        -(candidate.structural_score or 0.0),
        -len(candidate.mesh.elements),
        -counts.get("GRID", 0),
        candidate.name,
    )


def _candidate_sort_key(candidate_report: dict) -> tuple:
    return (
        0 if candidate_report["all_counts_exact"] else 1,
        0 if candidate_report["count_match"]["GRID"] else 1,
        0 if candidate_report["count_match"]["CTRIA3"] else 1,
        0 if candidate_report["count_match"]["CQUAD4"] else 1,
        candidate_report["node_metrics"]["rmse"] if candidate_report["node_metrics"]["rmse"] is not None else 1e9,
        candidate_report["node_metrics"]["mean_abs_error"]
        if candidate_report["node_metrics"]["mean_abs_error"] is not None
        else 1e9,
    )


def _select_candidate(candidates, truth_mesh: NasMesh | None, prop_id: int):
    from .compare import compare_meshes

    candidates = [_validate_candidate(candidate, prop_id) for candidate in candidates]
    if truth_mesh is None:
        ranked = sorted(candidates, key=_no_truth_sort_key)
        return ranked[0], None, [_candidate_leaderboard_entry(candidate) for candidate in ranked[:20]]

    scored = []
    for candidate in candidates:
        report = compare_meshes(candidate.mesh, truth_mesh)
        scored.append((candidate, report))
    scored.sort(
        key=lambda item: (
            0 if item[0].validation_report.get("valid") else 1,
            _candidate_sort_key(item[1]),
            -(item[0].structural_score or 0.0),
        )
    )
    best_candidate, best_report = scored[0]
    return best_candidate, best_report, [_candidate_leaderboard_entry(candidate, report) for candidate, report in scored[:20]]


def run_hawp_mode(options: PipelineOptions) -> dict:
    output_root = Path(options.output_dir)
    summary: dict[str, dict] = {}

    image_count = len(options.images)
    for image_index, image_name in enumerate(options.images, start=1):
        image_path = Path(image_name)
        print(f"[pipeline] hawp {image_index:03d}/{image_count:03d} {image_path.name}", flush=True)
        image_dir = output_root / image_path.stem
        image_rgb = _load_image_rgb(image_path)
        axis_details = detect_axis_calibration_details(image_rgb)
        calibration = axis_details.stabilized
        save_axis_debug(image_rgb, calibration, image_dir / "axis_debug.png")

        mesh_input_rgb = focus_mesh_region(strip_axis_artifacts(image_rgb, calibration))
        _save_image(mesh_input_rgb, image_dir / "mesh_branch.png")
        mesh_input_gray = image_dir / "mesh_branch_gray.png"
        _save_gray(mesh_input_rgb, mesh_input_gray)

        hawp_results = run_hawp_sweep(
            image_name=image_path.name,
            mesh_input_png=mesh_input_gray,
            mesh_input_rgb=mesh_input_rgb,
            output_dir=image_dir / "hawp",
            hawp_python=options.hawp_python,
            hawp_repo=options.hawp_repo,
            checkpoints_dir=options.hawp_checkpoints_dir,
            device=options.hawp_device,
            profile=options.profile,
            runner=options.hawp_runner,
            thresholds=tuple(options.hawp_thresholds) if options.hawp_thresholds else None,
            min_width=options.hawp_min_width,
            checkpoint_names=tuple(options.hawp_checkpoint_names) if options.hawp_checkpoint_names else None,
            input_variants=tuple(options.hawp_input_variants) if options.hawp_input_variants else None,
            widths=tuple(options.hawp_widths) if options.hawp_widths else None,
        )
        summary[image_path.stem] = {
            "axis": calibration.to_dict(),
            "axis_raw": axis_details.raw.to_dict(),
            "axis_stabilized": axis_details.stabilized.to_dict(),
            "axis_stabilization_applied": axis_details.stabilization_applied,
            "axis_stabilization_reason": axis_details.stabilization_reason,
            "hawp_runs": hawp_results,
        }

    (output_root / "hawp_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_full_mode(options: PipelineOptions) -> dict:
    from .compare import compare_nas_files, write_report
    from .nas import parse_nas, write_nas
    from .element_filter import (
        filter_black_center_elements,
        save_element_pixel_audit_overlay,
        write_element_pixel_audit,
    )
    from .reconstruct import (
        apply_candidate_coordinate_correction,
        apply_selected_marker_center_snap,
        build_candidate_grid,
        build_hawp_face_candidate,
        build_hawp_refined_candidates,
        build_hawp_variant_fused_candidate,
        extract_component_polygons,
        save_mesh_overlay,
        save_wireframe_overlay,
        simplify_wireframe_evidence,
    )
    from .mesh_solver import solve_hybrid_candidate

    output_root = Path(options.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    image_count = len(options.images)
    for image_index, image_name in enumerate(options.images, start=1):
        image_path = Path(image_name)
        print(f"[pipeline] full {image_index:03d}/{image_count:03d} {image_path.name}", flush=True)
        image_dir = output_root / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        image_rgb = _load_image_rgb(image_path)
        axis_details = detect_axis_calibration_details(image_rgb)
        calibration = axis_details.stabilized
        save_axis_debug(image_rgb, calibration, image_dir / "axis_debug.png")

        mesh_input_rgb = focus_mesh_region(strip_axis_artifacts(image_rgb, calibration))
        _save_image(mesh_input_rgb, image_dir / "mesh_branch.png")
        mesh_input_gray = image_dir / "mesh_branch_gray.png"
        _save_gray(mesh_input_rgb, mesh_input_gray)
        truth_path = _truth_path(image_path, options.truth_dir)
        truth_mesh = parse_nas(truth_path) if truth_path else None
        prop_id = _image_prop_id(image_path, options.prop_id)

        hawp_results = run_hawp_sweep(
            image_name=image_path.name,
            mesh_input_png=mesh_input_gray,
            mesh_input_rgb=mesh_input_rgb,
            output_dir=image_dir / "hawp",
            hawp_python=options.hawp_python,
            hawp_repo=options.hawp_repo,
            checkpoints_dir=options.hawp_checkpoints_dir,
            device=options.hawp_device,
            profile=options.profile,
            runner=options.hawp_runner,
            thresholds=tuple(options.hawp_thresholds) if options.hawp_thresholds else None,
            min_width=options.hawp_min_width,
            checkpoint_names=tuple(options.hawp_checkpoint_names) if options.hawp_checkpoint_names else None,
            input_variants=tuple(options.hawp_input_variants) if options.hawp_input_variants else None,
            widths=tuple(options.hawp_widths) if options.hawp_widths else None,
        )
        if options.candidate_pool == "all":
            from .deepvec import run_deepvec_sweep

            deepvec_results = run_deepvec_sweep(
                image_name=image_path.name,
                mesh_input_rgb=mesh_input_rgb,
                output_dir=image_dir / "deepvec",
                deepvec_python=options.deepvec_python,
                deepvec_repo=options.deepvec_repo,
                model_dir=options.deepvec_model_dir,
                profile=options.profile,
                diff_render_it=options.deepvec_diff_render_it,
                model_output_count=options.deepvec_model_output_count,
                overlap=options.deepvec_overlap,
            )
        else:
            deepvec_results = []

        classical_candidates = []
        classical_anchor = None
        classical_anchor_report = None
        if options.candidate_pool != "hawp_faces_only":
            components = extract_component_polygons(mesh_input_rgb, image_dir / "classical")
            classical_candidates = build_candidate_grid(
                components,
                calibration,
                prop_id=prop_id,
                image_shape=image_rgb.shape[:2],
            )
            classical_candidates = [
                apply_candidate_coordinate_correction(
                    candidate,
                    calibration,
                    axis_stabilization_applied=axis_details.stabilization_applied,
                )
                for candidate in classical_candidates
            ]
            if classical_candidates:
                classical_anchor, classical_anchor_report, _ = _select_candidate(classical_candidates, truth_mesh, prop_id)

        hawp_candidates = []
        evidence_wireframes: list[dict] = []
        hawp_face_candidates_by_name: dict[str, object] = {}
        for hawp_run in hawp_results:
            if hawp_run.get("status") != "ok":
                continue
            raw_json_path = hawp_run.get("raw_json_path")
            if not raw_json_path:
                continue
            raw_json = Path(raw_json_path)
            if not raw_json.exists():
                continue
            wireframe = json.loads(raw_json.read_text())
            config = hawp_run["config"]
            evidence_wireframe, _ = simplify_wireframe_evidence(
                wireframe,
                edge_threshold=float(config["threshold"]),
                image_shape=image_rgb.shape[:2],
            )
            if evidence_wireframe["edges"]:
                evidence_wireframes.append(evidence_wireframe)
            config_tag = (
                f"{Path(config['checkpoint_name']).stem}_{config['input_variant']}"
                f"_thr_{config['threshold']:.2f}_{config['width']}x{config['height']}"
            )

            face_candidate = build_hawp_face_candidate(
                candidate_name=f"hawp_faces_{config_tag}",
                wireframe=wireframe,
                edge_threshold=float(config["threshold"]),
                calibration=calibration,
                prop_id=prop_id,
                image_shape=image_rgb.shape[:2],
                mesh_image_rgb=mesh_input_rgb,
            )
            if face_candidate is not None:
                corrected_face_candidate = apply_candidate_coordinate_correction(
                    face_candidate,
                    calibration,
                    axis_stabilization_applied=axis_details.stabilization_applied,
                )
                hawp_candidates.append(corrected_face_candidate)
                hawp_face_candidates_by_name[corrected_face_candidate.name] = corrected_face_candidate

            if options.candidate_pool != "hawp_faces_only" and classical_anchor is not None:
                hawp_candidates.extend(
                    [
                        apply_candidate_coordinate_correction(
                            candidate,
                            calibration,
                            axis_stabilization_applied=axis_details.stabilization_applied,
                        )
                        for candidate in build_hawp_refined_candidates(
                            base_candidate=classical_anchor,
                            wireframe=wireframe,
                            edge_threshold=float(config["threshold"]),
                            calibration=calibration,
                            candidate_prefix=f"hawp_refine_{config_tag}",
                            image_shape=image_rgb.shape[:2],
                        )
                    ]
                )

        fusion_candidates = []
        for context_name, context_candidate in hawp_face_candidates_by_name.items():
            if "_roi_sharp_context_" not in context_name:
                continue
            sharp_name = context_name.replace("_roi_sharp_context_", "_roi_sharp_")
            sharp_candidate = hawp_face_candidates_by_name.get(sharp_name)
            if sharp_candidate is None:
                continue
            fused_candidate = build_hawp_variant_fused_candidate(
                base_candidate=context_candidate,
                guide_candidate=sharp_candidate,
                calibration=calibration,
                candidate_name=context_name.replace("hawp_faces_", "hawp_faces_fused_", 1),
                mesh_image_rgb=mesh_input_rgb,
            )
            if fused_candidate is not None:
                fusion_candidates.append(
                    apply_candidate_coordinate_correction(
                        fused_candidate,
                        calibration,
                        axis_stabilization_applied=axis_details.stabilization_applied,
                    )
                )
        hawp_candidates.extend(fusion_candidates)

        deepvec_candidates = []
        if options.candidate_pool == "all":
            for deepvec_run in deepvec_results:
                if deepvec_run.get("status") != "ok":
                    continue
                raw_json_path = deepvec_run.get("raw_json_path")
                if not raw_json_path:
                    continue
                raw_json = Path(raw_json_path)
                if not raw_json.exists():
                    continue
                wireframe = json.loads(raw_json.read_text())
                config = deepvec_run["config"]
                evidence_wireframe, _ = simplify_wireframe_evidence(
                    wireframe,
                    edge_threshold=float(config["threshold"]),
                    image_shape=image_rgb.shape[:2],
                )
                if evidence_wireframe["edges"]:
                    evidence_wireframes.append(evidence_wireframe)
                config_tag = (
                    f"{deepvec_run['runner']}_{config['input_variant']}"
                    f"_thr_{config['threshold']:.2f}_len_{config['min_length_px']:.0f}"
                )
                face_candidate = build_hawp_face_candidate(
                    candidate_name=f"deepvec_faces_{config_tag}",
                    wireframe=wireframe,
                    edge_threshold=float(config["threshold"]),
                    calibration=calibration,
                    prop_id=prop_id,
                    image_shape=image_rgb.shape[:2],
                    mesh_image_rgb=mesh_input_rgb,
                )
                if face_candidate is not None:
                    face_candidate.source = "deepvec_faces"
                    face_candidate.proposal_family = "deepvec_faces"
                    deepvec_candidates.append(
                        apply_candidate_coordinate_correction(
                            face_candidate,
                            calibration,
                            axis_stabilization_applied=axis_details.stabilization_applied,
                        )
                    )
                if classical_anchor is not None:
                    refined = build_hawp_refined_candidates(
                        base_candidate=classical_anchor,
                        wireframe=wireframe,
                        edge_threshold=float(config["threshold"]),
                        calibration=calibration,
                        candidate_prefix=f"deepvec_refine_{config_tag}",
                        image_shape=image_rgb.shape[:2],
                    )
                    for candidate in refined:
                        candidate.source = "deepvec_refine"
                        candidate.proposal_family = "deepvec_refine"
                        deepvec_candidates.append(
                            apply_candidate_coordinate_correction(
                                candidate,
                                calibration,
                                axis_stabilization_applied=axis_details.stabilization_applied,
                            )
                        )

        hybrid_candidate = None
        if options.candidate_pool == "all" and classical_anchor is not None:
            hybrid_candidate = solve_hybrid_candidate(
                base_candidate=classical_anchor,
                evidence_wireframes=evidence_wireframes,
                calibration=calibration,
                prop_id=prop_id,
                mesh_input_rgb=mesh_input_rgb,
            )
            if hybrid_candidate is not None:
                hybrid_candidate = apply_candidate_coordinate_correction(
                    hybrid_candidate,
                    calibration,
                    axis_stabilization_applied=axis_details.stabilization_applied,
                )
        if options.candidate_pool == "hawp_faces_only":
            all_candidates = [candidate for candidate in hawp_candidates if candidate.proposal_family == "hawp_faces"]
        elif options.candidate_pool == "hawp_only":
            all_candidates = list(hawp_candidates)
        else:
            all_candidates = classical_candidates + hawp_candidates + deepvec_candidates
            if hybrid_candidate is not None:
                all_candidates.append(hybrid_candidate)
        if not all_candidates:
            raise RuntimeError(f"No reconstruction candidates available for {image_path.name}")
        best_candidate, best_report, candidate_leaderboard = _select_candidate(all_candidates, truth_mesh, prop_id)
        best_candidate = apply_selected_marker_center_snap(
            best_candidate,
            calibration,
            mesh_input_rgb,
        )
        best_candidate = apply_candidate_coordinate_correction(
            best_candidate,
            calibration,
            axis_stabilization_applied=axis_details.stabilization_applied,
        )
        _validate_candidate(best_candidate, prop_id)
        best_report = None

        if options.filter_black_elements:
            prefilter_mesh = best_candidate.mesh
            save_mesh_overlay(
                mesh_input_rgb,
                prefilter_mesh,
                image_dir / "predicted_overlay_prefilter.png",
                label_nodes=False,
            )
            save_mesh_overlay(
                mesh_input_rgb,
                prefilter_mesh,
                image_dir / "predicted_overlay_prefilter_annotated.png",
                label_nodes=True,
            )
            filtered_mesh, element_pixel_audit = filter_black_center_elements(
                prefilter_mesh,
                mesh_input_rgb,
                black_threshold=options.black_pixel_threshold,
                patch_radius=options.black_patch_radius,
                min_black_fraction=options.black_patch_min_fraction,
            )
            save_element_pixel_audit_overlay(
                mesh_input_rgb,
                prefilter_mesh,
                element_pixel_audit,
                image_dir / "element_pixel_audit_overlay.png",
            )
            best_candidate.mesh = filtered_mesh
            _validate_candidate(best_candidate, prop_id)
            element_pixel_audit["filtered_counts_after_compaction"] = best_candidate.mesh.counts()
            audit_paths = write_element_pixel_audit(element_pixel_audit, image_dir)
            best_candidate.mesh.metadata["element_pixel_audit_paths"] = {
                key: str(path) for key, path in audit_paths.items()
            }
            best_report = None

        predicted_path = write_nas(best_candidate.mesh, image_dir / f"{image_path.stem}.nas")
        save_mesh_overlay(mesh_input_rgb, best_candidate.mesh, image_dir / "predicted_overlay.png", label_nodes=False)
        save_mesh_overlay(
            mesh_input_rgb,
            best_candidate.mesh,
            image_dir / "predicted_overlay_annotated.png",
            label_nodes=True,
        )
        classical_wireframe = classical_anchor.wireframe if classical_anchor is not None else {
            "vertices": [],
            "vertices-score": [],
            "edges": [],
            "edges-weights": [],
            "height": int(image_rgb.shape[0]),
            "width": int(image_rgb.shape[1]),
        }
        (image_dir / "classical_wireframe.json").write_text(json.dumps(classical_wireframe, indent=2))
        (image_dir / "selected_wireframe.json").write_text(json.dumps(best_candidate.wireframe, indent=2))
        save_wireframe_overlay(
            mesh_input_rgb,
            best_candidate.wireframe,
            image_dir / "selected_wireframe_overlay.png",
            label_vertices=False,
        )
        save_wireframe_overlay(
            mesh_input_rgb,
            best_candidate.wireframe,
            image_dir / "selected_wireframe_overlay_annotated.png",
            label_vertices=True,
        )

        if truth_path is not None:
            report_json = image_dir / "comparison.json"
            report_md = image_dir / "comparison.md"
            if best_report is None:
                best_report = compare_nas_files(predicted_path, truth_path)
            else:
                best_report["predicted_path"] = str(predicted_path)
                best_report["truth_path"] = str(truth_path)
            write_report(best_report, report_json, report_md)

        image_summary = {
            "profile": options.profile,
            "candidate_pool": options.candidate_pool,
            "axis": calibration.to_dict(),
            "axis_raw": axis_details.raw.to_dict(),
            "axis_stabilized": axis_details.stabilized.to_dict(),
            "axis_stabilization_applied": axis_details.stabilization_applied,
            "axis_stabilization_reason": axis_details.stabilization_reason,
            "hawp_runs": hawp_results,
            "deepvec_runs": deepvec_results,
            "selected_candidate": {
                "name": best_candidate.name,
                "source": best_candidate.source,
                "proposal_family": best_candidate.proposal_family,
                "merge_threshold": best_candidate.merge_threshold,
                "ratio_to_triangle": best_candidate.ratio_to_triangle,
                "ratio_to_quad": best_candidate.ratio_to_quad,
                "counts": best_candidate.mesh.counts(),
                "structural_score": best_candidate.structural_score,
                "solver_status": best_candidate.solver_status,
                "validation_report": best_candidate.validation_report,
                "predicted_path": str(predicted_path),
                "element_pixel_filter": best_candidate.mesh.metadata.get("element_pixel_filter"),
                "element_pixel_audit_paths": best_candidate.mesh.metadata.get("element_pixel_audit_paths"),
                "coordinate_correction_mode": best_candidate.mesh.metadata.get("coordinate_correction_mode"),
                "coordinate_correction_stats": best_candidate.mesh.metadata.get("coordinate_correction_stats"),
                "wireframe_leaf_vertex_count": best_candidate.mesh.metadata.get("wireframe_leaf_vertex_count"),
                "wireframe_long_leaf_vertex_count": best_candidate.mesh.metadata.get(
                    "wireframe_long_leaf_vertex_count"
                ),
                "variant_fusion_base": best_candidate.mesh.metadata.get("variant_fusion_base"),
                "variant_fusion_guide": best_candidate.mesh.metadata.get("variant_fusion_guide"),
                "variant_fusion_snap_count": best_candidate.mesh.metadata.get("variant_fusion_snap_count"),
                "variant_fusion_marker_snap_count": best_candidate.mesh.metadata.get("variant_fusion_marker_snap_count"),
                "variant_fusion_marker_preserve_count": best_candidate.mesh.metadata.get("variant_fusion_marker_preserve_count"),
                "variant_fusion_marker_refine_count": best_candidate.mesh.metadata.get("variant_fusion_marker_refine_count"),
                "final_marker_center_snap_count": best_candidate.mesh.metadata.get("final_marker_center_snap_count"),
                "final_marker_center_snaps": best_candidate.mesh.metadata.get("final_marker_center_snaps"),
            },
            "classical_anchor_candidate": (
                {
                    "name": classical_anchor.name,
                    "source": classical_anchor.source,
                    "proposal_family": classical_anchor.proposal_family,
                    "merge_threshold": classical_anchor.merge_threshold,
                    "ratio_to_triangle": classical_anchor.ratio_to_triangle,
                    "ratio_to_quad": classical_anchor.ratio_to_quad,
                    "counts": classical_anchor.mesh.counts(),
                    "validation_report": classical_anchor.validation_report,
                }
                if classical_anchor is not None
                else None
            ),
            "hawp_candidate_count": len(hawp_candidates),
            "deepvec_candidate_count": len(deepvec_candidates),
            "hybrid_candidate_count": 1 if hybrid_candidate is not None else 0,
            "candidate_leaderboard": candidate_leaderboard,
            "candidate_reports": candidate_leaderboard,
        }
        if truth_path is not None:
            image_summary["comparison"] = best_report
        summary[image_path.stem] = image_summary

    (output_root / "full_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_compare_mode(options: PipelineOptions) -> dict:
    from .compare import compare_nas_files, write_report

    output_root = Path(options.output_dir)
    summary: dict[str, dict] = {}
    for image_name in options.images:
        image_path = Path(image_name)
        truth_path = _truth_path(image_path, options.truth_dir)
        predicted_path = output_root / image_path.stem / f"{image_path.stem}.nas"
        if truth_path is None:
            raise FileNotFoundError(f"Ground truth NAS not found for {image_path.name}")
        if not predicted_path.exists():
            raise FileNotFoundError(f"Predicted NAS not found: {predicted_path}")
        report = compare_nas_files(predicted_path, truth_path)
        write_report(
            report,
            output_root / image_path.stem / "comparison.json",
            output_root / image_path.stem / "comparison.md",
        )
        summary[image_path.stem] = report

    (output_root / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
