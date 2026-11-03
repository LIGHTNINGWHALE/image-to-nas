from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .nas import NasMesh


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    polygon = np.asarray(points, dtype=float)
    if len(polygon) == 0:
        return np.zeros(2, dtype=float)
    if len(polygon) < 3:
        return polygon.mean(axis=0)

    x = polygon[:, 0]
    y = polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-8:
        return polygon.mean(axis=0)

    factor = 1.0 / (6.0 * area)
    centroid_x = float(np.sum((x + np.roll(x, -1)) * cross) * factor)
    centroid_y = float(np.sum((y + np.roll(y, -1)) * cross) * factor)
    return np.array([centroid_x, centroid_y], dtype=float)


def _sample_image_patch(
    image_rgb: np.ndarray,
    center_xy: np.ndarray,
    *,
    black_threshold: int,
    patch_radius: int,
) -> dict:
    height, width = image_rgb.shape[:2]
    x = float(center_xy[0])
    y = float(center_xy[1])
    x_idx = int(round(x))
    y_idx = int(round(y))
    in_bounds = 0 <= x_idx < width and 0 <= y_idx < height

    if not in_bounds:
        return {
            "in_bounds": False,
            "center_xy": [x, y],
            "center_rgb": None,
            "center_is_black": False,
            "patch_black_fraction": 0.0,
            "patch_mean_rgb": None,
        }

    center_rgb = image_rgb[y_idx, x_idx].astype(int).tolist()
    center_is_black = max(center_rgb) <= black_threshold

    x0 = max(0, x_idx - patch_radius)
    x1 = min(width, x_idx + patch_radius + 1)
    y0 = max(0, y_idx - patch_radius)
    y1 = min(height, y_idx + patch_radius + 1)
    patch = image_rgb[y0:y1, x0:x1]
    patch_black_mask = np.max(patch, axis=2) <= black_threshold
    patch_black_fraction = float(np.mean(patch_black_mask)) if patch.size else 0.0
    patch_mean_rgb = np.mean(patch.reshape(-1, 3), axis=0).round(3).tolist() if patch.size else None

    return {
        "in_bounds": True,
        "center_xy": [x, y],
        "center_rgb": center_rgb,
        "center_is_black": bool(center_is_black),
        "patch_black_fraction": patch_black_fraction,
        "patch_mean_rgb": patch_mean_rgb,
    }


def audit_mesh_elements_against_image(
    mesh: NasMesh,
    image_rgb: np.ndarray,
    *,
    black_threshold: int = 24,
    patch_radius: int = 2,
    min_black_fraction: float = 0.6,
) -> dict:
    node_lookup = {node.node_id: node for node in mesh.nodes}
    element_reports: list[dict] = []
    removed_ids: list[int] = []

    for element in mesh.elements:
        polygon = []
        valid_polygon = True
        for node_id in element.node_ids:
            node = node_lookup.get(node_id)
            if node is None or node.pixel_x is None or node.pixel_y is None:
                valid_polygon = False
                break
            polygon.append((float(node.pixel_x), float(node.pixel_y)))

        if not valid_polygon:
            element_reports.append(
                {
                    "element_id": int(element.element_id),
                    "card": element.card,
                    "node_ids": list(element.node_ids),
                    "centroid_px": None,
                    "sample": None,
                    "removed": False,
                    "reason": "missing_pixel_coords",
                }
            )
            continue

        polygon_points = np.asarray(polygon, dtype=float)
        centroid = _polygon_centroid(polygon_points)
        sample = _sample_image_patch(
            image_rgb,
            centroid,
            black_threshold=black_threshold,
            patch_radius=patch_radius,
        )

        removed = bool(
            sample["in_bounds"]
            and sample["center_is_black"]
            and sample["patch_black_fraction"] >= min_black_fraction
        )
        reason = (
            f"center_black_and_patch_black_fraction>={min_black_fraction:.2f}"
            if removed
            else "kept"
        )
        if removed:
            removed_ids.append(int(element.element_id))

        element_reports.append(
            {
                "element_id": int(element.element_id),
                "card": element.card,
                "node_ids": list(element.node_ids),
                "centroid_px": [float(centroid[0]), float(centroid[1])],
                "sample": sample,
                "removed": removed,
                "reason": reason,
            }
        )

    filtered_element_count = len(mesh.elements) - len(removed_ids)
    return {
        "config": {
            "black_threshold": int(black_threshold),
            "patch_radius": int(patch_radius),
            "min_black_fraction": float(min_black_fraction),
            "image_shape": [int(image_rgb.shape[0]), int(image_rgb.shape[1])],
        },
        "original_counts": mesh.counts(),
        "removed_element_ids": removed_ids,
        "removed_element_count": len(removed_ids),
        "kept_element_count": filtered_element_count,
        "elements": element_reports,
    }


def filter_black_center_elements(
    mesh: NasMesh,
    image_rgb: np.ndarray,
    *,
    black_threshold: int = 24,
    patch_radius: int = 2,
    min_black_fraction: float = 0.6,
) -> tuple[NasMesh, dict]:
    report = audit_mesh_elements_against_image(
        mesh,
        image_rgb,
        black_threshold=black_threshold,
        patch_radius=patch_radius,
        min_black_fraction=min_black_fraction,
    )
    removed_ids = set(report["removed_element_ids"])
    filtered_elements = [element for element in mesh.elements if element.element_id not in removed_ids]

    metadata = dict(mesh.metadata)
    metadata["element_pixel_filter"] = {
        "enabled": True,
        "black_threshold": int(black_threshold),
        "patch_radius": int(patch_radius),
        "min_black_fraction": float(min_black_fraction),
        "removed_element_ids": sorted(removed_ids),
        "removed_element_count": len(removed_ids),
        "original_element_count": len(mesh.elements),
        "filtered_element_count": len(filtered_elements),
    }

    filtered_mesh = NasMesh(
        nodes=list(mesh.nodes),
        elements=filtered_elements,
        metadata=metadata,
    )
    report["filtered_counts"] = filtered_mesh.counts()
    return filtered_mesh, report


def save_element_pixel_audit_overlay(
    image_rgb: np.ndarray,
    mesh: NasMesh,
    report: dict,
    save_path: str | Path,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    node_lookup = {node.node_id: node for node in mesh.nodes}
    removed_ids = set(report.get("removed_element_ids", []))

    for element_report in report.get("elements", []):
        if int(element_report["element_id"]) not in removed_ids:
            continue
        polygon = []
        for node_id in element_report["node_ids"]:
            node = node_lookup.get(int(node_id))
            if node is None or node.pixel_x is None or node.pixel_y is None:
                polygon = []
                break
            polygon.append((float(node.pixel_x), float(node.pixel_y)))
        if len(polygon) < 3:
            continue
        draw.polygon(polygon, outline=(255, 64, 64))
        centroid = element_report.get("centroid_px")
        if centroid is not None:
            radius = 4
            draw.ellipse(
                (
                    centroid[0] - radius,
                    centroid[1] - radius,
                    centroid[0] + radius,
                    centroid[1] + radius,
                ),
                fill=(255, 255, 255),
                outline=(255, 64, 64),
            )

    image.save(save_path)
    return save_path


def write_element_pixel_audit(
    report: dict,
    output_dir: str | Path,
    *,
    stem: str = "element_pixel_audit",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    txt_path = output_dir / f"{stem}.txt"

    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# Element Pixel Audit",
        "",
        f"- Removed elements: {report.get('removed_element_count', 0)}",
        f"- Kept elements: {report.get('kept_element_count', 0)}",
        f"- Original counts: {report.get('original_counts', {})}",
        f"- Filtered counts: {report.get('filtered_counts', {})}",
        f"- Filtered counts after compaction: {report.get('filtered_counts_after_compaction', report.get('filtered_counts', {}))}",
        "",
        "## Config",
        "",
        f"- Black threshold: {report['config']['black_threshold']}",
        f"- Patch radius: {report['config']['patch_radius']}",
        f"- Min black fraction: {report['config']['min_black_fraction']}",
        "",
        "## Removed Elements",
        "",
    ]

    removed = [element for element in report.get("elements", []) if element.get("removed")]
    if not removed:
        lines.append("No elements were removed.")
    else:
        lines.append("| Element | Card | Nodes | Centroid (px) | Center RGB | Patch black fraction |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for element in removed:
            sample = element.get("sample") or {}
            lines.append(
                "| "
                f"{element['element_id']} | "
                f"{element['card']} | "
                f"{element['node_ids']} | "
                f"{None if element['centroid_px'] is None else [round(v, 2) for v in element['centroid_px']]} | "
                f"{sample.get('center_rgb')} | "
                f"{sample.get('patch_black_fraction')} |"
            )

    md_path.write_text("\n".join(lines) + "\n")

    txt_lines = [
        "Element Pixel Audit",
        f"Removed elements: {report.get('removed_element_count', 0)}",
        f"Kept elements: {report.get('kept_element_count', 0)}",
        f"Original counts: {report.get('original_counts', {})}",
        f"Filtered counts: {report.get('filtered_counts', {})}",
        f"Filtered counts after compaction: {report.get('filtered_counts_after_compaction', report.get('filtered_counts', {}))}",
        "",
    ]
    if not removed:
        txt_lines.append("No elements were removed.")
    else:
        txt_lines.append("Removed elements:")
        for element in removed:
            sample = element.get("sample") or {}
            txt_lines.append(
                f"- element {element['element_id']} {element['card']} nodes={element['node_ids']} "
                f"centroid={element['centroid_px']} center_rgb={sample.get('center_rgb')} "
                f"patch_black_fraction={sample.get('patch_black_fraction')}"
            )
    txt_path.write_text("\n".join(txt_lines) + "\n")

    return {"json": json_path, "md": md_path, "txt": txt_path}
