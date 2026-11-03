from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, distance_matrix

from .axis import AxisCalibration
from .coordinate_correction import apply_fixed_coordinate_correction
from .geometry import (
    cluster_points,
    collapse_short_edges,
    contour_arc,
    dedupe_sequence,
    fit_line_tls,
    intersect_lines,
    nearest_point_index,
    order_polygon,
    polygon_area,
    rdp,
    unique_edges_from_cycles,
)
from .nas import Element, NasMesh, Node


@dataclass(slots=True)
class ComponentPolygon:
    component_id: int
    area_px: int
    contour_px: np.ndarray
    polygon_px: np.ndarray


@dataclass(slots=True)
class ReconstructionCandidate:
    name: str
    merge_threshold: float
    ratio_to_triangle: float
    ratio_to_quad: float
    mesh: NasMesh
    wireframe: dict
    source: str = "classical"
    proposal_family: str = "classical"
    structural_score: float | None = None
    solver_status: str | None = None
    validation_report: dict | None = None


def _yellow_component_mask(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    return (
        (image[:, :, 0] > 160)
        & (image[:, :, 1] > 160)
        & (image[:, :, 2] < 100)
    )


def _white_marker_points(image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
    image = np.asarray(image_rgb)
    white_mask = (
        (image[:, :, 0] > 215)
        & (image[:, :, 1] > 215)
        & (image[:, :, 2] > 215)
    )
    if not np.any(white_mask):
        return []

    labels, num_labels = ndi.label(white_mask)
    markers: list[tuple[np.ndarray, float]] = []
    for component_id in range(1, num_labels + 1):
        component_mask = labels == component_id
        area = int(component_mask.sum())
        if area < 8 or area > 320:
            continue
        ys, xs = np.where(component_mask)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        if max(width, height) > 24 or min(width, height) < 2:
            continue
        center = np.array([float(xs.mean()), float(ys.mean())], dtype=float)
        radius = 0.5 * max(width, height)
        center = _refine_white_marker_center(image, center, float(radius))
        markers.append((center, float(radius)))
    return markers


def _refine_white_marker_center(
    image_rgb: np.ndarray,
    seed_center: np.ndarray,
    seed_radius: float,
) -> np.ndarray:
    """Refine a white marker seed to the full anti-aliased disk center."""
    image = np.asarray(image_rgb)
    height, width = image.shape[:2]
    center = np.asarray(seed_center, dtype=float)
    cx, cy = float(center[0]), float(center[1])
    search_radius = max(7.0, float(seed_radius) + 4.0)
    x0 = max(0, int(math.floor(cx - search_radius - 1.0)))
    x1 = min(width, int(math.ceil(cx + search_radius + 2.0)))
    y0 = max(0, int(math.floor(cy - search_radius - 1.0)))
    y1 = min(height, int(math.ceil(cy + search_radius + 2.0)))
    if x1 <= x0 or y1 <= y0:
        return center

    patch = image[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance_to_seed = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    min_rgb = patch.min(axis=2)
    max_rgb = patch.max(axis=2)
    spread = max_rgb - min_rgb
    local_disk = distance_to_seed <= search_radius

    marker_mask = (min_rgb >= 135.0) & (spread <= 65.0) & local_disk
    if int(marker_mask.sum()) < 8:
        return center

    labels, _ = ndi.label(marker_mask)
    core_mask = (min_rgb > 215.0) & local_disk
    component_ids = labels[core_mask]
    component_ids = component_ids[component_ids > 0]
    if component_ids.size == 0:
        return center
    marker_mask = labels == int(np.bincount(component_ids).argmax())

    ys, xs = np.where(marker_mask)
    if len(xs) < 8:
        return center
    marker_width = int(xs.max() - xs.min() + 1)
    marker_height = int(ys.max() - ys.min() + 1)
    marker_area = int(marker_mask.sum())
    if marker_area > 140 or max(marker_width, marker_height) > 14:
        marker_mask = (min_rgb >= 165.0) & (spread <= 55.0) & local_disk
        labels, _ = ndi.label(marker_mask)
        component_ids = labels[core_mask]
        component_ids = component_ids[component_ids > 0]
        if component_ids.size == 0:
            return center
        marker_mask = labels == int(np.bincount(component_ids).argmax())
        ys, xs = np.where(marker_mask)
        if len(xs) < 8:
            return center

    brightness_weight = np.clip((min_rgb - 120.0) / 135.0, 0.0, 1.0) ** 2
    neutrality_weight = np.clip((80.0 - spread) / 80.0, 0.0, 1.0)
    weights = brightness_weight * neutrality_weight * marker_mask
    if float(weights.sum()) <= 1e-6:
        return center

    weighted_center = np.array(
        [
            float((xx * weights).sum() / weights.sum()),
            float((yy * weights).sum() / weights.sum()),
        ],
        dtype=float,
    )

    distance = ndi.distance_transform_edt(marker_mask)
    if float(distance.max()) > 0.0:
        peak_mask = distance >= max(0.5, 0.85 * float(distance.max()))
        peak_weights = distance * peak_mask
        if float(peak_weights.sum()) > 1e-6:
            plateau_center = np.array(
                [
                    float((xx * peak_weights).sum() / peak_weights.sum()),
                    float((yy * peak_weights).sum() / peak_weights.sum()),
                ],
                dtype=float,
            )
            refined = 0.65 * weighted_center + 0.35 * plateau_center
        else:
            refined = weighted_center
    else:
        refined = weighted_center

    delta = refined - center
    shift = float(np.linalg.norm(delta))
    max_shift = 2.0
    if shift <= 1e-6:
        return center
    if shift > max_shift:
        refined = center + delta / shift * max_shift
    return refined.astype(float)


def apply_candidate_coordinate_correction(
    candidate: ReconstructionCandidate,
    calibration: AxisCalibration,
    *,
    axis_stabilization_applied: bool,
) -> ReconstructionCandidate:
    if not candidate.mesh.nodes:
        candidate.mesh.metadata["coordinate_correction_mode"] = "raw"
        candidate.mesh.metadata["coordinate_correction_stats"] = {}
        return candidate

    node_pixel_array = np.array(
        [[float(node.pixel_x), float(node.pixel_y)] for node in candidate.mesh.nodes],
        dtype=float,
    )
    raw_world_xy = calibration.pixel_to_world(node_pixel_array)
    correction_result = apply_fixed_coordinate_correction(
        raw_world_xy,
        axis_stabilization_applied=axis_stabilization_applied,
    )
    corrected_world_xy = correction_result.corrected_world_xy

    nodes = [
        Node(
            node_id=node.node_id,
            x=float(corrected_world_xy[idx, 0]),
            y=float(corrected_world_xy[idx, 1]),
            z=float(node.z),
            pixel_x=float(node.pixel_x),
            pixel_y=float(node.pixel_y),
        )
        for idx, node in enumerate(candidate.mesh.nodes)
    ]
    candidate.mesh = NasMesh(
        nodes=nodes,
        elements=list(candidate.mesh.elements),
        metadata={
            **candidate.mesh.metadata,
            "coordinate_correction_mode": correction_result.mode,
            "coordinate_correction_stats": correction_result.to_dict(),
        },
    )
    return candidate


def _contour_to_polygon(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fig, axis = plt.subplots()
    contour_set = axis.contour(mask.astype(float), levels=[0.5])
    try:
        path = max(contour_set.get_paths(), key=lambda entry: len(entry.vertices))
        contour = path.vertices
    finally:
        plt.close(fig)

    hull = ConvexHull(contour)
    hull_points = contour[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])
    best = None
    for epsilon in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        simplified = rdp(hull_points, epsilon)
        if np.linalg.norm(simplified[0] - simplified[-1]) < 1e-6:
            simplified = simplified[:-1]
        if len(simplified) <= 5:
            return contour, _refine_polygon_corners(contour, order_polygon(simplified))
        best = order_polygon(simplified)
    if best is None:
        raise ValueError("Failed to extract a polygon from the connected component.")
    return contour, _refine_polygon_corners(contour, best)


def _refine_polygon_corners(contour: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    corner_indices = [nearest_point_index(contour, corner) for corner in polygon]
    order_idx = np.argsort(corner_indices)
    ordered_indices = [corner_indices[idx] for idx in order_idx]
    ordered_polygon = polygon[order_idx]

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    num_corners = len(ordered_polygon)
    for idx in range(num_corners):
        arc = contour_arc(contour, ordered_indices[idx], ordered_indices[(idx + 1) % num_corners])
        if len(arc) < 4:
            arc = np.vstack([ordered_polygon[idx], ordered_polygon[(idx + 1) % num_corners]])
        lines.append(fit_line_tls(arc))

    refined = np.array(
        [
            intersect_lines(lines[idx - 1], lines[idx])
            for idx in range(num_corners)
        ],
        dtype=float,
    )
    return order_polygon(refined)


def extract_component_polygons(
    mesh_image_rgb: np.ndarray,
    debug_dir: Path | None = None,
) -> list[ComponentPolygon]:
    yellow_mask = _yellow_component_mask(mesh_image_rgb)
    labels, num_labels = ndi.label(yellow_mask)
    components: list[ComponentPolygon] = []

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        color_mask = np.zeros((*labels.shape, 3), dtype=np.uint8)

    color_table = np.array(
        [
            [255, 179, 0],
            [128, 62, 117],
            [255, 104, 0],
            [166, 189, 215],
            [193, 0, 32],
            [206, 162, 98],
            [129, 112, 102],
            [0, 125, 52],
            [246, 118, 142],
            [0, 83, 138],
            [255, 122, 92],
            [83, 55, 122],
            [255, 142, 0],
            [179, 40, 81],
            [244, 200, 0],
            [127, 24, 13],
            [147, 170, 0],
            [89, 51, 21],
            [241, 58, 19],
            [35, 44, 22],
        ],
        dtype=np.uint8,
    )

    for component_id in range(1, num_labels + 1):
        component_mask = labels == component_id
        area = int(component_mask.sum())
        if area <= 300:
            continue

        ys, xs = np.where(component_mask)
        sub_slice = np.s_[ys.min() - 2 : ys.max() + 3, xs.min() - 2 : xs.max() + 3]
        contour, polygon = _contour_to_polygon(component_mask[sub_slice])
        contour[:, 0] += xs.min() - 2
        contour[:, 1] += ys.min() - 2
        polygon[:, 0] += xs.min() - 2
        polygon[:, 1] += ys.min() - 2
        components.append(
            ComponentPolygon(
                component_id=component_id,
                area_px=area,
                contour_px=contour,
                polygon_px=order_polygon(polygon),
            )
        )

        if debug_dir is not None:
            color_mask[component_mask] = color_table[(component_id - 1) % len(color_table)]

    if debug_dir is not None:
        Image.fromarray(color_mask, mode="RGB").save(debug_dir / "component_mask.png")
        overlay = Image.fromarray(np.asarray(mesh_image_rgb).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(overlay)
        for component in components:
            pts = [tuple(point) for point in component.polygon_px.tolist()]
            draw.line(pts + [pts[0]], fill=(0, 255, 255), width=3)
            centroid = component.polygon_px.mean(axis=0)
            draw.text((centroid[0], centroid[1]), str(component.component_id), fill=(255, 255, 255))
        overlay.save(debug_dir / "component_polygons.png")

    return components


def _canonical_cycle(cycle: list[int]) -> tuple[int, ...] | None:
    if len(cycle) < 3 or len(set(cycle)) != len(cycle):
        return None

    minimum = min(cycle)
    variants: list[tuple[int, ...]] = []
    for sequence in (cycle, list(reversed(cycle))):
        for idx, value in enumerate(sequence):
            if value == minimum:
                variants.append(tuple(sequence[idx:] + sequence[:idx]))
    return min(variants) if variants else None


def _wireframe_subgraph(
    wireframe: dict,
    edge_threshold: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    vertices = np.array(wireframe["vertices"], dtype=float)
    edges = np.array(wireframe["edges"], dtype=int)
    weights = np.array(wireframe["edges-weights"], dtype=float)

    keep = weights >= edge_threshold
    if not np.any(keep):
        return np.empty((0, 2), dtype=float), [], []

    filtered_edges = edges[keep]
    filtered_weights = weights[keep]
    used_vertices = np.unique(filtered_edges.reshape(-1))
    remap = {int(vertex_idx): new_idx for new_idx, vertex_idx in enumerate(used_vertices.tolist())}
    remapped_edges = [
        (remap[int(left)], remap[int(right)])
        for left, right in filtered_edges.tolist()
        if int(left) in remap and int(right) in remap
    ]
    remapped_weights = filtered_weights.tolist()
    return vertices[used_vertices], remapped_edges, remapped_weights


def _reindex_wireframe_graph(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) == 0 or not edges:
        return np.empty((0, 2), dtype=float), [], []

    merged_weights: dict[tuple[int, int], float] = {}
    for (left, right), weight in zip(edges, edge_weights):
        left = int(left)
        right = int(right)
        if left == right:
            continue
        edge = tuple(sorted((left, right)))
        merged_weights[edge] = max(float(weight), merged_weights.get(edge, 0.0))

    if not merged_weights:
        return np.empty((0, 2), dtype=float), [], []

    used_vertices = sorted({vertex_idx for edge in merged_weights for vertex_idx in edge})
    remap = {vertex_idx: new_idx for new_idx, vertex_idx in enumerate(used_vertices)}
    reindexed_vertices = np.asarray(vertices_px, dtype=float)[used_vertices]
    reindexed_edges = [(remap[left], remap[right]) for left, right in sorted(merged_weights)]
    reindexed_weights = [float(merged_weights[edge]) for edge in sorted(merged_weights)]
    return reindexed_vertices, reindexed_edges, reindexed_weights


def _snap_wireframe_vertices(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    snap_distance_px: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) < 2 or snap_distance_px <= 0:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    clustered_vertices, cluster_ids = cluster_points(vertices_px, snap_distance_px)
    clustered_edges = [
        (cluster_ids[left], cluster_ids[right])
        for left, right in edges
        if cluster_ids[left] != cluster_ids[right]
    ]
    return _reindex_wireframe_graph(clustered_vertices, clustered_edges, edge_weights)


def _filter_wireframe_edges_by_length(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    min_length_px: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if not edges:
        return np.empty((0, 2), dtype=float), [], []

    kept_edges: list[tuple[int, int]] = []
    kept_weights: list[float] = []
    for edge, weight in zip(edges, edge_weights):
        left, right = edge
        length = float(np.linalg.norm(vertices_px[left] - vertices_px[right]))
        if length < min_length_px:
            continue
        kept_edges.append(edge)
        kept_weights.append(float(weight))
    return _reindex_wireframe_graph(vertices_px, kept_edges, kept_weights)


def _collapse_duplicate_bridge_edges(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    max_bridge_length_px: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) < 2 or not edges or max_bridge_length_px <= 0:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    current_vertices, current_edges, current_weights = _reindex_wireframe_graph(vertices_px, edges, edge_weights)
    while current_edges:
        adjacency = _wireframe_adjacency(len(current_vertices), current_edges)
        best_score: tuple[float, int, int] | None = None
        collapse_left = -1
        collapse_right = -1
        for edge_idx, (left, right) in enumerate(current_edges):
            length = float(np.linalg.norm(current_vertices[left] - current_vertices[right]))
            if length > max_bridge_length_px:
                continue
            shared_neighbors = adjacency[left] & adjacency[right]
            if len(shared_neighbors) < 2:
                continue
            score = (length, -len(shared_neighbors), edge_idx)
            if best_score is None or score < best_score:
                best_score = score
                collapse_left = left
                collapse_right = right

        if best_score is None:
            break

        merged_vertices = np.asarray(current_vertices, dtype=float).copy()
        merged_vertices[collapse_left] = 0.5 * (
            current_vertices[collapse_left] + current_vertices[collapse_right]
        )

        rewired_edges: list[tuple[int, int]] = []
        rewired_weights: list[float] = []
        for (left, right), weight in zip(current_edges, current_weights):
            new_left = collapse_left if left == collapse_right else left
            new_right = collapse_left if right == collapse_right else right
            if new_left == new_right:
                continue
            rewired_edges.append((new_left, new_right))
            rewired_weights.append(float(weight))
        current_vertices, current_edges, current_weights = _reindex_wireframe_graph(
            merged_vertices,
            rewired_edges,
            rewired_weights,
        )

    return current_vertices, current_edges, current_weights


def _collapse_vertices_to_markers(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    marker_points: list[tuple[np.ndarray, float]],
    *,
    merge_margin_px: float,
    max_chain_edge_px: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) < 2 or not edges or not marker_points:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    current_vertices, current_edges, current_weights = _reindex_wireframe_graph(vertices_px, edges, edge_weights)
    for marker_center, marker_radius in marker_points:
        if len(current_vertices) < 2 or not current_edges:
            break

        adjacency = _wireframe_adjacency(len(current_vertices), current_edges)
        base_radius = max(merge_margin_px, marker_radius + merge_margin_px)
        expand_radius = max(base_radius * 1.5, base_radius + max_chain_edge_px)
        cluster = {
            vertex_idx
            for vertex_idx, point in enumerate(current_vertices)
            if float(np.linalg.norm(point - marker_center)) <= base_radius
        }
        if len(cluster) < 2:
            continue

        queue = list(cluster)
        while queue:
            vertex_idx = queue.pop()
            for neighbor_idx in adjacency.get(vertex_idx, ()):
                if neighbor_idx in cluster:
                    continue
                edge_length = float(np.linalg.norm(current_vertices[vertex_idx] - current_vertices[neighbor_idx]))
                if edge_length > max_chain_edge_px:
                    continue
                if float(np.linalg.norm(current_vertices[neighbor_idx] - marker_center)) > expand_radius:
                    continue
                cluster.add(neighbor_idx)
                queue.append(neighbor_idx)

        if len(cluster) < 2:
            continue

        outside_neighbors = {
            neighbor_idx
            for vertex_idx in cluster
            for neighbor_idx in adjacency.get(vertex_idx, ())
            if neighbor_idx not in cluster
        }
        if len(outside_neighbors) < 2:
            continue
        if len(cluster) > 6:
            continue

        representative = min(cluster)
        cluster_points = current_vertices[sorted(cluster)]
        merged_point = 0.7 * marker_center + 0.3 * np.mean(cluster_points, axis=0)
        merged_vertices = np.asarray(current_vertices, dtype=float).copy()
        merged_vertices[representative] = merged_point

        rewired_edges: list[tuple[int, int]] = []
        rewired_weights: list[float] = []
        for (left, right), weight in zip(current_edges, current_weights):
            new_left = representative if left in cluster else left
            new_right = representative if right in cluster else right
            if new_left == new_right:
                continue
            rewired_edges.append((new_left, new_right))
            rewired_weights.append(float(weight))
        current_vertices, current_edges, current_weights = _reindex_wireframe_graph(
            merged_vertices,
            rewired_edges,
            rewired_weights,
        )

    return current_vertices, current_edges, current_weights


def _wireframe_adjacency(
    n_vertices: int,
    edges: list[tuple[int, int]],
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(n_vertices)}
    for left, right in edges:
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _wireframe_edge_weight_map(
    edges: list[tuple[int, int]],
    edge_weights: list[float],
) -> dict[tuple[int, int], float]:
    return {
        tuple(sorted((int(left), int(right)))): float(weight)
        for (left, right), weight in zip(edges, edge_weights)
        if int(left) != int(right)
    }


def _wireframe_leaf_stats(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    long_edge_threshold_px: float = 45.0,
) -> dict[str, object]:
    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    leaf_edges: list[dict[str, float | int]] = []
    long_leaf_edges: list[dict[str, float | int]] = []
    for vertex_idx, neighbors in adjacency.items():
        if len(neighbors) != 1:
            continue
        neighbor_idx = next(iter(neighbors))
        length_px = float(np.linalg.norm(vertices_px[vertex_idx] - vertices_px[neighbor_idx]))
        entry = {
            "vertex": int(vertex_idx),
            "neighbor": int(neighbor_idx),
            "length_px": length_px,
        }
        leaf_edges.append(entry)
        if length_px >= long_edge_threshold_px:
            long_leaf_edges.append(entry)

    return {
        "wireframe_leaf_vertex_count": int(len(leaf_edges)),
        "wireframe_long_leaf_vertex_count": int(len(long_leaf_edges)),
        "wireframe_leaf_long_edge_threshold_px": float(long_edge_threshold_px),
        "wireframe_long_leaf_edges": long_leaf_edges,
    }


def _project_point_to_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        projection = np.array(start, dtype=float)
        return 0.0, projection, float(np.linalg.norm(point - projection))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return t, projection, float(np.linalg.norm(point - projection))


def _snap_vertices_to_supporting_edges(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    max_distance_px: float,
    min_param_margin: float = 0.06,
) -> np.ndarray:
    if len(vertices_px) < 3 or not edges or max_distance_px <= 0:
        return np.asarray(vertices_px, dtype=float)

    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    snapped_vertices = np.asarray(vertices_px, dtype=float).copy()
    source_vertices = np.asarray(vertices_px, dtype=float)

    for vertex_idx, point in enumerate(source_vertices):
        degree = len(adjacency.get(vertex_idx, ()))
        best_projection: np.ndarray | None = None
        best_key: tuple[float, float, float] | None = None

        for edge, weight in zip(edges, edge_weights):
            left, right = edge
            if vertex_idx in (left, right):
                continue
            start = source_vertices[left]
            end = source_vertices[right]
            edge_length = float(np.linalg.norm(end - start))
            if edge_length <= max_distance_px * 1.25:
                continue
            t, projection, distance = _project_point_to_segment(point, start, end)
            if t <= min_param_margin or t >= 1.0 - min_param_margin:
                continue
            if distance > max_distance_px:
                continue
            key = (distance, -float(weight), -edge_length)
            if best_key is None or key < best_key:
                best_key = key
                best_projection = projection

        if best_projection is None:
            continue

        alpha = 0.9 if degree <= 2 else 0.65
        snapped_vertices[vertex_idx] = (1.0 - alpha) * point + alpha * best_projection

    return snapped_vertices


def _split_edges_at_existing_vertices(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    point_tolerance_px: float,
    min_param_margin: float = 0.05,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) < 3 or not edges:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    vertices_px = np.asarray(vertices_px, dtype=float)
    split_edges: list[tuple[int, int]] = []
    split_weights: list[float] = []

    for edge, weight in zip(edges, edge_weights):
        left, right = edge
        start = vertices_px[left]
        end = vertices_px[right]
        breakpoints: list[tuple[float, int]] = [(0.0, int(left)), (1.0, int(right))]

        for vertex_idx, point in enumerate(vertices_px):
            if vertex_idx in (left, right):
                continue
            t, _, distance = _project_point_to_segment(point, start, end)
            if t <= min_param_margin or t >= 1.0 - min_param_margin:
                continue
            if distance > point_tolerance_px:
                continue
            breakpoints.append((t, int(vertex_idx)))

        breakpoints.sort(key=lambda item: (item[0], item[1]))
        deduped: list[tuple[float, int]] = []
        for t, vertex_idx in breakpoints:
            if deduped and deduped[-1][1] == vertex_idx:
                continue
            deduped.append((t, vertex_idx))

        for (_, edge_left), (_, edge_right) in zip(deduped, deduped[1:]):
            if edge_left == edge_right:
                continue
            split_edges.append((edge_left, edge_right))
            split_weights.append(float(weight))

    return _reindex_wireframe_graph(vertices_px, split_edges, split_weights)


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _segment_intersection(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
    min_param_margin: float,
) -> tuple[float, float, np.ndarray] | None:
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    determinant = _cross_2d(direction_a, direction_b)
    if abs(determinant) <= 1e-6:
        return None

    offset = start_b - start_a
    t = _cross_2d(offset, direction_b) / determinant
    u = _cross_2d(offset, direction_a) / determinant
    if t <= min_param_margin or t >= 1.0 - min_param_margin:
        return None
    if u <= min_param_margin or u >= 1.0 - min_param_margin:
        return None

    intersection = start_a + t * direction_a
    return float(t), float(u), np.asarray(intersection, dtype=float)


def _split_edges_at_intersections(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    merge_distance_px: float,
    min_param_margin: float = 0.04,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) < 4 or len(edges) < 2:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    raw_vertices = [np.asarray(point, dtype=float) for point in np.asarray(vertices_px, dtype=float)]
    split_refs: dict[int, list[tuple[float, int]]] = {
        edge_idx: [(0.0, int(edge[0])), (1.0, int(edge[1]))]
        for edge_idx, edge in enumerate(edges)
    }
    added_intersections = False

    for left_idx in range(len(edges)):
        start_a_idx, end_a_idx = edges[left_idx]
        if start_a_idx == end_a_idx:
            continue
        start_a = np.asarray(vertices_px[start_a_idx], dtype=float)
        end_a = np.asarray(vertices_px[end_a_idx], dtype=float)
        for right_idx in range(left_idx + 1, len(edges)):
            start_b_idx, end_b_idx = edges[right_idx]
            if start_b_idx == end_b_idx:
                continue
            if {start_a_idx, end_a_idx}.intersection({start_b_idx, end_b_idx}):
                continue
            start_b = np.asarray(vertices_px[start_b_idx], dtype=float)
            end_b = np.asarray(vertices_px[end_b_idx], dtype=float)
            intersection = _segment_intersection(
                start_a,
                end_a,
                start_b,
                end_b,
                min_param_margin=min_param_margin,
            )
            if intersection is None:
                continue
            t_a, t_b, point = intersection
            point_idx = len(raw_vertices)
            raw_vertices.append(point)
            split_refs[left_idx].append((t_a, point_idx))
            split_refs[right_idx].append((t_b, point_idx))
            added_intersections = True

    if not added_intersections:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    raw_vertices_array = np.asarray(raw_vertices, dtype=float)
    if merge_distance_px > 0:
        merged_vertices, cluster_ids = cluster_points(raw_vertices_array, merge_distance_px)
    else:
        merged_vertices = raw_vertices_array
        cluster_ids = list(range(len(raw_vertices_array)))

    split_edges: list[tuple[int, int]] = []
    split_weights: list[float] = []
    for edge_idx, refs in split_refs.items():
        ordered_refs = sorted(
            ((float(t), int(cluster_ids[vertex_idx])) for t, vertex_idx in refs),
            key=lambda item: (item[0], item[1]),
        )
        deduped: list[tuple[float, int]] = []
        for t, vertex_idx in ordered_refs:
            if deduped and deduped[-1][1] == vertex_idx:
                continue
            deduped.append((t, vertex_idx))
        for (_, left), (_, right) in zip(deduped, deduped[1:]):
            if left == right:
                continue
            split_edges.append((left, right))
            split_weights.append(float(edge_weights[edge_idx]))

    return _reindex_wireframe_graph(merged_vertices, split_edges, split_weights)


def _is_nearly_collinear_vertex(
    vertex_idx: int,
    vertices_px: np.ndarray,
    adjacency: dict[int, set[int]],
    angle_tolerance_deg: float,
) -> bool:
    neighbors = list(adjacency.get(vertex_idx, ()))
    if len(neighbors) != 2:
        return False

    left, right = neighbors
    vec_left = vertices_px[left] - vertices_px[vertex_idx]
    vec_right = vertices_px[right] - vertices_px[vertex_idx]
    norm_left = float(np.linalg.norm(vec_left))
    norm_right = float(np.linalg.norm(vec_right))
    if norm_left <= 1e-6 or norm_right <= 1e-6:
        return True

    cosine = float(np.dot(vec_left, vec_right) / (norm_left * norm_right))
    angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
    return abs(angle - 180.0) <= angle_tolerance_deg


def _vertex_pair_angle_deg(
    vertex_idx: int,
    first_neighbor_idx: int,
    second_neighbor_idx: int,
    vertices_px: np.ndarray,
) -> float:
    vec_left = vertices_px[first_neighbor_idx] - vertices_px[vertex_idx]
    vec_right = vertices_px[second_neighbor_idx] - vertices_px[vertex_idx]
    norm_left = float(np.linalg.norm(vec_left))
    norm_right = float(np.linalg.norm(vec_right))
    if norm_left <= 1e-6 or norm_right <= 1e-6:
        return 180.0

    cosine = float(np.dot(vec_left, vec_right) / (norm_left * norm_right))
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def _find_mixed_confidence_split_vertices(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    edge_threshold: float,
) -> list[dict]:
    if len(vertices_px) == 0 or len(edges) == 0:
        return []

    adjacency_edges: dict[int, list[tuple[int, int, float]]] = {idx: [] for idx in range(len(vertices_px))}
    for edge_idx, ((left, right), weight) in enumerate(zip(edges, edge_weights)):
        edge_weight = float(weight)
        adjacency_edges[left].append((edge_idx, right, edge_weight))
        adjacency_edges[right].append((edge_idx, left, edge_weight))

    all_weights = np.asarray(edge_weights, dtype=float)
    low_threshold = max(float(edge_threshold) * 2.0, 0.20)
    high_threshold = max(0.80, float(np.percentile(all_weights, 75)))
    suspicious: list[dict] = []

    for vertex_idx, incident in adjacency_edges.items():
        if len(incident) != 4:
            continue
        ordered = sorted(incident, key=lambda item: item[2])
        low_edges = ordered[:2]
        high_edges = ordered[2:]
        low_max = max(item[2] for item in low_edges)
        high_min = min(item[2] for item in high_edges)
        if low_max > low_threshold:
            continue
        if high_min < high_threshold:
            continue
        if (float(np.mean([item[2] for item in high_edges])) - float(np.mean([item[2] for item in low_edges]))) < 0.45:
            continue

        low_angle = _vertex_pair_angle_deg(vertex_idx, low_edges[0][1], low_edges[1][1], vertices_px)
        high_angle = _vertex_pair_angle_deg(vertex_idx, high_edges[0][1], high_edges[1][1], vertices_px)
        if low_angle < 150.0 or high_angle < 150.0:
            continue

        suspicious.append(
            {
                "vertex_idx": int(vertex_idx),
                "low_edge_indices": [int(item[0]) for item in low_edges],
                "high_edge_indices": [int(item[0]) for item in high_edges],
                "low_edge_max": float(low_max),
                "high_edge_min": float(high_min),
                "low_angle_deg": float(low_angle),
                "high_angle_deg": float(high_angle),
            }
        )

    return suspicious


def _prune_mixed_confidence_split_vertices(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    edge_threshold: float,
    max_spur_length_px: float,
    min_keep_score: float,
    angle_tolerance_deg: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], dict]:
    suspicious = _find_mixed_confidence_split_vertices(
        vertices_px,
        edges,
        edge_weights,
        edge_threshold=edge_threshold,
    )
    if not suspicious:
        return vertices_px, edges, edge_weights, {
            "mixed_confidence_pruned_vertex_count": 0,
            "mixed_confidence_pruned_vertices": [],
        }

    remove_edge_indices = {
        edge_idx
        for item in suspicious
        for edge_idx in item["low_edge_indices"]
    }
    kept_edges: list[tuple[int, int]] = []
    kept_weights: list[float] = []
    for edge_idx, (edge, weight) in enumerate(zip(edges, edge_weights)):
        if edge_idx in remove_edge_indices:
            continue
        kept_edges.append(edge)
        kept_weights.append(float(weight))

    pruned_vertices_px, pruned_edges, pruned_weights = _reindex_wireframe_graph(
        vertices_px,
        kept_edges,
        kept_weights,
    )
    pruned_vertices_px, pruned_edges, pruned_weights = _prune_spur_edges(
        pruned_vertices_px,
        pruned_edges,
        pruned_weights,
        max_spur_length_px=max_spur_length_px,
        min_keep_score=min_keep_score,
        max_iterations=4,
    )
    pruned_vertices_px, pruned_edges, pruned_weights = _compress_collinear_wireframe(
        pruned_vertices_px,
        pruned_edges,
        pruned_weights,
        angle_tolerance_deg=angle_tolerance_deg,
    )
    return pruned_vertices_px, pruned_edges, pruned_weights, {
        "mixed_confidence_pruned_vertex_count": int(len(suspicious)),
        "mixed_confidence_pruned_vertices": [int(item["vertex_idx"]) + 1 for item in suspicious],
    }


def _prune_boundary_ear_vertices(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    edge_threshold: float,
    marker_points: list[tuple[np.ndarray, float]] | None = None,
    require_global_min: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], dict]:
    if len(vertices_px) < 3 or len(edges) < 3:
        return vertices_px, edges, edge_weights, {
            "boundary_ear_pruned_vertex_count": 0,
            "boundary_ear_pruned_vertices": [],
        }

    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    edge_weight_map = _wireframe_edge_weight_map(edges, edge_weights)
    low_threshold = max(float(edge_threshold) * 3.0, 0.20)
    global_min_weight = float(np.min(np.asarray(edge_weights, dtype=float)))
    remove_edge_indices: set[int] = set()
    removed_vertices: list[int] = []

    for vertex_idx in range(len(vertices_px)):
        neighbors = sorted(adjacency.get(vertex_idx, ()))
        if len(neighbors) != 2:
            continue
        left, right = neighbors
        chord_edge = tuple(sorted((left, right)))
        chord_weight = edge_weight_map.get(chord_edge)
        if chord_weight is None:
            continue

        left_edge = tuple(sorted((vertex_idx, left)))
        right_edge = tuple(sorted((vertex_idx, right)))
        left_weight = edge_weight_map.get(left_edge, 0.0)
        right_weight = edge_weight_map.get(right_edge, 0.0)
        if require_global_min:
            if max(left_weight, right_weight) > global_min_weight + 1e-6:
                continue
        if max(left_weight, right_weight) > low_threshold:
            continue
        if float(chord_weight) < max(left_weight, right_weight) + 0.08:
            continue

        angle_deg = _vertex_pair_angle_deg(vertex_idx, left, right, vertices_px)
        if angle_deg < 155.0:
            continue

        point = vertices_px[vertex_idx]
        chord = vertices_px[right] - vertices_px[left]
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= 1e-6:
            continue
        height_to_chord = abs(float(np.cross(vertices_px[left] - point, vertices_px[right] - point))) / chord_length
        if height_to_chord > 3.0:
            continue

        if marker_points:
            anchored = False
            for marker_center, marker_radius in marker_points:
                if float(np.linalg.norm(point - marker_center)) <= marker_radius + 3.0:
                    anchored = True
                    break
            if anchored:
                continue

        removed_vertices.append(int(vertex_idx) + 1)
        for edge_idx, (edge, _) in enumerate(zip(edges, edge_weights)):
            if tuple(sorted(edge)) in {left_edge, right_edge}:
                remove_edge_indices.add(edge_idx)

    if not remove_edge_indices:
        return vertices_px, edges, edge_weights, {
            "boundary_ear_pruned_vertex_count": 0,
            "boundary_ear_pruned_vertices": [],
        }

    kept_edges: list[tuple[int, int]] = []
    kept_weights: list[float] = []
    for edge_idx, (edge, weight) in enumerate(zip(edges, edge_weights)):
        if edge_idx in remove_edge_indices:
            continue
        kept_edges.append(edge)
        kept_weights.append(float(weight))

    pruned_vertices_px, pruned_edges, pruned_weights = _reindex_wireframe_graph(
        vertices_px,
        kept_edges,
        kept_weights,
    )
    return pruned_vertices_px, pruned_edges, pruned_weights, {
        "boundary_ear_pruned_vertex_count": int(len(removed_vertices)),
        "boundary_ear_pruned_vertices": removed_vertices,
    }


def _compress_collinear_wireframe(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    angle_tolerance_deg: float = 10.0,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    if len(vertices_px) < 3 or len(edges) < 2:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    edge_weight_map = _wireframe_edge_weight_map(edges, edge_weights)
    keep_vertex = [
        len(adjacency[idx]) != 2 or not _is_nearly_collinear_vertex(idx, vertices_px, adjacency, angle_tolerance_deg)
        for idx in range(len(vertices_px))
    ]
    if sum(keep_vertex) < 2:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    compressed_edges: dict[tuple[int, int], float] = {}
    visited_directed: set[tuple[int, int]] = set()

    for start_idx, should_keep in enumerate(keep_vertex):
        if not should_keep:
            continue
        for neighbor_idx in adjacency[start_idx]:
            if (start_idx, neighbor_idx) in visited_directed:
                continue

            prev_idx = start_idx
            current_idx = neighbor_idx
            path_weight = edge_weight_map.get(tuple(sorted((start_idx, neighbor_idx))), 0.0)
            visited_directed.add((start_idx, neighbor_idx))

            while not keep_vertex[current_idx]:
                next_candidates = [candidate for candidate in adjacency[current_idx] if candidate != prev_idx]
                if len(next_candidates) != 1:
                    break
                next_idx = next_candidates[0]
                visited_directed.add((current_idx, next_idx))
                path_weight = max(
                    path_weight,
                    edge_weight_map.get(tuple(sorted((current_idx, next_idx))), 0.0),
                )
                prev_idx, current_idx = current_idx, next_idx

            if start_idx == current_idx:
                continue

            edge = tuple(sorted((start_idx, current_idx)))
            compressed_edges[edge] = max(path_weight, compressed_edges.get(edge, 0.0))

    if not compressed_edges:
        return _reindex_wireframe_graph(vertices_px, edges, edge_weights)

    return _reindex_wireframe_graph(
        vertices_px,
        list(compressed_edges),
        [compressed_edges[edge] for edge in compressed_edges],
    )


def _compress_weak_collinear_bridge_chains(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    max_edge_weight: float = 0.35,
    angle_tolerance_deg: float = 8.0,
    max_short_edge_px: float = 25.0,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], dict]:
    if len(vertices_px) < 3 or len(edges) < 2:
        return vertices_px, edges, edge_weights, {
            "weak_collinear_bridge_compressed_vertex_count": 0,
            "weak_collinear_bridge_compressed_vertices": [],
        }

    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    edge_weight_map = _wireframe_edge_weight_map(edges, edge_weights)
    compress_vertex = [False for _ in range(len(vertices_px))]

    for vertex_idx, neighbors in adjacency.items():
        if len(neighbors) != 2:
            continue
        left, right = sorted(neighbors)
        left_edge = tuple(sorted((vertex_idx, left)))
        right_edge = tuple(sorted((vertex_idx, right)))
        left_weight = float(edge_weight_map.get(left_edge, 1.0))
        right_weight = float(edge_weight_map.get(right_edge, 1.0))
        if max(left_weight, right_weight) > max_edge_weight:
            continue
        left_len = float(np.linalg.norm(vertices_px[vertex_idx] - vertices_px[left]))
        right_len = float(np.linalg.norm(vertices_px[vertex_idx] - vertices_px[right]))
        if min(left_len, right_len) > max_short_edge_px:
            continue
        if not _is_nearly_collinear_vertex(vertex_idx, vertices_px, adjacency, angle_tolerance_deg):
            continue
        compress_vertex[vertex_idx] = True

    # HAWP sometimes turns one faint straight boundary into a short zig-zag chain.
    # Local angle tests miss it when several sub-pixel marker points create alternating
    # 120-160 degree bends, so also compress globally straight low-confidence chains.
    for start_idx, neighbors in adjacency.items():
        for first_neighbor_idx in neighbors:
            first_edge = tuple(sorted((start_idx, first_neighbor_idx)))
            if float(edge_weight_map.get(first_edge, 1.0)) > max_edge_weight:
                continue

            path = [start_idx, first_neighbor_idx]
            prev_idx = start_idx
            current_idx = first_neighbor_idx
            seen = {start_idx}
            path_ok = True
            while True:
                if current_idx in seen:
                    path_ok = False
                    break
                seen.add(current_idx)
                current_neighbors = sorted(adjacency[current_idx])
                if len(current_neighbors) != 2:
                    break

                current_edges = [tuple(sorted((current_idx, neighbor))) for neighbor in current_neighbors]
                current_weights = [float(edge_weight_map.get(edge, 1.0)) for edge in current_edges]
                if max(current_weights) > max_edge_weight:
                    break

                next_candidates = [neighbor for neighbor in current_neighbors if neighbor != prev_idx]
                if len(next_candidates) != 1:
                    path_ok = False
                    break
                next_idx = next_candidates[0]
                path.append(next_idx)
                prev_idx, current_idx = current_idx, next_idx

            intermediates = path[1:-1]
            if not path_ok or len(intermediates) < 2:
                continue
            if any(len(adjacency[vertex_idx]) != 2 for vertex_idx in intermediates):
                continue

            start = vertices_px[path[0]]
            end = vertices_px[path[-1]]
            chord = end - start
            chord_len = float(np.linalg.norm(chord))
            if chord_len < 35.0:
                continue
            path_len = float(
                sum(np.linalg.norm(vertices_px[left] - vertices_px[right]) for left, right in zip(path[:-1], path[1:]))
            )
            if path_len / max(chord_len, 1e-6) > 1.08:
                continue

            denom = float(np.dot(chord, chord))
            max_deviation = 0.0
            for vertex_idx in intermediates:
                point = vertices_px[vertex_idx]
                t = float(np.clip(np.dot(point - start, chord) / denom, 0.0, 1.0))
                projection = start + t * chord
                max_deviation = max(max_deviation, float(np.linalg.norm(point - projection)))
            if max_deviation > 4.0:
                continue

            for vertex_idx in intermediates:
                compress_vertex[vertex_idx] = True

    if not any(compress_vertex):
        return vertices_px, edges, edge_weights, {
            "weak_collinear_bridge_compressed_vertex_count": 0,
            "weak_collinear_bridge_compressed_vertices": [],
        }

    compressed_edges: dict[tuple[int, int], float] = {}
    visited_directed: set[tuple[int, int]] = set()

    for start_idx, should_compress in enumerate(compress_vertex):
        if should_compress:
            continue
        for neighbor_idx in adjacency[start_idx]:
            if (start_idx, neighbor_idx) in visited_directed:
                continue

            prev_idx = start_idx
            current_idx = neighbor_idx
            path_weight = float(edge_weight_map.get(tuple(sorted((start_idx, neighbor_idx))), 0.0))
            visited_directed.add((start_idx, neighbor_idx))

            while compress_vertex[current_idx]:
                next_candidates = [candidate for candidate in adjacency[current_idx] if candidate != prev_idx]
                if len(next_candidates) != 1:
                    break
                next_idx = next_candidates[0]
                visited_directed.add((current_idx, next_idx))
                path_weight = max(
                    path_weight,
                    float(edge_weight_map.get(tuple(sorted((current_idx, next_idx))), 0.0)),
                )
                prev_idx, current_idx = current_idx, next_idx

            if start_idx == current_idx:
                continue
            edge = tuple(sorted((start_idx, current_idx)))
            compressed_edges[edge] = max(path_weight, compressed_edges.get(edge, 0.0))

    if not compressed_edges:
        return vertices_px, edges, edge_weights, {
            "weak_collinear_bridge_compressed_vertex_count": 0,
            "weak_collinear_bridge_compressed_vertices": [],
        }

    return_vertices, return_edges, return_weights = _reindex_wireframe_graph(
        vertices_px,
        list(compressed_edges),
        [compressed_edges[edge] for edge in compressed_edges],
    )
    return return_vertices, return_edges, return_weights, {
        "weak_collinear_bridge_compressed_vertex_count": int(sum(compress_vertex)),
        "weak_collinear_bridge_compressed_vertices": [
            int(vertex_idx) + 1 for vertex_idx, compressed in enumerate(compress_vertex) if compressed
        ],
    }


def _prune_spur_edges(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    max_spur_length_px: float,
    min_keep_score: float,
    max_iterations: int = 8,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    current_vertices, current_edges, current_weights = _reindex_wireframe_graph(vertices_px, edges, edge_weights)
    if not current_edges:
        return current_vertices, current_edges, current_weights

    for _ in range(max_iterations):
        adjacency = _wireframe_adjacency(len(current_vertices), current_edges)
        degrees = {idx: len(neighbors) for idx, neighbors in adjacency.items()}

        kept_edges: list[tuple[int, int]] = []
        kept_weights: list[float] = []
        removed_count = 0
        for edge, weight in zip(current_edges, current_weights):
            left, right = edge
            length = float(np.linalg.norm(current_vertices[left] - current_vertices[right]))
            is_spur = (degrees[left] == 1 or degrees[right] == 1) and (
                length <= max_spur_length_px or float(weight) < min_keep_score
            )
            if is_spur:
                removed_count += 1
                continue
            kept_edges.append(edge)
            kept_weights.append(float(weight))

        if removed_count == 0:
            break
        current_vertices, current_edges, current_weights = _reindex_wireframe_graph(
            current_vertices,
            kept_edges,
            kept_weights,
        )
        if not current_edges:
            break

    return current_vertices, current_edges, current_weights


def _prune_weak_quad_diagonal_edges(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    edge_threshold: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], dict]:
    if len(vertices_px) < 4 or len(edges) < 5:
        return vertices_px, edges, edge_weights, {
            "weak_quad_diagonal_pruned_edge_count": 0,
            "weak_quad_diagonal_pruned_edges": [],
        }

    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    edge_weight_map = _wireframe_edge_weight_map(edges, edge_weights)
    low_edge_threshold = max(0.42, float(edge_threshold) * 8.0)
    remove_edge_indices: set[int] = set()
    removed_edges: list[list[int]] = []

    for edge_idx, (left, right) in enumerate(edges):
        edge = tuple(sorted((int(left), int(right))))
        weight = float(edge_weights[edge_idx])
        if weight > low_edge_threshold:
            continue
        direct_length = float(np.linalg.norm(vertices_px[left] - vertices_px[right]))
        if direct_length < 50.0 and weight > 0.10:
            continue

        shared_neighbors = sorted(set(adjacency.get(left, ())).intersection(adjacency.get(right, ())))
        if len(shared_neighbors) < 2:
            continue

        for first_idx, first in enumerate(shared_neighbors):
            for second in shared_neighbors[first_idx + 1 :]:
                path_edges = [
                    tuple(sorted((left, first))),
                    tuple(sorted((right, first))),
                    tuple(sorted((left, second))),
                    tuple(sorted((right, second))),
                ]
                path_weights = [float(edge_weight_map.get(path_edge, 0.0)) for path_edge in path_edges]
                if min(path_weights) <= weight + 0.025:
                    continue
                if float(np.mean(path_weights)) <= weight + 0.20:
                    continue

                polygon = np.asarray(
                    [
                        vertices_px[first],
                        vertices_px[left],
                        vertices_px[second],
                        vertices_px[right],
                    ],
                    dtype=float,
                )
                if abs(polygon_area(order_polygon(polygon))) < 20.0:
                    continue

                remove_edge_indices.add(edge_idx)
                removed_edges.append([int(left) + 1, int(right) + 1])
                break
            if edge_idx in remove_edge_indices:
                break

    if not remove_edge_indices:
        return vertices_px, edges, edge_weights, {
            "weak_quad_diagonal_pruned_edge_count": 0,
            "weak_quad_diagonal_pruned_edges": [],
        }

    kept_edges: list[tuple[int, int]] = []
    kept_weights: list[float] = []
    for edge_idx, (edge, weight) in enumerate(zip(edges, edge_weights)):
        if edge_idx in remove_edge_indices:
            continue
        kept_edges.append(edge)
        kept_weights.append(float(weight))

    pruned_vertices_px, pruned_edges, pruned_weights = _reindex_wireframe_graph(
        vertices_px,
        kept_edges,
        kept_weights,
    )
    return pruned_vertices_px, pruned_edges, pruned_weights, {
        "weak_quad_diagonal_pruned_edge_count": int(len(remove_edge_indices)),
        "weak_quad_diagonal_pruned_edges": removed_edges,
    }


def _angle_between_vectors_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / segment_len_sq, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _edge_weight_between_raw_vertices(
    raw_edges: list[tuple[int, int]],
    raw_weights: list[float],
    left_idx: int,
    right_idx: int,
) -> float | None:
    wanted = tuple(sorted((int(left_idx), int(right_idx))))
    best: float | None = None
    for edge, weight in zip(raw_edges, raw_weights):
        if tuple(sorted((int(edge[0]), int(edge[1])))) != wanted:
            continue
        best = max(float(weight), best if best is not None else 0.0)
    return best


def _recover_low_confidence_gap_corner(
    vertices_px: np.ndarray,
    endpoint: int,
    close_vertex: int,
    *,
    raw_vertices_px: np.ndarray | None,
    raw_edges: list[tuple[int, int]] | None,
    raw_weights: list[float] | None,
) -> tuple[np.ndarray, float, float] | None:
    if raw_vertices_px is None or raw_edges is None or raw_weights is None:
        return None
    if len(raw_vertices_px) == 0:
        return None

    endpoint_pt = vertices_px[endpoint]
    close_pt = vertices_px[close_vertex]
    processed_dists = np.linalg.norm(raw_vertices_px[:, None, :] - vertices_px[None, :, :], axis=2)
    raw_to_endpoint = np.where(processed_dists[:, endpoint] <= 6.0)[0]
    raw_to_close = np.where(processed_dists[:, close_vertex] <= 6.0)[0]
    if len(raw_to_endpoint) == 0 or len(raw_to_close) == 0:
        return None

    best: tuple[float, np.ndarray, float, float] | None = None
    for raw_idx, raw_point in enumerate(raw_vertices_px):
        if float(np.min(processed_dists[raw_idx])) <= 7.0:
            continue

        endpoint_dist = float(np.linalg.norm(raw_point - endpoint_pt))
        close_dist = float(np.linalg.norm(raw_point - close_pt))
        if not (8.0 <= endpoint_dist <= 28.0):
            continue
        if not (35.0 <= close_dist <= 95.0):
            continue

        segment_distance = _point_to_segment_distance(raw_point, endpoint_pt, close_pt)
        if not (5.0 <= segment_distance <= 28.0):
            continue

        endpoint_weight = max(
            (
                _edge_weight_between_raw_vertices(raw_edges, raw_weights, raw_idx, int(endpoint_raw_idx)) or 0.0
                for endpoint_raw_idx in raw_to_endpoint
            ),
            default=0.0,
        )
        close_weight = max(
            (
                _edge_weight_between_raw_vertices(raw_edges, raw_weights, raw_idx, int(close_raw_idx)) or 0.0
                for close_raw_idx in raw_to_close
            ),
            default=0.0,
        )
        if min(endpoint_weight, close_weight) < 0.025:
            continue
        if max(endpoint_weight, close_weight) < 0.045:
            continue

        score = -(endpoint_weight + close_weight) + 0.01 * abs(endpoint_dist - 16.0) + 0.002 * close_dist
        if best is None or score < best[0]:
            best = (score, np.asarray(raw_point, dtype=float), float(endpoint_weight), float(close_weight))

    if best is None:
        return None
    return best[1], best[2], best[3]


def _add_marker_gap_closing_edges(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    *,
    raw_vertices_px: np.ndarray | None = None,
    raw_edges: list[tuple[int, int]] | None = None,
    raw_weights: list[float] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], dict]:
    if len(vertices_px) < 3 or len(edges) < 2:
        return vertices_px, edges, edge_weights, {
            "marker_gap_added_edge_count": 0,
            "marker_gap_added_edges": [],
            "marker_gap_inserted_corner_count": 0,
            "marker_gap_inserted_corners": [],
        }

    adjacency = _wireframe_adjacency(len(vertices_px), edges)
    edge_weight_map = _wireframe_edge_weight_map(edges, edge_weights)
    existing_edges = {tuple(sorted(edge)) for edge in edges}
    added_edges: dict[tuple[int, int], float] = {}
    inserted_corners: list[dict] = []
    working_vertices = np.asarray(vertices_px, dtype=float)

    for endpoint, endpoint_neighbors in adjacency.items():
        if len(endpoint_neighbors) != 1:
            continue
        pivot = next(iter(endpoint_neighbors))
        pivot_edge = tuple(sorted((endpoint, pivot)))
        pivot_weight = float(edge_weight_map.get(pivot_edge, 0.0))
        pivot_length = float(np.linalg.norm(vertices_px[endpoint] - vertices_px[pivot]))
        if not (35.0 <= pivot_length <= 95.0):
            continue

        for close_vertex in adjacency.get(pivot, ()):
            if close_vertex == endpoint:
                continue
            direct_edge = tuple(sorted((endpoint, close_vertex)))
            if direct_edge in existing_edges or direct_edge in added_edges:
                continue

            short_edge = tuple(sorted((pivot, close_vertex)))
            short_weight = float(edge_weight_map.get(short_edge, 0.0))
            short_length = float(np.linalg.norm(vertices_px[pivot] - vertices_px[close_vertex]))
            direct_length = float(np.linalg.norm(vertices_px[endpoint] - vertices_px[close_vertex]))
            if short_length > 22.0:
                continue
            if not (35.0 <= direct_length <= 100.0):
                continue
            if abs(direct_length - pivot_length) > max(12.0, pivot_length * 0.22):
                continue

            angle_deg = _angle_between_vectors_deg(
                vertices_px[pivot] - vertices_px[endpoint],
                vertices_px[close_vertex] - vertices_px[endpoint],
            )
            if angle_deg > 18.0:
                continue

            corner = _recover_low_confidence_gap_corner(
                working_vertices,
                endpoint,
                close_vertex,
                raw_vertices_px=raw_vertices_px,
                raw_edges=raw_edges,
                raw_weights=raw_weights,
            )
            if corner is not None:
                corner_point, endpoint_weight, close_weight = corner
                new_idx = len(working_vertices)
                working_vertices = np.vstack([working_vertices, corner_point])
                added_edges[tuple(sorted((endpoint, new_idx)))] = max(0.05, float(endpoint_weight))
                added_edges[tuple(sorted((new_idx, close_vertex)))] = max(0.05, float(close_weight))
                inserted_corners.append(
                    {
                        "corner_vertex": int(new_idx) + 1,
                        "endpoint": int(endpoint) + 1,
                        "close_vertex": int(close_vertex) + 1,
                        "corner_xy": [float(corner_point[0]), float(corner_point[1])],
                    }
                )
            else:
                added_edges[direct_edge] = max(0.05, min(pivot_weight, short_weight))

    if not added_edges:
        return working_vertices, edges, edge_weights, {
            "marker_gap_added_edge_count": 0,
            "marker_gap_added_edges": [],
            "marker_gap_inserted_corner_count": 0,
            "marker_gap_inserted_corners": [],
        }

    new_edges = list(edges)
    new_weights = [float(weight) for weight in edge_weights]
    for edge, weight in sorted(added_edges.items()):
        new_edges.append(edge)
        new_weights.append(float(weight))

    return working_vertices, new_edges, new_weights, {
        "marker_gap_added_edge_count": int(len(added_edges)),
        "marker_gap_added_edges": [[int(left) + 1, int(right) + 1] for left, right in sorted(added_edges)],
        "marker_gap_inserted_corner_count": int(len(inserted_corners)),
        "marker_gap_inserted_corners": inserted_corners,
    }


def _prepare_hawp_graph(
    wireframe: dict,
    edge_threshold: float,
    image_shape: tuple[int, int] | None = None,
    preserve_details: bool = False,
    marker_points: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], dict]:
    all_raw_vertices_px = np.asarray(wireframe.get("vertices", []), dtype=float)
    all_raw_edges = [tuple(map(int, edge)) for edge in wireframe.get("edges", [])]
    all_raw_weights = [float(weight) for weight in wireframe.get("edges-weights", [])]
    raw_vertices_px, raw_edges, raw_weights = _wireframe_subgraph(wireframe, edge_threshold)
    raw_stats = {
        "raw_vertex_count": int(len(raw_vertices_px)),
        "raw_edge_count": int(len(raw_edges)),
    }
    if len(raw_vertices_px) < 2 or not raw_edges:
        return raw_vertices_px, raw_edges, raw_weights, raw_stats

    if image_shape is not None:
        image_diag = math.hypot(float(image_shape[1]), float(image_shape[0]))
    else:
        span = np.ptp(raw_vertices_px, axis=0)
        image_diag = float(np.linalg.norm(span))

    if preserve_details:
        min_length_px = max(2.0, image_diag * 0.00125)
        snap_distance_px = min(max(3.0, image_diag * 0.0018), 9.0)
        support_snap_distance_px = max(2.5, snap_distance_px * 0.9)
        split_tolerance_px = max(2.0, snap_distance_px * 0.65)
        intersection_merge_px = max(1.5, snap_distance_px * 0.45)
        duplicate_bridge_px = min(max(5.0, snap_distance_px * 1.45), 8.0)
        marker_merge_px = min(max(5.5, snap_distance_px * 1.5), 9.5)
        max_spur_length_px = max(7.0, image_diag * 0.00325)
        min_keep_score = max(0.10, edge_threshold * 2.5)
        angle_tolerance_deg = 4.0
    else:
        min_length_px = max(4.0, image_diag * 0.0025)
        snap_distance_px = min(max(4.0, image_diag * 0.00225), 8.0)
        support_snap_distance_px = 0.0
        split_tolerance_px = 0.0
        intersection_merge_px = 0.0
        duplicate_bridge_px = 0.0
        marker_merge_px = 0.0
        max_spur_length_px = max(10.0, image_diag * 0.0055)
        min_keep_score = max(0.15, edge_threshold * 4.0)
        angle_tolerance_deg = 10.0

    processed_vertices_px, processed_edges, processed_weights = _filter_wireframe_edges_by_length(
        raw_vertices_px,
        raw_edges,
        raw_weights,
        min_length_px=min_length_px,
    )
    processed_vertices_px, processed_edges, processed_weights = _snap_wireframe_vertices(
        processed_vertices_px,
        processed_edges,
        processed_weights,
        snap_distance_px=snap_distance_px,
    )
    if preserve_details:
        processed_vertices_px = _snap_vertices_to_supporting_edges(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            max_distance_px=support_snap_distance_px,
        )
        processed_vertices_px, processed_edges, processed_weights = _split_edges_at_existing_vertices(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            point_tolerance_px=split_tolerance_px,
        )
        processed_vertices_px, processed_edges, processed_weights = _split_edges_at_intersections(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            merge_distance_px=intersection_merge_px,
        )
        processed_vertices_px, processed_edges, processed_weights, mixed_split_prune_stats = _prune_mixed_confidence_split_vertices(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            edge_threshold=edge_threshold,
            max_spur_length_px=max_spur_length_px,
            min_keep_score=min_keep_score,
            angle_tolerance_deg=angle_tolerance_deg,
        )
        processed_vertices_px, processed_edges, processed_weights, boundary_ear_prune_stats = _prune_boundary_ear_vertices(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            edge_threshold=edge_threshold,
            marker_points=marker_points,
        )
    else:
        mixed_split_prune_stats = {
            "mixed_confidence_pruned_vertex_count": 0,
            "mixed_confidence_pruned_vertices": [],
        }
        boundary_ear_prune_stats = {
            "boundary_ear_pruned_vertex_count": 0,
            "boundary_ear_pruned_vertices": [],
        }
    processed_vertices_px, processed_edges, processed_weights = _compress_collinear_wireframe(
        processed_vertices_px,
        processed_edges,
        processed_weights,
        angle_tolerance_deg=angle_tolerance_deg,
    )
    processed_vertices_px, processed_edges, processed_weights = _prune_spur_edges(
        processed_vertices_px,
        processed_edges,
        processed_weights,
        max_spur_length_px=max_spur_length_px,
        min_keep_score=min_keep_score,
    )
    if duplicate_bridge_px > 0:
        processed_vertices_px, processed_edges, processed_weights = _collapse_duplicate_bridge_edges(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            max_bridge_length_px=duplicate_bridge_px,
        )
    if marker_points and marker_merge_px > 0:
        processed_vertices_px, processed_edges, processed_weights = _collapse_vertices_to_markers(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            marker_points,
            merge_margin_px=marker_merge_px,
            max_chain_edge_px=max(4.5, duplicate_bridge_px),
        )
    processed_vertices_px, processed_edges, processed_weights = _snap_wireframe_vertices(
        processed_vertices_px,
        processed_edges,
        processed_weights,
        snap_distance_px=max(0.0, snap_distance_px * 0.75),
    )
    if preserve_details:
        processed_vertices_px, processed_edges, processed_weights = _split_edges_at_existing_vertices(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            point_tolerance_px=max(1.5, split_tolerance_px * 0.75),
        )
        processed_vertices_px, processed_edges, processed_weights = _collapse_duplicate_bridge_edges(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            max_bridge_length_px=max(4.5, duplicate_bridge_px * 0.9),
        )
        processed_vertices_px, processed_edges, processed_weights, weak_diagonal_prune_stats = _prune_weak_quad_diagonal_edges(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            edge_threshold=edge_threshold,
        )
        processed_vertices_px, processed_edges, processed_weights, marker_gap_add_stats = _add_marker_gap_closing_edges(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            raw_vertices_px=all_raw_vertices_px,
            raw_edges=all_raw_edges,
            raw_weights=all_raw_weights,
        )
        processed_vertices_px, processed_edges, processed_weights, weak_collinear_bridge_stats = _compress_weak_collinear_bridge_chains(
            processed_vertices_px,
            processed_edges,
            processed_weights,
        )
        processed_vertices_px, processed_edges, processed_weights, boundary_ear_prune_final_stats = _prune_boundary_ear_vertices(
            processed_vertices_px,
            processed_edges,
            processed_weights,
            edge_threshold=edge_threshold,
            marker_points=marker_points,
            require_global_min=True,
        )
        boundary_ear_prune_stats = {
            "boundary_ear_pruned_vertex_count": int(
                boundary_ear_prune_stats.get("boundary_ear_pruned_vertex_count", 0)
                + boundary_ear_prune_final_stats.get("boundary_ear_pruned_vertex_count", 0)
            ),
            "boundary_ear_pruned_vertices": [
                *boundary_ear_prune_stats.get("boundary_ear_pruned_vertices", []),
                *boundary_ear_prune_final_stats.get("boundary_ear_pruned_vertices", []),
            ],
        }
    else:
        weak_diagonal_prune_stats = {
            "weak_quad_diagonal_pruned_edge_count": 0,
            "weak_quad_diagonal_pruned_edges": [],
        }
        marker_gap_add_stats = {
            "marker_gap_added_edge_count": 0,
            "marker_gap_added_edges": [],
            "marker_gap_inserted_corner_count": 0,
            "marker_gap_inserted_corners": [],
        }
        weak_collinear_bridge_stats = {
            "weak_collinear_bridge_compressed_vertex_count": 0,
            "weak_collinear_bridge_compressed_vertices": [],
        }

    if len(processed_vertices_px) < 3 or len(processed_edges) < 3:
        return raw_vertices_px, raw_edges, raw_weights, raw_stats

    stats = {
        **raw_stats,
        "processed_vertex_count": int(len(processed_vertices_px)),
        "processed_edge_count": int(len(processed_edges)),
        "snap_distance_px": float(snap_distance_px),
        "min_edge_length_px": float(min_length_px),
        "duplicate_bridge_px": float(duplicate_bridge_px),
        "marker_merge_px": float(marker_merge_px),
        "max_spur_length_px": float(max_spur_length_px),
        "preserve_details": bool(preserve_details),
        **mixed_split_prune_stats,
        **boundary_ear_prune_stats,
        **weak_diagonal_prune_stats,
        **marker_gap_add_stats,
        **weak_collinear_bridge_stats,
    }
    return processed_vertices_px, processed_edges, processed_weights, stats


def simplify_wireframe_evidence(
    wireframe: dict,
    *,
    edge_threshold: float,
    image_shape: tuple[int, int] | None = None,
) -> tuple[dict, dict]:
    vertices_px, edges, edge_weights, stats = _prepare_hawp_graph(
        wireframe,
        edge_threshold=edge_threshold,
        image_shape=image_shape,
    )
    simplified = {
        "vertices": vertices_px.tolist(),
        "vertices-score": [1.0 for _ in range(len(vertices_px))],
        "edges": [[int(left), int(right)] for left, right in edges],
        "edges-weights": [float(weight) for weight in edge_weights],
        "height": int(image_shape[0]) if image_shape is not None else wireframe.get("height"),
        "width": int(image_shape[1]) if image_shape is not None else wireframe.get("width"),
    }
    return simplified, stats


def _mixed_confidence_split_vertex_stats(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    edge_threshold: float,
) -> dict:
    if len(vertices_px) == 0 or len(edges) == 0:
        return {
            "suspicious_mixed_split_vertex_count": 0,
            "suspicious_mixed_split_vertices": [],
            "suspicious_mixed_split_low_edge_max": None,
            "suspicious_mixed_split_high_edge_min": None,
        }

    adjacency_weights: dict[int, list[float]] = {idx: [] for idx in range(len(vertices_px))}
    for (left, right), weight in zip(edges, edge_weights):
        adjacency_weights[left].append(float(weight))
        adjacency_weights[right].append(float(weight))

    all_weights = np.asarray(edge_weights, dtype=float)
    low_threshold = max(float(edge_threshold) * 2.0, 0.20)
    high_threshold = max(0.80, float(np.percentile(all_weights, 75)))
    suspicious_vertices: list[int] = []
    low_edge_max_values: list[float] = []
    high_edge_min_values: list[float] = []

    for vertex_idx, incident_weights in adjacency_weights.items():
        if len(incident_weights) != 4:
            continue
        ordered = sorted(float(weight) for weight in incident_weights)
        low_weights = ordered[:2]
        high_weights = ordered[2:]
        low_max = max(low_weights)
        high_min = min(high_weights)
        if low_max > low_threshold:
            continue
        if high_min < high_threshold:
            continue
        if (np.mean(high_weights) - np.mean(low_weights)) < 0.45:
            continue
        suspicious_vertices.append(int(vertex_idx) + 1)
        low_edge_max_values.append(float(low_max))
        high_edge_min_values.append(float(high_min))

    return {
        "suspicious_mixed_split_vertex_count": int(len(suspicious_vertices)),
        "suspicious_mixed_split_vertices": suspicious_vertices,
        "suspicious_mixed_split_low_edge_max": float(max(low_edge_max_values)) if low_edge_max_values else None,
        "suspicious_mixed_split_high_edge_min": float(min(high_edge_min_values)) if high_edge_min_values else None,
    }


def _trace_planar_faces(vertices_px: np.ndarray, edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(vertices_px))}
    for left, right in edges:
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    neighbor_order: dict[int, list[int]] = {}
    for vertex_idx, neighbors in adjacency.items():
        if not neighbors:
            continue
        neighbor_order[vertex_idx] = sorted(
            neighbors,
            key=lambda neighbor_idx: math.atan2(
                vertices_px[neighbor_idx, 1] - vertices_px[vertex_idx, 1],
                vertices_px[neighbor_idx, 0] - vertices_px[vertex_idx, 0],
            ),
        )

    directed_edges = {(left, right) for left, right in edges if left != right} | {
        (right, left) for left, right in edges if left != right
    }
    visited: set[tuple[int, int]] = set()
    unique_faces: dict[tuple[int, ...], float] = {}

    for start in sorted(directed_edges):
        if start in visited:
            continue
        current = start
        face: list[int] = []

        while current not in visited:
            visited.add(current)
            left, right = current
            face.append(left)
            ordered_neighbors = neighbor_order.get(right)
            if not ordered_neighbors:
                break
            try:
                prev_idx = ordered_neighbors.index(left)
            except ValueError:
                break
            current = (right, ordered_neighbors[prev_idx - 1])
            if current == start:
                break

        canonical = _canonical_cycle(face)
        if canonical is None:
            continue
        polygon = vertices_px[list(canonical)]
        unique_faces[canonical] = polygon_area(polygon)

    return [
        list(cycle)
        for cycle, area in sorted(unique_faces.items(), key=lambda item: abs(item[1]))
        if area > 0
    ]


def _polygonize_face_cycles(vertices_px: np.ndarray, edges: list[tuple[int, int]]) -> list[list[int]]:
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize
    except Exception:
        return []

    vertices_array = np.asarray(vertices_px, dtype=float)
    if len(vertices_array) < 3 or len(edges) < 3:
        return []

    rounded_vertices = [tuple(round(float(value), 6) for value in vertex) for vertex in vertices_array.tolist()]
    coord_to_idx = {coord: idx for idx, coord in enumerate(rounded_vertices)}
    polygons = list(
        polygonize(
            [
                LineString([tuple(vertices_array[left].tolist()), tuple(vertices_array[right].tolist())])
                for left, right in edges
                if left != right
            ]
        )
    )
    if not polygons:
        return []

    cycles_by_key: dict[tuple[int, ...], list[int]] = {}
    for polygon in polygons:
        coords = list(polygon.exterior.coords)[:-1]
        if len(coords) < 3:
            continue

        cycle: list[int] = []
        valid = True
        for x_coord, y_coord in coords:
            key = (round(float(x_coord), 6), round(float(y_coord), 6))
            vertex_idx = coord_to_idx.get(key)
            if vertex_idx is None:
                distances = np.linalg.norm(vertices_array - np.array([x_coord, y_coord], dtype=float), axis=1)
                vertex_idx = int(np.argmin(distances))
                if float(distances[vertex_idx]) > 1e-3:
                    valid = False
                    break
            cycle.append(int(vertex_idx))

        if not valid:
            continue
        canonical = _canonical_cycle(dedupe_sequence(cycle))
        if canonical is None or len(canonical) not in {3, 4}:
            continue
        cycles_by_key[canonical] = list(canonical)

    return [cycles_by_key[key] for key in sorted(cycles_by_key)]


def _cycle_edges(cycle: list[int] | tuple[int, ...]) -> list[tuple[int, int]]:
    return [
        tuple(sorted((int(cycle[idx]), int(cycle[(idx + 1) % len(cycle)]))))
        for idx in range(len(cycle))
    ]


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray, tol: float = 1e-4) -> bool:
    segment = end - start
    rel = point - start
    cross = abs(float(segment[0] * rel[1] - segment[1] * rel[0]))
    if cross > tol * max(1.0, float(np.linalg.norm(segment))):
        return False
    dot = float(np.dot(rel, segment))
    if dot < -tol:
        return False
    denom = float(np.dot(segment, segment))
    if dot - denom > tol:
        return False
    return True


def _segment_pairs_conflict(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    shared_endpoint = any(
        np.linalg.norm(point_a - point_b) <= tol
        for point_a in (first_start, first_end)
        for point_b in (second_start, second_end)
    )
    if shared_endpoint:
        return False

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)

    return (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (o3 > tol and o4 < -tol or o3 < -tol and o4 > tol)


def _point_in_polygon_strict(point: np.ndarray, polygon: np.ndarray, tol: float = 1e-4) -> bool:
    inside = False
    n_points = len(polygon)
    for idx in range(n_points):
        start = polygon[idx]
        end = polygon[(idx + 1) % n_points]
        if _point_on_segment(point, start, end, tol=tol):
            return False
        if ((start[1] > point[1]) != (end[1] > point[1])) and (
            point[0] < (end[0] - start[0]) * (point[1] - start[1]) / max(end[1] - start[1], 1e-9) + start[0]
        ):
            inside = not inside
    return inside


def _cycles_conflict(
    vertices_px: np.ndarray,
    cycle_a: list[int] | tuple[int, ...],
    cycle_b: list[int] | tuple[int, ...],
) -> bool:
    polygon_a = order_polygon(vertices_px[list(cycle_a)])
    polygon_b = order_polygon(vertices_px[list(cycle_b)])

    min_ax, min_ay = np.min(polygon_a, axis=0)
    max_ax, max_ay = np.max(polygon_a, axis=0)
    min_bx, min_by = np.min(polygon_b, axis=0)
    max_bx, max_by = np.max(polygon_b, axis=0)
    if max_ax < min_bx or max_bx < min_ax or max_ay < min_by or max_by < min_ay:
        return False

    for idx_a in range(len(polygon_a)):
        start_a = polygon_a[idx_a]
        end_a = polygon_a[(idx_a + 1) % len(polygon_a)]
        for idx_b in range(len(polygon_b)):
            start_b = polygon_b[idx_b]
            end_b = polygon_b[(idx_b + 1) % len(polygon_b)]
            if _segment_pairs_conflict(start_a, end_a, start_b, end_b):
                return True

    centroid_a = np.mean(polygon_a, axis=0)
    centroid_b = np.mean(polygon_b, axis=0)
    if _point_in_polygon_strict(centroid_a, polygon_b):
        return True
    if _point_in_polygon_strict(centroid_b, polygon_a):
        return True

    return False


def _enumerate_small_cycles(
    n_vertices: int,
    edges: list[tuple[int, int]],
    max_length: int = 4,
) -> list[list[int]]:
    adjacency = _wireframe_adjacency(n_vertices, edges)
    cycles: set[tuple[int, ...]] = set()

    for start_idx in range(n_vertices):
        stack: list[tuple[int, list[int]]] = [(start_idx, [start_idx])]
        while stack:
            current_idx, path = stack.pop()
            if len(path) > max_length:
                continue
            for neighbor_idx in adjacency.get(current_idx, ()):
                if neighbor_idx == start_idx:
                    if 3 <= len(path) <= max_length:
                        canonical = _canonical_cycle(path)
                        if canonical is not None and len(canonical) <= max_length:
                            cycles.add(canonical)
                    continue
                if neighbor_idx in path or len(path) >= max_length:
                    continue
                stack.append((neighbor_idx, path + [neighbor_idx]))

    return [list(cycle) for cycle in sorted(cycles, key=lambda item: (len(item), item))]


def _augment_face_cycles(
    vertices_px: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    seed_faces: list[list[int]],
) -> list[list[int]]:
    edge_weight_map = _wireframe_edge_weight_map(edges, edge_weights)
    candidates: dict[tuple[int, ...], dict] = {}

    def register_candidate(cycle: list[int] | tuple[int, ...], seeded: bool = False) -> None:
        canonical = _canonical_cycle(list(cycle))
        if canonical is None or len(canonical) not in {3, 4}:
            return
        polygon = vertices_px[list(canonical)]
        area = abs(polygon_area(polygon))
        if area <= 20.0:
            return
        cycle_edges = _cycle_edges(canonical)
        mean_weight = float(np.mean([edge_weight_map.get(edge, 0.0) for edge in cycle_edges])) if cycle_edges else 0.0
        existing = candidates.get(canonical)
        payload = {
            "cycle": list(canonical),
            "edges": cycle_edges,
            "area": area,
            "mean_weight": mean_weight,
            "seeded": seeded,
        }
        if existing is None:
            candidates[canonical] = payload
        else:
            existing["seeded"] = existing["seeded"] or seeded
            existing["area"] = max(existing["area"], area)
            existing["mean_weight"] = max(existing["mean_weight"], mean_weight)

    for cycle in seed_faces:
        register_candidate(cycle, seeded=True)
    for cycle in _enumerate_small_cycles(len(vertices_px), edges, max_length=4):
        register_candidate(cycle, seeded=False)

    selected: list[list[int]] = []
    selected_keys: set[tuple[int, ...]] = set()
    selected_edge_use: Counter[tuple[int, int]] = Counter()
    for candidate in sorted(
        (payload for payload in candidates.values() if payload["seeded"]),
        key=lambda item: (-item["area"], -item["mean_weight"], item["cycle"]),
    ):
        cycle_key = tuple(candidate["cycle"])
        if cycle_key in selected_keys:
            continue
        selected.append(candidate["cycle"])
        selected_keys.add(cycle_key)
        selected_edge_use.update(candidate["edges"])

    while True:
        used_vertices = {vertex_idx for cycle in selected for vertex_idx in cycle}
        used_edges = {edge for cycle in selected for edge in _cycle_edges(cycle)}
        best_candidate = None
        best_key = None

        for candidate in candidates.values():
            cycle_key = tuple(candidate["cycle"])
            if cycle_key in selected_keys:
                continue
            if any(selected_edge_use[edge] >= 2 for edge in candidate["edges"]):
                continue
            new_vertices = sum(vertex_idx not in used_vertices for vertex_idx in candidate["cycle"])
            new_edges = sum(edge not in used_edges for edge in candidate["edges"])
            if new_vertices == 0 and new_edges == 0:
                continue
            key = (
                new_vertices,
                new_edges,
                candidate["seeded"],
                candidate["area"],
                candidate["mean_weight"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate["cycle"])
        selected_keys.add(tuple(best_candidate["cycle"]))
        selected_edge_use.update(best_candidate["edges"])

    return selected


def _coarsen_points_to_target(
    points: np.ndarray,
    target_count: int,
    merge_thresholds: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 32.0),
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= target_count:
        return points

    best = points
    best_key = (abs(len(points) - target_count), 0 if len(points) >= target_count else 1, 0.0)
    for merge_threshold in merge_thresholds:
        if merge_threshold <= 0:
            centers = points
        else:
            centers, _ = cluster_points(points, merge_threshold)
        count = len(centers)
        score = (abs(count - target_count), 0 if count >= target_count else 1, merge_threshold)
        if score < best_key:
            best = centers
            best_key = score
    return best


def build_hawp_face_candidate(
    candidate_name: str,
    wireframe: dict,
    edge_threshold: float,
    calibration: AxisCalibration,
    prop_id: int,
    image_shape: tuple[int, int] | None = None,
    mesh_image_rgb: np.ndarray | None = None,
) -> ReconstructionCandidate | None:
    marker_points = _white_marker_points(mesh_image_rgb) if mesh_image_rgb is not None else None
    vertices_px, edges, edge_weights, hawp_graph_stats = _prepare_hawp_graph(
        wireframe,
        edge_threshold,
        image_shape=image_shape,
        preserve_details=True,
        marker_points=marker_points,
    )
    if len(vertices_px) < 3 or len(edges) < 3:
        return None

    element_cycles = _polygonize_face_cycles(vertices_px, edges)
    if not element_cycles:
        faces = _trace_planar_faces(vertices_px, edges)
        element_cycles = _augment_face_cycles(
            vertices_px,
            edges,
            edge_weights,
            [cycle for cycle in faces if len(cycle) in {3, 4}],
        )
    if not element_cycles:
        return None

    unique_wireframe_edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in edges
        if int(left) != int(right)
    }
    used_wireframe_edges = {
        edge
        for cycle in element_cycles
        for edge in _cycle_edges(cycle)
    }
    edge_coverage = float(len(used_wireframe_edges) / max(1, len(unique_wireframe_edges)))
    unused_edge_count = int(max(0, len(unique_wireframe_edges) - len(used_wireframe_edges)))
    element_area_px_sum = float(
        sum(
            abs(polygon_area(order_polygon(vertices_px[list(cycle)])))
            for cycle in element_cycles
        )
    )
    leaf_stats = _wireframe_leaf_stats(vertices_px, edges)
    mixed_split_stats = _mixed_confidence_split_vertex_stats(
        vertices_px,
        edges,
        edge_weights,
        edge_threshold=edge_threshold,
    )

    world_xy = calibration.pixel_to_world(vertices_px)
    nodes = [
        Node(
            node_id=node_idx,
            x=float(x_world),
            y=float(y_world),
            pixel_x=float(x_px),
            pixel_y=float(y_px),
        )
        for node_idx, ((x_world, y_world), (x_px, y_px)) in enumerate(zip(world_xy, vertices_px.tolist()), start=1)
    ]
    elements = [
        Element(
            card="CTRIA3" if len(cycle) == 3 else "CQUAD4",
            element_id=element_idx,
            prop_id=prop_id,
            node_ids=[node_idx + 1 for node_idx in cycle],
        )
        for element_idx, cycle in enumerate(element_cycles, start=1)
    ]
    candidate_wireframe = {
        "vertices": vertices_px.tolist(),
        "vertices-score": [1.0 for _ in range(len(vertices_px))],
        "edges": [[left, right] for left, right in edges],
        "edges-weights": [float(weight) for weight in edge_weights],
        "height": int(image_shape[0]) if image_shape is not None else int(round(calibration.origin_y + 80)),
        "width": int(image_shape[1]) if image_shape is not None else int(round(calibration.x_axis_end + 80)),
    }
    return ReconstructionCandidate(
        name=candidate_name,
        merge_threshold=0.0,
        ratio_to_triangle=0.0,
        ratio_to_quad=0.0,
        mesh=NasMesh(
            nodes=nodes,
            elements=elements,
            metadata={
                "source": "hawp_faces",
                "edge_threshold": edge_threshold,
                "wireframe_edge_count": int(len(unique_wireframe_edges)),
                "used_wireframe_edge_count": int(len(used_wireframe_edges)),
                "unused_wireframe_edge_count": unused_edge_count,
                "wireframe_edge_coverage": edge_coverage,
                "element_area_px_sum": element_area_px_sum,
                **leaf_stats,
                **mixed_split_stats,
                **hawp_graph_stats,
            },
        ),
        wireframe=candidate_wireframe,
        source="hawp_faces",
        proposal_family="hawp_faces",
    )


def build_hawp_variant_fused_candidate(
    base_candidate: ReconstructionCandidate,
    guide_candidate: ReconstructionCandidate,
    calibration: AxisCalibration,
    candidate_name: str,
    *,
    max_snap_distance_px: float = 14.0,
    min_max_displacement_px: float = 6.0,
    min_snap_count: int = 2,
    mesh_image_rgb: np.ndarray | None = None,
) -> ReconstructionCandidate | None:
    base_vertices = np.asarray(base_candidate.wireframe.get("vertices", []), dtype=float)
    guide_vertices = np.asarray(guide_candidate.wireframe.get("vertices", []), dtype=float)
    if len(base_vertices) == 0 or len(guide_vertices) == 0:
        return None
    if len(base_candidate.mesh.nodes) != len(base_vertices):
        return None

    distances = distance_matrix(base_vertices, guide_vertices)
    row_ind, col_ind = linear_sum_assignment(distances)
    marker_points = _white_marker_points(mesh_image_rgb) if mesh_image_rgb is not None else []

    def marker_support(base_point: np.ndarray, guide_point: np.ndarray) -> dict[str, float | int | bool]:
        if not marker_points:
            return {
                "marker_supported": False,
                "marker_index": -1,
                "base_marker_distance_px": 1.0e9,
                "guide_marker_distance_px": 1.0e9,
                "marker_improvement_px": 0.0,
                "marker_radius_px": 0.0,
            }
        marker_centers = np.asarray([marker[0] for marker in marker_points], dtype=float)
        base_marker_distances = np.linalg.norm(marker_centers - base_point, axis=1)
        marker_idx = int(np.argmin(base_marker_distances))
        marker_center, marker_radius = marker_points[marker_idx]
        base_marker_distance = float(base_marker_distances[marker_idx])
        guide_marker_distance = float(np.linalg.norm(guide_point - marker_center))
        marker_improvement = base_marker_distance - guide_marker_distance
        guide_threshold = max(float(marker_radius) + 1.5, 3.0)
        return {
            "marker_supported": bool(
                base_marker_distance <= 8.0
                and guide_marker_distance <= guide_threshold
                and marker_improvement >= 1.5
            ),
            "marker_index": marker_idx,
            "base_marker_distance_px": base_marker_distance,
            "guide_marker_distance_px": guide_marker_distance,
            "marker_improvement_px": marker_improvement,
            "marker_radius_px": float(marker_radius),
        }

    candidate_snap_records: list[dict[str, float | int | bool | str]] = []
    candidate_displacements: list[float] = []
    for base_idx, guide_idx in zip(row_ind.tolist(), col_ind.tolist()):
        distance_px = float(distances[base_idx, guide_idx])
        if distance_px <= 1e-6 or distance_px > max_snap_distance_px:
            continue
        support = marker_support(base_vertices[base_idx], guide_vertices[guide_idx])
        candidate_displacements.append(distance_px)
        candidate_snap_records.append(
            {
                "base_vertex": int(base_idx) + 1,
                "guide_vertex": int(guide_idx) + 1,
                "distance_px": distance_px,
                **support,
            }
        )

    if len(candidate_snap_records) < min_snap_count:
        return None
    max_displacement = float(np.max(candidate_displacements))
    marker_preserve_records: list[dict[str, float | int | bool | str]] = []
    if max_displacement >= min_max_displacement_px:
        snap_records = []
        for record in candidate_snap_records:
            marker_worsening = float(record["guide_marker_distance_px"]) - float(record["base_marker_distance_px"])
            if float(record["base_marker_distance_px"]) <= 8.0 and marker_worsening >= 1.25:
                marker_preserve_records.append(
                    {
                        **record,
                        "snap_reason": "marker_preserve_base_nearer",
                        "marker_worsening_px": marker_worsening,
                    }
                )
                continue
            snap_records.append(
                {
                    **record,
                    "snap_reason": "large_variant_displacement",
                }
            )
    else:
        snap_records = [
            {
                **record,
                "snap_reason": "marker_supported_small_displacement",
            }
            for record in candidate_snap_records
            if bool(record.get("marker_supported")) and float(record["distance_px"]) >= 2.0
        ]
        if not snap_records:
            return None

    fused_vertices = base_vertices.copy()
    displacements: list[float] = []
    marker_snap_count = 0
    for record in snap_records:
        base_idx = int(record["base_vertex"]) - 1
        guide_idx = int(record["guide_vertex"]) - 1
        fused_vertices[base_idx] = guide_vertices[guide_idx]
        displacements.append(float(record["distance_px"]))
        if bool(record.get("marker_supported")):
            marker_snap_count += 1

    marker_refine_records: list[dict[str, float | int | str]] = []
    for record in marker_preserve_records:
        marker_idx = int(record.get("marker_index", -1))
        if marker_idx < 0 or marker_idx >= len(marker_points):
            continue
        base_idx = int(record["base_vertex"]) - 1
        marker_center = np.asarray(marker_points[marker_idx][0], dtype=float)
        original_point = fused_vertices[base_idx].copy()
        refine_delta = marker_center - original_point
        refine_distance = float(np.linalg.norm(refine_delta))
        if refine_distance <= 1e-6:
            continue
        refine_step = 0.95 * refine_delta
        refine_step_distance = float(np.linalg.norm(refine_step))
        if refine_step_distance > 5.0:
            refine_step = refine_step / refine_step_distance * 5.0
            refine_step_distance = 5.0
        fused_vertices[base_idx] = original_point + refine_step
        marker_refine_records.append(
            {
                "base_vertex": int(record["base_vertex"]),
                "guide_vertex": int(record["guide_vertex"]),
                "marker_index": marker_idx,
                "refine_reason": "marker_preserve_partial_centering",
                "refine_dx_px": float(refine_step[0]),
                "refine_dy_px": float(refine_step[1]),
                "refine_distance_px": refine_step_distance,
                "base_marker_distance_before_px": float(refine_distance),
                "base_marker_distance_after_px": float(np.linalg.norm(fused_vertices[base_idx] - marker_center)),
            }
        )

    if not displacements:
        return None

    world_xy = calibration.pixel_to_world(fused_vertices)
    nodes = [
        Node(
            node_id=node.node_id,
            x=float(world_xy[idx, 0]),
            y=float(world_xy[idx, 1]),
            z=float(node.z),
            pixel_x=float(fused_vertices[idx, 0]),
            pixel_y=float(fused_vertices[idx, 1]),
        )
        for idx, node in enumerate(base_candidate.mesh.nodes)
    ]
    elements = [
        Element(
            card=element.card,
            element_id=element.element_id,
            prop_id=element.prop_id,
            node_ids=list(element.node_ids),
        )
        for element in base_candidate.mesh.elements
    ]
    fused_wireframe = {
        **base_candidate.wireframe,
        "vertices": fused_vertices.tolist(),
    }
    metadata = {
        **base_candidate.mesh.metadata,
        "source": "hawp_faces_fusion",
        "variant_fusion_base": base_candidate.name,
        "variant_fusion_guide": guide_candidate.name,
        "variant_fusion_snap_count": int(len(snap_records)),
        "variant_fusion_marker_snap_count": int(marker_snap_count),
        "variant_fusion_marker_preserve_count": int(len(marker_preserve_records)),
        "variant_fusion_marker_refine_count": int(len(marker_refine_records)),
        "variant_fusion_mean_displacement_px": float(np.mean(displacements)),
        "variant_fusion_max_displacement_px": float(np.max(displacements)),
        "variant_fusion_snap_distance_px": float(max_snap_distance_px),
        "variant_fusion_min_max_displacement_px": float(min_max_displacement_px),
        "variant_fusion_snaps": snap_records[:50],
        "variant_fusion_marker_preserves": marker_preserve_records[:50],
        "variant_fusion_marker_refines": marker_refine_records[:50],
        "variant_fusion_priority": 1,
    }
    return ReconstructionCandidate(
        name=candidate_name,
        merge_threshold=base_candidate.merge_threshold,
        ratio_to_triangle=base_candidate.ratio_to_triangle,
        ratio_to_quad=base_candidate.ratio_to_quad,
        mesh=NasMesh(nodes=nodes, elements=elements, metadata=metadata),
        wireframe=fused_wireframe,
        source="hawp_faces_fusion",
        proposal_family="hawp_faces",
    )


def apply_selected_marker_center_snap(
    candidate: ReconstructionCandidate,
    calibration: AxisCalibration,
    mesh_image_rgb: np.ndarray,
    *,
    max_snap_distance_px: float = 4.5,
) -> ReconstructionCandidate:
    marker_points = _white_marker_points(mesh_image_rgb)
    if not marker_points or not candidate.mesh.nodes:
        candidate.mesh.metadata["final_marker_center_snap_count"] = 0
        candidate.mesh.metadata["final_marker_center_snaps"] = []
        return candidate

    node_pixels = np.asarray(
        [[float(node.pixel_x), float(node.pixel_y)] for node in candidate.mesh.nodes],
        dtype=float,
    )
    marker_snap_candidates: dict[int, list[tuple[float, int, np.ndarray, float]]] = {}
    for node_idx, point in enumerate(node_pixels):
        best_marker_idx = -1
        best_marker_center: np.ndarray | None = None
        best_marker_radius = 0.0
        best_distance = float("inf")
        for marker_idx, (marker_center, marker_radius) in enumerate(marker_points):
            distance = float(np.linalg.norm(point - marker_center))
            if distance < best_distance:
                best_marker_idx = marker_idx
                best_marker_center = np.asarray(marker_center, dtype=float)
                best_marker_radius = float(marker_radius)
                best_distance = distance
        if best_marker_idx < 0 or best_marker_center is None:
            continue
        snap_threshold = min(float(max_snap_distance_px), max(1.0, best_marker_radius * 1.5))
        if best_distance <= snap_threshold:
            marker_snap_candidates.setdefault(best_marker_idx, []).append(
                (best_distance, node_idx, best_marker_center, best_marker_radius)
            )

    snapped_pixels = node_pixels.copy()
    snap_records: list[dict[str, float | int | str]] = []
    for marker_idx, candidates in marker_snap_candidates.items():
        candidates.sort(key=lambda item: item[0])
        distance_before, node_idx, marker_center, marker_radius = candidates[0]
        snap_delta = marker_center - snapped_pixels[node_idx]
        snap_distance = float(np.linalg.norm(snap_delta))
        if snap_distance <= 1e-6 or snap_distance > max_snap_distance_px:
            continue
        snapped_pixels[node_idx] = marker_center
        snap_records.append(
            {
                "node_id": int(candidate.mesh.nodes[node_idx].node_id),
                "node_index": int(node_idx),
                "marker_index": int(marker_idx),
                "snap_reason": "selected_marker_center_small_snap",
                "snap_dx_px": float(snap_delta[0]),
                "snap_dy_px": float(snap_delta[1]),
                "snap_distance_px": snap_distance,
                "base_marker_distance_before_px": float(distance_before),
                "base_marker_distance_after_px": 0.0,
                "marker_radius_px": float(marker_radius),
                "competing_vertex_count": int(len(candidates)),
            }
        )

    if not snap_records:
        candidate.mesh.metadata["final_marker_center_snap_count"] = 0
        candidate.mesh.metadata["final_marker_center_snaps"] = []
        return candidate

    raw_world_xy = calibration.pixel_to_world(snapped_pixels)
    nodes = [
        Node(
            node_id=node.node_id,
            x=float(raw_world_xy[idx, 0]),
            y=float(raw_world_xy[idx, 1]),
            z=float(node.z),
            pixel_x=float(snapped_pixels[idx, 0]),
            pixel_y=float(snapped_pixels[idx, 1]),
        )
        for idx, node in enumerate(candidate.mesh.nodes)
    ]

    wireframe = dict(candidate.wireframe)
    vertices = np.asarray(wireframe.get("vertices", []), dtype=float)
    if len(vertices) > 0:
        snapped_vertices = vertices.copy()
        for idx, node in enumerate(candidate.mesh.nodes):
            vertex_idx = int(node.node_id) - 1
            if 0 <= vertex_idx < len(snapped_vertices):
                snapped_vertices[vertex_idx] = snapped_pixels[idx]
        wireframe["vertices"] = snapped_vertices.tolist()
        candidate.wireframe = wireframe

    candidate.mesh = NasMesh(
        nodes=nodes,
        elements=list(candidate.mesh.elements),
        metadata={
            **candidate.mesh.metadata,
            "final_marker_center_snap_count": int(len(snap_records)),
            "final_marker_center_snaps": snap_records[:80],
            "final_marker_center_snap_max_distance_px": float(max_snap_distance_px),
        },
    )
    return candidate


def build_hawp_refined_candidates(
    base_candidate: ReconstructionCandidate,
    wireframe: dict,
    edge_threshold: float,
    calibration: AxisCalibration,
    candidate_prefix: str,
    image_shape: tuple[int, int] | None = None,
    blend_alphas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> list[ReconstructionCandidate]:
    base_vertices_px = np.array(base_candidate.wireframe["vertices"], dtype=float)
    if len(base_vertices_px) == 0:
        return []

    active_vertices_px, active_edges, active_weights, hawp_graph_stats = _prepare_hawp_graph(
        wireframe,
        edge_threshold,
        image_shape=image_shape,
    )
    if len(active_vertices_px) < len(base_vertices_px):
        active_vertices_px, active_edges, active_weights = _wireframe_subgraph(wireframe, edge_threshold)
        hawp_graph_stats = {
            **hawp_graph_stats,
            "processed_fallback": True,
            "processed_vertex_count": int(len(active_vertices_px)),
            "processed_edge_count": int(len(active_edges)),
        }
    if len(active_vertices_px) < len(base_vertices_px):
        return []

    matching_cost = distance_matrix(base_vertices_px, active_vertices_px)
    row_ind, col_ind = linear_sum_assignment(matching_cost)
    if len(row_ind) != len(base_vertices_px):
        return []

    aligned_vertices_px = np.zeros_like(base_vertices_px)
    aligned_source_indices = np.full(len(base_vertices_px), -1, dtype=int)
    aligned_vertices_px[row_ind] = active_vertices_px[col_ind]
    aligned_source_indices[row_ind] = col_ind
    assignment_errors = matching_cost[row_ind, col_ind]

    active_adjacency = _wireframe_adjacency(len(active_vertices_px), active_edges)
    active_edge_weight_map = _wireframe_edge_weight_map(active_edges, active_weights)
    base_adjacency = _wireframe_adjacency(
        len(base_vertices_px),
        [tuple(edge) for edge in base_candidate.wireframe["edges"]],
    )

    if image_shape is not None:
        image_diag = math.hypot(float(image_shape[1]), float(image_shape[0]))
    else:
        span = np.ptp(base_vertices_px, axis=0)
        image_diag = float(np.linalg.norm(span))
    max_projection_px = max(12.0, image_diag * 0.012)

    segment_guided_targets = np.array(aligned_vertices_px, copy=True)
    for base_idx, source_idx in enumerate(aligned_source_indices.tolist()):
        if source_idx < 0:
            continue

        source_neighbors = list(active_adjacency.get(source_idx, ()))
        local_projections: list[tuple[float, np.ndarray, float]] = []
        for neighbor_idx in source_neighbors:
            start = active_vertices_px[source_idx]
            end = active_vertices_px[neighbor_idx]
            segment = end - start
            denom = float(np.dot(segment, segment))
            if denom <= 1e-6:
                continue
            t = float(np.clip(np.dot(base_vertices_px[base_idx] - start, segment) / denom, 0.0, 1.0))
            projection = start + t * segment
            distance = float(np.linalg.norm(base_vertices_px[base_idx] - projection))
            if distance > max_projection_px:
                continue
            weight = active_edge_weight_map.get(tuple(sorted((source_idx, neighbor_idx))), 0.0)
            local_projections.append((distance, projection, weight))

        if local_projections:
            local_projections.sort(key=lambda item: (item[0], -item[2]))
            best_projection = local_projections[0][1]
            if len(base_adjacency.get(base_idx, ())) <= 2:
                target_point = 0.65 * best_projection + 0.35 * aligned_vertices_px[base_idx]
            else:
                target_point = 0.35 * best_projection + 0.65 * aligned_vertices_px[base_idx]
        else:
            target_point = aligned_vertices_px[base_idx]

        displacement = target_point - base_vertices_px[base_idx]
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm > max_projection_px:
            target_point = base_vertices_px[base_idx] + displacement / displacement_norm * max_projection_px
        segment_guided_targets[base_idx] = target_point

    candidates: list[ReconstructionCandidate] = []
    for alpha in blend_alphas:
        blended_vertices_px = (1.0 - alpha) * base_vertices_px + alpha * segment_guided_targets
        blended_world_xy = calibration.pixel_to_world(blended_vertices_px)
        nodes = [
            Node(
                node_id=base_node.node_id,
                x=float(x_world),
                y=float(y_world),
                pixel_x=float(x_px),
                pixel_y=float(y_px),
            )
            for base_node, (x_world, y_world), (x_px, y_px) in zip(
                base_candidate.mesh.nodes,
                blended_world_xy.tolist(),
                blended_vertices_px.tolist(),
            )
        ]
        candidate_wireframe = {
            "vertices": blended_vertices_px.tolist(),
            "vertices-score": [1.0 for _ in range(len(blended_vertices_px))],
            "edges": base_candidate.wireframe["edges"],
            "edges-weights": base_candidate.wireframe["edges-weights"],
            "height": int(image_shape[0]) if image_shape is not None else base_candidate.wireframe.get("height"),
            "width": int(image_shape[1]) if image_shape is not None else base_candidate.wireframe.get("width"),
        }
        candidates.append(
            ReconstructionCandidate(
                name=f"{candidate_prefix}_blend_{alpha:.2f}",
                merge_threshold=base_candidate.merge_threshold,
                ratio_to_triangle=base_candidate.ratio_to_triangle,
                ratio_to_quad=base_candidate.ratio_to_quad,
                mesh=NasMesh(
                    nodes=nodes,
                    elements=list(base_candidate.mesh.elements),
                    metadata={
                        **base_candidate.mesh.metadata,
                        "source": "hawp_refine",
                        "edge_threshold": edge_threshold,
                        "blend_alpha": alpha,
                        "hawp_assignment_mean_px": float(np.mean(assignment_errors)),
                        "hawp_assignment_max_px": float(np.max(assignment_errors)),
                        **hawp_graph_stats,
                    },
                ),
                wireframe=candidate_wireframe,
                source="hawp_refine",
                proposal_family="hawp_refine",
            )
        )
    return candidates


def build_candidate_grid(
    components: list[ComponentPolygon],
    calibration: AxisCalibration,
    prop_id: int,
    image_shape: tuple[int, int] | None = None,
    merge_thresholds: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0),
    ratio_to_triangle_values: tuple[float, ...] = (0.03, 0.04, 0.05, 0.06, 0.08, 0.1),
    ratio_to_quad_values: tuple[float, ...] = (0.03, 0.05, 0.08, 0.1, 0.15),
) -> list[ReconstructionCandidate]:
    candidates: list[ReconstructionCandidate] = []
    if not components:
        return candidates

    for ratio_to_triangle in ratio_to_triangle_values:
        for ratio_to_quad in ratio_to_quad_values:
            collapsed_components = [
                collapse_short_edges(component.polygon_px, ratio_to_triangle, ratio_to_quad)
                for component in components
            ]
            if not collapsed_components:
                continue
            all_points = np.vstack(collapsed_components)
            for merge_threshold in merge_thresholds:
                centers_px, point_cluster_ids = cluster_points(all_points, merge_threshold)
                point_cluster_ids = list(point_cluster_ids)

                pixel_nodes: list[Node] = []
                world_xy = calibration.pixel_to_world(centers_px)
                for node_idx, ((x_world, y_world), (x_px, y_px)) in enumerate(
                    zip(world_xy, centers_px.tolist()),
                    start=1,
                ):
                    pixel_nodes.append(
                        Node(
                            node_id=node_idx,
                            x=float(x_world),
                            y=float(y_world),
                            pixel_x=float(x_px),
                            pixel_y=float(y_px),
                        )
                    )

                elements: list[Element] = []
                element_cycles: list[list[int]] = []
                cursor = 0
                for component_idx, polygon in enumerate(collapsed_components, start=1):
                    raw_cycle = point_cluster_ids[cursor : cursor + len(polygon)]
                    cursor += len(polygon)
                    cycle = [cluster_id + 1 for cluster_id in dedupe_sequence(raw_cycle)]
                    if len(cycle) not in {3, 4}:
                        continue
                    card = "CTRIA3" if len(cycle) == 3 else "CQUAD4"
                    elements.append(
                        Element(
                            card=card,
                            element_id=component_idx,
                            prop_id=prop_id,
                            node_ids=cycle,
                        )
                    )
                    element_cycles.append(cycle)

                wireframe_edges = unique_edges_from_cycles(element_cycles)
                wireframe = {
                    "vertices": [[node.pixel_x, node.pixel_y] for node in pixel_nodes],
                    "vertices-score": [1.0 for _ in pixel_nodes],
                    "edges": [[left - 1, right - 1] for left, right in wireframe_edges],
                    "edges-weights": [1.0 for _ in wireframe_edges],
                    "height": int(image_shape[0]) if image_shape is not None else int(round(calibration.origin_y + 80)),
                    "width": int(image_shape[1]) if image_shape is not None else int(round(calibration.x_axis_end + 80)),
                }
                mesh = NasMesh(
                    nodes=pixel_nodes,
                    elements=elements,
                    metadata={
                        "merge_threshold": merge_threshold,
                        "ratio_to_triangle": ratio_to_triangle,
                        "ratio_to_quad": ratio_to_quad,
                    },
                )
                candidates.append(
                    ReconstructionCandidate(
                        name=f"merge_{merge_threshold:g}_tri_{ratio_to_triangle:.2f}_quad_{ratio_to_quad:.2f}",
                        merge_threshold=merge_threshold,
                        ratio_to_triangle=ratio_to_triangle,
                        ratio_to_quad=ratio_to_quad,
                        mesh=mesh,
                        wireframe=wireframe,
                        source="classical",
                        proposal_family="classical",
                    )
                )

    return candidates


def save_mesh_overlay(
    image_rgb: np.ndarray,
    mesh: NasMesh,
    save_path: Path,
    *,
    label_nodes: bool = False,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    node_lookup = {node.node_id: node for node in mesh.nodes}
    for element in mesh.elements:
        points = [
            (node_lookup[node_id].pixel_x, node_lookup[node_id].pixel_y)
            for node_id in element.node_ids
        ]
        color = (0, 255, 255) if element.card == "CQUAD4" else (255, 128, 0)
        draw.line(points + [points[0]], fill=color, width=3)

    for node in mesh.nodes:
        radius = 5
        draw.ellipse(
            (node.pixel_x - radius, node.pixel_y - radius, node.pixel_x + radius, node.pixel_y + radius),
            fill=(255, 0, 0),
            outline=(255, 255, 255),
        )
        if label_nodes:
            draw.text((node.pixel_x + 6, node.pixel_y + 6), str(node.node_id), fill=(255, 255, 255))

    image.save(save_path)


def save_wireframe_overlay(
    image_rgb: np.ndarray,
    wireframe: dict,
    save_path: Path,
    *,
    label_vertices: bool = False,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    vertices = np.asarray(wireframe.get("vertices", []), dtype=float)
    edges = np.asarray(wireframe.get("edges", []), dtype=int)
    weights = np.asarray(wireframe.get("edges-weights", []), dtype=float)

    for edge_idx, edge in enumerate(edges.tolist()):
        if len(edge) != 2:
            continue
        left, right = edge
        if left >= len(vertices) or right >= len(vertices) or left < 0 or right < 0:
            continue
        weight = float(weights[edge_idx]) if edge_idx < len(weights) else 1.0
        color = (0, 255, 255) if weight >= 0.5 else (255, 192, 0)
        draw.line([tuple(vertices[left].tolist()), tuple(vertices[right].tolist())], fill=color, width=2)

    for vertex_idx, vertex in enumerate(vertices.tolist(), start=1):
        radius = 3
        draw.ellipse(
            (vertex[0] - radius, vertex[1] - radius, vertex[0] + radius, vertex[1] + radius),
            fill=(255, 0, 0),
            outline=(255, 255, 255),
        )
        if label_vertices and len(vertices) <= 120:
            draw.text((vertex[0] + 4, vertex[1] + 4), str(vertex_idx), fill=(255, 255, 255))

    image.save(save_path)
