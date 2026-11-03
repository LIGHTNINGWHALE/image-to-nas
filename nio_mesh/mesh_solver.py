from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
from ortools.sat.python import cp_model

from .axis import AxisCalibration
from .geometry import order_polygon, polygon_area, unique_edges_from_cycles
from .mesh_validator import validate_mesh
from .nas import Element, NasMesh, Node
from .reconstruct import ReconstructionCandidate


def _wireframe_segments(wireframe: dict) -> list[tuple[np.ndarray, np.ndarray, float]]:
    vertices = np.asarray(wireframe.get("vertices", []), dtype=float)
    edges = np.asarray(wireframe.get("edges", []), dtype=int)
    weights = np.asarray(wireframe.get("edges-weights", []), dtype=float)
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for edge, weight in zip(edges.tolist(), weights.tolist()):
        left, right = edge
        if left == right or left >= len(vertices) or right >= len(vertices):
            continue
        segments.append((vertices[left], vertices[right], float(weight)))
    return segments


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _segment_support_score(
    start: np.ndarray,
    end: np.ndarray,
    evidence_segments: list[tuple[np.ndarray, np.ndarray, float]],
) -> float:
    edge = end - start
    edge_length = float(np.linalg.norm(edge))
    if edge_length <= 1e-6:
        return 0.0
    best = 0.0
    for seg_start, seg_end, weight in evidence_segments:
        seg_vec = seg_end - seg_start
        seg_length = float(np.linalg.norm(seg_vec))
        if seg_length <= 1e-6:
            continue
        cosine = float(np.dot(edge, seg_vec) / (edge_length * seg_length))
        if abs(cosine) < 0.90:
            continue
        distance = 0.5 * (_point_line_distance(start, seg_start, seg_end) + _point_line_distance(end, seg_start, seg_end))
        distance_score = max(0.0, 1.0 - distance / max(8.0, edge_length * 0.35))
        overlap_score = min(edge_length, seg_length) / max(edge_length, seg_length)
        best = max(best, float(weight) * distance_score * overlap_score)
    return best


def _centroid_inside_mask(points: np.ndarray, mask: np.ndarray) -> bool:
    centroid = points.mean(axis=0)
    x = int(round(float(centroid[0])))
    y = int(round(float(centroid[1])))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def _yellow_mask(mesh_input_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(mesh_input_rgb)
    return (image[:, :, 0] > 160) & (image[:, :, 1] > 160) & (image[:, :, 2] < 110)


def _candidate_faces_from_base_nodes(
    base_candidate: ReconstructionCandidate,
    evidence_segments: list[tuple[np.ndarray, np.ndarray, float]],
    mesh_input_rgb: np.ndarray,
) -> list[dict]:
    base_points = np.asarray([[node.pixel_x, node.pixel_y] for node in base_candidate.mesh.nodes], dtype=float)
    yellow_mask = _yellow_mask(mesh_input_rgb)
    base_cycles = {tuple(element.node_ids) for element in base_candidate.mesh.elements}
    candidates: list[dict] = []
    pair_scores: dict[tuple[int, int], float] = {}
    base_edges = {tuple(sorted(edge)) for edge in unique_edges_from_cycles([element.node_ids for element in base_candidate.mesh.elements])}
    base_edge_lengths = [
        float(np.linalg.norm(base_points[left - 1] - base_points[right - 1]))
        for left, right in base_edges
        if 0 < left <= len(base_points) and 0 < right <= len(base_points)
    ]
    if base_edge_lengths:
        max_candidate_edge_length = max(24.0, float(np.percentile(base_edge_lengths, 95)) * 1.25)
    else:
        max_candidate_edge_length = 120.0

    candidate_neighbors: dict[int, set[int]] = {idx: set() for idx in range(len(base_points))}
    candidate_pair_set: set[tuple[int, int]] = set()
    for left, right in combinations(range(len(base_points)), 2):
        edge = tuple(sorted((left + 1, right + 1)))
        distance = float(np.linalg.norm(base_points[left] - base_points[right]))
        if edge not in base_edges and distance > max_candidate_edge_length:
            continue
        support = _segment_support_score(base_points[left], base_points[right], evidence_segments)
        if edge in base_edges:
            support = max(support, 0.55)
        pair_scores[edge] = support
        candidate_pair_set.add(edge)
        candidate_neighbors[left].add(right)
        candidate_neighbors[right].add(left)

    graph_neighbors: dict[int, set[int]] = {idx + 1: set() for idx in range(len(base_points))}
    for left, right in candidate_pair_set:
        graph_neighbors[left].add(right)
        graph_neighbors[right].add(left)

    seen_cycles: set[tuple[int, ...]] = set()

    def consider_cycle(node_ids: tuple[int, ...], card: str) -> None:
        polygon = order_polygon(base_points[[node_id - 1 for node_id in node_ids]])
        ordered_indices = []
        for point in polygon:
            ordered_indices.append(int(np.argmin(np.sum((base_points - point) ** 2, axis=1))) + 1)
        if len(set(ordered_indices)) != len(node_ids):
            return
        cycle = tuple(ordered_indices)
        if cycle in seen_cycles:
            return
        edge_scores = []
        for idx in range(len(polygon)):
            edge = tuple(sorted((ordered_indices[idx], ordered_indices[(idx + 1) % len(polygon)])))
            if edge not in candidate_pair_set:
                return
            edge_scores.append(pair_scores.get(edge, 0.0))
        seen_cycles.add(cycle)
        area = abs(polygon_area(polygon))
        if area <= 20.0:
            return
        if not _centroid_inside_mask(polygon, yellow_mask):
            return
        mean_edge_score = float(np.mean(edge_scores))
        if mean_edge_score < (0.20 if card == "CTRIA3" else 0.25):
            return
        if min(edge_scores) < (0.12 if card == "CTRIA3" else 0.15):
            return
        lengths = [
            float(np.linalg.norm(polygon[(idx + 1) % len(polygon)] - polygon[idx]))
            for idx in range(len(polygon))
        ]
        sliver_penalty = max(0.0, 1.0 - min(lengths) / max(max(lengths), 1e-6))
        candidates.append(
            {
                "card": card,
                "node_ids": list(cycle),
                "edge_scores": edge_scores,
                "score": mean_edge_score * 20.0 + area * 0.002 - sliver_penalty * 4.0 + (8.0 if cycle in base_cycles else 0.0),
            }
        )

    for a in range(1, len(base_points) + 1):
        for b in sorted(neighbor for neighbor in graph_neighbors[a] if neighbor > a):
            common = graph_neighbors[a].intersection(graph_neighbors[b])
            for c in sorted(neighbor for neighbor in common if neighbor > b):
                consider_cycle((a, b, c), "CTRIA3")

    for a in range(1, len(base_points) + 1):
        for b in graph_neighbors[a]:
            if b == a:
                continue
            for c in graph_neighbors[b]:
                if c in (a, b):
                    continue
                for d in graph_neighbors[c].intersection(graph_neighbors[a]):
                    if d in (a, b, c):
                        continue
                    if min(a, b, c, d) != a:
                        continue
                    consider_cycle((a, b, c, d), "CQUAD4")

    # Keep the search bounded but always preserve all original faces as options.
    selected: list[dict] = []
    for card in ("CTRIA3", "CQUAD4"):
        card_candidates = [item for item in candidates if item["card"] == card]
        card_candidates.sort(key=lambda item: item["score"], reverse=True)
        selected.extend(card_candidates[:120])
    for element in base_candidate.mesh.elements:
        cycle = tuple(element.node_ids)
        if not any(tuple(candidate["node_ids"]) == cycle for candidate in selected):
            selected.append(
                {
                    "card": element.card,
                    "node_ids": list(cycle),
                    "edge_scores": [1.0] * len(cycle),
                    "score": 25.0,
                }
            )
    return selected


def _solve_face_selection(face_candidates: list[dict], base_candidate: ReconstructionCandidate) -> tuple[list[dict], str]:
    target_counts = base_candidate.mesh.counts()
    tri_target = target_counts.get("CTRIA3", 0)
    quad_target = target_counts.get("CQUAD4", 0)

    model = cp_model.CpModel()
    variables = [model.NewBoolVar(f"face_{idx}") for idx in range(len(face_candidates))]

    tri_indices = [idx for idx, candidate in enumerate(face_candidates) if candidate["card"] == "CTRIA3"]
    quad_indices = [idx for idx, candidate in enumerate(face_candidates) if candidate["card"] == "CQUAD4"]
    model.Add(sum(variables[idx] for idx in tri_indices) == tri_target)
    model.Add(sum(variables[idx] for idx in quad_indices) == quad_target)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for idx, candidate in enumerate(face_candidates):
        node_ids = candidate["node_ids"]
        for edge_idx, node_id in enumerate(node_ids):
            edge = tuple(sorted((node_id, node_ids[(edge_idx + 1) % len(node_ids)])))
            edge_to_faces.setdefault(edge, []).append(idx)
    for face_indices in edge_to_faces.values():
        model.Add(sum(variables[idx] for idx in face_indices) <= 2)

    scaled_scores = [int(round(candidate["score"] * 1000.0)) for candidate in face_candidates]
    model.Maximize(sum(score * variable for score, variable in zip(scaled_scores, variables)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], solver.StatusName(status)
    selected = [candidate for idx, candidate in enumerate(face_candidates) if solver.Value(variables[idx])]
    return selected, solver.StatusName(status)


def _refine_nodes(
    base_candidate: ReconstructionCandidate,
    evidence_wireframes: list[dict],
    calibration: AxisCalibration,
) -> list[Node]:
    base_points = np.asarray([[node.pixel_x, node.pixel_y] for node in base_candidate.mesh.nodes], dtype=float)
    support_points: list[np.ndarray] = []
    for wireframe in evidence_wireframes:
        vertices = np.asarray(wireframe.get("vertices", []), dtype=float)
        if len(vertices):
            support_points.append(vertices)
    if not support_points:
        return list(base_candidate.mesh.nodes)

    support = np.vstack(support_points)
    refined_points = np.array(base_points, copy=True)
    for idx, point in enumerate(base_points):
        distances = np.linalg.norm(support - point, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])
        if nearest_distance > 24.0:
            continue
        alpha = max(0.15, min(0.70, 1.0 - nearest_distance / 24.0))
        refined_points[idx] = (1.0 - alpha) * point + alpha * support[nearest_idx]

    world_xy = calibration.pixel_to_world(refined_points)
    return [
        replace(
            node,
            x=float(world_xy[idx, 0]),
            y=float(world_xy[idx, 1]),
            pixel_x=float(refined_points[idx, 0]),
            pixel_y=float(refined_points[idx, 1]),
        )
        for idx, node in enumerate(base_candidate.mesh.nodes)
    ]


def solve_hybrid_candidate(
    base_candidate: ReconstructionCandidate,
    evidence_wireframes: list[dict],
    calibration: AxisCalibration,
    prop_id: int,
    mesh_input_rgb: np.ndarray,
    candidate_name: str = "hybrid_solver_union",
) -> ReconstructionCandidate | None:
    evidence_segments: list[tuple[np.ndarray, np.ndarray, float]] = _wireframe_segments(base_candidate.wireframe)
    for wireframe in evidence_wireframes:
        evidence_segments.extend(_wireframe_segments(wireframe))
    if not evidence_segments:
        return None

    face_candidates = _candidate_faces_from_base_nodes(base_candidate, evidence_segments, mesh_input_rgb)
    if not face_candidates:
        return None
    selected_faces, solver_status = _solve_face_selection(face_candidates, base_candidate)
    if not selected_faces:
        return None

    nodes = _refine_nodes(base_candidate, evidence_wireframes, calibration)
    elements = [
        Element(
            card=face["card"],
            element_id=element_idx,
            prop_id=prop_id,
            node_ids=list(face["node_ids"]),
        )
        for element_idx, face in enumerate(selected_faces, start=1)
    ]
    edge_cycles = [face["node_ids"] for face in selected_faces]
    wireframe_edges = unique_edges_from_cycles(edge_cycles)
    wireframe = {
        "vertices": [[node.pixel_x, node.pixel_y] for node in nodes],
        "vertices-score": [1.0 for _ in nodes],
        "edges": [[left - 1, right - 1] for left, right in wireframe_edges],
        "edges-weights": [1.0 for _ in wireframe_edges],
        "height": int(mesh_input_rgb.shape[0]),
        "width": int(mesh_input_rgb.shape[1]),
    }
    mesh = NasMesh(
        nodes=nodes,
        elements=elements,
        metadata={
            "source": "hybrid_solver",
            "solver_status": solver_status,
            "candidate_face_count": len(face_candidates),
        },
    )
    validation_report = validate_mesh(mesh, expected_prop_id=prop_id)
    return ReconstructionCandidate(
        name=candidate_name,
        merge_threshold=base_candidate.merge_threshold,
        ratio_to_triangle=base_candidate.ratio_to_triangle,
        ratio_to_quad=base_candidate.ratio_to_quad,
        mesh=mesh,
        wireframe=wireframe,
        source="hybrid_solver",
        proposal_family="hybrid_solver",
        structural_score=float(validation_report["structural_score"]),
        solver_status=solver_status,
        validation_report=validation_report,
    )
