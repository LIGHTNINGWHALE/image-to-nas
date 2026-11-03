from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from .geometry import polygon_area
from .nas import NasMesh


def _segments_intersect(first_start: np.ndarray, first_end: np.ndarray, second_start: np.ndarray, second_end: np.ndarray) -> bool:
    def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(first_start, second_start, second_end) != ccw(first_end, second_start, second_end) and ccw(
        first_start,
        first_end,
        second_start,
    ) != ccw(first_start, first_end, second_end)


def validate_mesh(mesh: NasMesh, expected_prop_id: int | None = None) -> dict:
    issues: list[str] = []
    node_ids = [node.node_id for node in mesh.nodes]
    element_ids = [element.element_id for element in mesh.elements]
    if len(node_ids) != len(set(node_ids)):
        issues.append("duplicate_node_ids")
    if len(element_ids) != len(set(element_ids)):
        issues.append("duplicate_element_ids")

    node_lookup = {node.node_id: node for node in mesh.nodes}
    edge_counter: Counter[tuple[int, int]] = Counter()
    element_signatures: set[tuple[str, tuple[int, ...]]] = set()
    referenced_nodes: set[int] = set()
    invalid_area_count = 0
    self_intersections = 0

    for element in mesh.elements:
        if element.card not in {"CTRIA3", "CQUAD4"}:
            issues.append(f"unsupported_card:{element.card}")
            continue
        expected_node_count = 3 if element.card == "CTRIA3" else 4
        if len(element.node_ids) != expected_node_count:
            issues.append(f"wrong_node_count:{element.element_id}")
            continue
        if expected_prop_id is not None and element.prop_id != expected_prop_id:
            issues.append(f"unexpected_prop_id:{element.element_id}")
        signature = (element.card, tuple(sorted(element.node_ids)))
        if signature in element_signatures:
            issues.append(f"duplicate_element:{element.element_id}")
        element_signatures.add(signature)
        try:
            polygon = np.asarray(
                [
                    [
                        node_lookup[node_id].pixel_x if node_lookup[node_id].pixel_x is not None else node_lookup[node_id].x,
                        node_lookup[node_id].pixel_y if node_lookup[node_id].pixel_y is not None else node_lookup[node_id].y,
                    ]
                    for node_id in element.node_ids
                ],
                dtype=float,
            )
        except KeyError:
            issues.append(f"unknown_node_ref:{element.element_id}")
            continue
        if np.any(~np.isfinite(polygon)):
            issues.append(f"non_finite_node:{element.element_id}")
            continue
        area = polygon_area(polygon)
        if abs(area) <= 1e-4:
            invalid_area_count += 1
        referenced_nodes.update(element.node_ids)
        for idx, node_id in enumerate(element.node_ids):
            nxt = element.node_ids[(idx + 1) % len(element.node_ids)]
            if node_id == nxt:
                issues.append(f"zero_length_edge:{element.element_id}")
                continue
            edge_counter[tuple(sorted((node_id, nxt)))] += 1
        segments = [(polygon[idx], polygon[(idx + 1) % len(polygon)]) for idx in range(len(polygon))]
        for (first_idx, first_seg), (second_idx, second_seg) in combinations(enumerate(segments), 2):
            if abs(first_idx - second_idx) <= 1 or {first_idx, second_idx} == {0, len(segments) - 1}:
                continue
            if _segments_intersect(first_seg[0], first_seg[1], second_seg[0], second_seg[1]):
                self_intersections += 1

    orphan_nodes = [node.node_id for node in mesh.nodes if node.node_id not in referenced_nodes]
    overused_edges = [edge for edge, count in edge_counter.items() if count > 2]
    if invalid_area_count:
        issues.append("degenerate_elements")
    if self_intersections:
        issues.append("self_intersections")
    if orphan_nodes:
        issues.append("orphan_nodes")
    if overused_edges:
        issues.append("edge_incidence_gt_2")

    structural_score = float(
        max(
            0.0,
            len(mesh.elements) * 5.0
            + len(referenced_nodes) * 1.5
            - 8.0 * len(issues)
            - 3.0 * len(orphan_nodes)
            - 2.0 * len(overused_edges),
        )
    )
    return {
        "valid": len(issues) == 0,
        "issues": sorted(set(issues)),
        "orphan_nodes": orphan_nodes,
        "edge_incidence_gt_2": [list(edge) for edge in overused_edges],
        "structural_score": structural_score,
        "node_count": len(mesh.nodes),
        "element_count": len(mesh.elements),
    }
