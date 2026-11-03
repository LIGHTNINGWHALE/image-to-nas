from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(slots=True)
class DisjointSet:
    parent: list[int]

    @classmethod
    def create(cls, n_items: int) -> "DisjointSet":
        return cls(parent=list(range(n_items)))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def order_polygon(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    if polygon_area(ordered) < 0:
        ordered = ordered[::-1]
    return ordered


def perpendicular_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        return np.linalg.norm(points - start, axis=1)
    rel = points - start
    return np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / norm


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    distances = perpendicular_distances(points, start, end)
    index = int(np.argmax(distances))
    max_distance = float(distances[index])
    if max_distance > epsilon:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def fit_line_tls(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    direction = vh[0]
    direction = direction / np.linalg.norm(direction)
    return centroid, direction


def intersect_lines(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    point_a, direction_a = first
    point_b, direction_b = second
    system = np.array(
        [
            [direction_a[0], -direction_b[0]],
            [direction_a[1], -direction_b[1]],
        ],
        dtype=float,
    )
    offset = point_b - point_a
    determinant = float(np.linalg.det(system))
    if abs(determinant) < 1e-8:
        return (point_a + point_b) / 2.0
    scale_a, _ = np.linalg.solve(system, offset)
    return point_a + scale_a * direction_a


def nearest_point_index(points: np.ndarray, target: np.ndarray) -> int:
    distances = np.sum((points - target) ** 2, axis=1)
    return int(np.argmin(distances))


def contour_arc(points: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if end_idx >= start_idx:
        return points[start_idx : end_idx + 1]
    return np.vstack([points[start_idx:], points[: end_idx + 1]])


def collapse_short_edges(
    polygon: np.ndarray,
    ratio_to_triangle: float,
    ratio_to_quad: float,
) -> np.ndarray:
    polygon = order_polygon(polygon)
    changed = True
    while changed:
        changed = False
        num_vertices = len(polygon)
        if num_vertices <= 3:
            break
        edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        median_length = float(np.median(edges))
        if num_vertices == 5 and edges.min() <= ratio_to_quad * median_length:
            shortest_idx = int(np.argmin(edges))
            polygon = np.delete(polygon, (shortest_idx + 1) % num_vertices, axis=0)
            changed = True
        elif num_vertices == 4 and edges.min() <= ratio_to_triangle * median_length:
            shortest_idx = int(np.argmin(edges))
            polygon = np.delete(polygon, (shortest_idx + 1) % num_vertices, axis=0)
            changed = True
    return order_polygon(polygon)


def dedupe_sequence(sequence: list[int]) -> list[int]:
    deduped: list[int] = []
    for item in sequence:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def cluster_points(points: np.ndarray, merge_threshold: float) -> tuple[np.ndarray, list[int]]:
    points = np.asarray(points, dtype=float)
    dsu = DisjointSet.create(len(points))
    for left, right in combinations(range(len(points)), 2):
        if np.linalg.norm(points[left] - points[right]) <= merge_threshold:
            dsu.union(left, right)

    grouped: dict[int, list[np.ndarray]] = {}
    for index, point in enumerate(points):
        grouped.setdefault(dsu.find(index), []).append(point)

    root_to_id = {root: idx for idx, root in enumerate(grouped)}
    centers = np.array([np.mean(group, axis=0) for group in grouped.values()], dtype=float)
    ids = [root_to_id[dsu.find(index)] for index in range(len(points))]
    return centers, ids


def unique_edges_from_cycles(cycles: list[list[int]]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for cycle in cycles:
        for idx, node_id in enumerate(cycle):
            nxt = cycle[(idx + 1) % len(cycle)]
            edge = tuple(sorted((node_id, nxt)))
            edges.add(edge)
    return sorted(edges)
