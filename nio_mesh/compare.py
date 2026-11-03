from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from .nas import NasMesh, parse_nas


def _mesh_node_array(mesh: NasMesh) -> np.ndarray:
    return np.array([[node.x, node.y] for node in mesh.nodes], dtype=float)


def compare_meshes(predicted: NasMesh, truth: NasMesh) -> dict:
    predicted_counts = predicted.counts()
    truth_counts = truth.counts()

    predicted_xy = _mesh_node_array(predicted)
    truth_xy = _mesh_node_array(truth)

    if len(predicted_xy) and len(truth_xy):
        distances = distance_matrix(predicted_xy, truth_xy)
        row_ind, col_ind = linear_sum_assignment(distances)
        matched_distances = distances[row_ind, col_ind]
    else:
        row_ind = np.array([], dtype=int)
        col_ind = np.array([], dtype=int)
        matched_distances = np.array([], dtype=float)

    predicted_node_ids = [node.node_id for node in predicted.nodes]
    truth_node_ids = [node.node_id for node in truth.nodes]

    pred_to_truth = {
        predicted_node_ids[pred_index]: truth_node_ids[truth_index]
        for pred_index, truth_index in zip(row_ind.tolist(), col_ind.tolist())
    }

    matched_pred_ids = set(pred_to_truth)
    matched_truth_ids = set(pred_to_truth.values())
    unmatched_pred_nodes = [
        node.node_id for node in predicted.nodes if node.node_id not in matched_pred_ids
    ]
    unmatched_truth_nodes = [
        node.node_id for node in truth.nodes if node.node_id not in matched_truth_ids
    ]

    truth_elements = Counter(
        (element.card, tuple(sorted(element.node_ids))) for element in truth.elements
    )
    predicted_elements = Counter()
    unresolved_elements: list[dict] = []

    for element in predicted.elements:
        try:
            mapped_nodes = tuple(sorted(pred_to_truth[node_id] for node_id in element.node_ids))
        except KeyError:
            unresolved_elements.append(
                {
                    "element_id": element.element_id,
                    "card": element.card,
                    "reason": "unmatched node in prediction",
                    "node_ids": element.node_ids,
                }
            )
            continue
        predicted_elements[(element.card, mapped_nodes)] += 1

    unexpected_elements = list((predicted_elements - truth_elements).elements())
    missing_elements = list((truth_elements - predicted_elements).elements())

    count_match = {
        "GRID": predicted_counts.get("GRID", 0) == truth_counts.get("GRID", 0),
        "CTRIA3": predicted_counts.get("CTRIA3", 0) == truth_counts.get("CTRIA3", 0),
        "CQUAD4": predicted_counts.get("CQUAD4", 0) == truth_counts.get("CQUAD4", 0),
    }

    return {
        "predicted_counts": predicted_counts,
        "truth_counts": truth_counts,
        "count_match": count_match,
        "node_metrics": {
            "matched_nodes": int(len(matched_distances)),
            "mean_abs_error": float(matched_distances.mean()) if matched_distances.size else None,
            "rmse": float(np.sqrt(np.mean(matched_distances**2))) if matched_distances.size else None,
            "max_error": float(matched_distances.max()) if matched_distances.size else None,
        },
        "unmatched_pred_nodes": unmatched_pred_nodes,
        "unmatched_truth_nodes": unmatched_truth_nodes,
        "unexpected_elements": [
            {"card": card, "truth_node_ids": list(node_ids)} for card, node_ids in unexpected_elements
        ],
        "missing_elements": [
            {"card": card, "truth_node_ids": list(node_ids)} for card, node_ids in missing_elements
        ],
        "unresolved_elements": unresolved_elements,
        "all_counts_exact": all(count_match.values()),
    }


def compare_nas_files(predicted_path: str | Path, truth_path: str | Path) -> dict:
    report = compare_meshes(parse_nas(predicted_path), parse_nas(truth_path))
    report["predicted_path"] = str(Path(predicted_path))
    report["truth_path"] = str(Path(truth_path))
    return report


def write_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> None:
    import json

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        f"# Comparison Report: {Path(report['predicted_path']).stem}",
        "",
        "## Count Match",
        f"- GRID: {report['predicted_counts'].get('GRID', 0)} vs {report['truth_counts'].get('GRID', 0)}",
        f"- CTRIA3: {report['predicted_counts'].get('CTRIA3', 0)} vs {report['truth_counts'].get('CTRIA3', 0)}",
        f"- CQUAD4: {report['predicted_counts'].get('CQUAD4', 0)} vs {report['truth_counts'].get('CQUAD4', 0)}",
        "",
        "## Node Error",
        f"- matched nodes: {report['node_metrics']['matched_nodes']}",
        f"- mean abs error: {report['node_metrics']['mean_abs_error']}",
        f"- rmse: {report['node_metrics']['rmse']}",
        f"- max error: {report['node_metrics']['max_error']}",
        "",
        "## Unmatched Nodes",
        f"- predicted: {report['unmatched_pred_nodes']}",
        f"- truth: {report['unmatched_truth_nodes']}",
        "",
        "## Element Issues",
        f"- unexpected: {report['unexpected_elements']}",
        f"- missing: {report['missing_elements']}",
        f"- unresolved: {report['unresolved_elements']}",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")

