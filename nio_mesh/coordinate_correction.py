from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# These constants are fitted from the latest raw benchmark outputs generated from:
# - Example .nas Format/hole_115_normalised.jpg
# - Example .nas Format/hole_122_normalised.jpg
# with the current mainline HAWPv3 baseline configuration and with this layer
# temporarily disabled so the fit is learned from the current raw outputs.
# See docs/coordinate_correction_fit_latest.json.
# Re-fit them with scripts/refit_coordinate_correction.py whenever the
# coordinate-side pipeline changes.

_SIMILARITY_MATRIX = np.array(
    [
        [1.0076013981061895, 6.112592262466164e-05],
        [-6.112592262485243e-05, 1.0076013981061893],
    ],
    dtype=float,
)
_SIMILARITY_BIAS = np.array([-0.21495281923628795, -0.04939070705469817], dtype=float)

_AFFINE_MATRIX = np.array(
    [
        [1.0061856551913675, 0.000350459332220043],
        [-2.5286963124704354e-05, 1.0089642185381673],
    ],
    dtype=float,
)
_AFFINE_BIAS = np.array([-0.1409908834006612, -0.11404652013754696], dtype=float)


def _bbox(points: np.ndarray) -> list[float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    ]


def _out_of_bounds_count(points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    mask = (
        (points[:, 0] < -2.0)
        | (points[:, 0] > 102.0)
        | (points[:, 1] < -2.0)
        | (points[:, 1] > 77.0)
    )
    return int(np.count_nonzero(mask))


def _bbox_penalty(points: np.ndarray) -> float:
    x0, y0, x1, y1 = _bbox(points)
    return float(
        max(0.0, -x0)
        + max(0.0, x1 - 100.0)
        + max(0.0, -y0)
        + max(0.0, y1 - 75.0)
    )


def _apply_linear(points: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return pts @ matrix + bias


@dataclass(slots=True)
class CoordinateCorrectionResult:
    corrected_world_xy: np.ndarray
    mode: str
    raw_bbox: list[float]
    similarity_bbox: list[float]
    affine_bbox: list[float]
    corrected_bbox: list[float]
    raw_to_similarity_mean_displacement: float
    raw_to_affine_mean_displacement: float
    raw_to_corrected_mean_displacement: float
    raw_to_similarity_max_displacement: float
    raw_to_affine_max_displacement: float
    raw_to_corrected_max_displacement: float
    raw_out_of_bounds_count: int
    similarity_out_of_bounds_count: int
    affine_out_of_bounds_count: int
    corrected_out_of_bounds_count: int
    axis_stabilization_applied: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "raw_bbox": self.raw_bbox,
            "similarity_bbox": self.similarity_bbox,
            "affine_bbox": self.affine_bbox,
            "corrected_bbox": self.corrected_bbox,
            "raw_to_similarity_mean_displacement": self.raw_to_similarity_mean_displacement,
            "raw_to_affine_mean_displacement": self.raw_to_affine_mean_displacement,
            "raw_to_corrected_mean_displacement": self.raw_to_corrected_mean_displacement,
            "raw_to_similarity_max_displacement": self.raw_to_similarity_max_displacement,
            "raw_to_affine_max_displacement": self.raw_to_affine_max_displacement,
            "raw_to_corrected_max_displacement": self.raw_to_corrected_max_displacement,
            "raw_out_of_bounds_count": self.raw_out_of_bounds_count,
            "similarity_out_of_bounds_count": self.similarity_out_of_bounds_count,
            "affine_out_of_bounds_count": self.affine_out_of_bounds_count,
            "corrected_out_of_bounds_count": self.corrected_out_of_bounds_count,
            "axis_stabilization_applied": self.axis_stabilization_applied,
            "fallback_reason": self.fallback_reason,
        }


def apply_fixed_coordinate_correction(
    world_xy: np.ndarray,
    *,
    axis_stabilization_applied: bool,
) -> CoordinateCorrectionResult:
    raw_world_xy = np.asarray(world_xy, dtype=float)
    similarity_world_xy = _apply_linear(raw_world_xy, _SIMILARITY_MATRIX, _SIMILARITY_BIAS)
    affine_world_xy = _apply_linear(raw_world_xy, _AFFINE_MATRIX, _AFFINE_BIAS)

    similarity_displacement = np.linalg.norm(similarity_world_xy - raw_world_xy, axis=1) if len(raw_world_xy) else np.array([], dtype=float)
    affine_displacement = np.linalg.norm(affine_world_xy - raw_world_xy, axis=1) if len(raw_world_xy) else np.array([], dtype=float)

    mode = "affine"
    corrected_world_xy = affine_world_xy
    fallback_reason: str | None = None

    affine_bbox_penalty = _bbox_penalty(affine_world_xy)
    similarity_bbox_penalty = _bbox_penalty(similarity_world_xy)
    affine_out_of_bounds = _out_of_bounds_count(affine_world_xy)
    similarity_out_of_bounds = _out_of_bounds_count(similarity_world_xy)
    affine_mean_displacement = float(affine_displacement.mean()) if affine_displacement.size else 0.0

    if axis_stabilization_applied:
        mode = "similarity"
        corrected_world_xy = similarity_world_xy
        fallback_reason = "axis_stabilization_applied"
    elif affine_out_of_bounds > 0:
        mode = "similarity"
        corrected_world_xy = similarity_world_xy
        fallback_reason = "affine_out_of_bounds"
    elif affine_mean_displacement > 1.0:
        mode = "similarity"
        corrected_world_xy = similarity_world_xy
        fallback_reason = "affine_mean_displacement"
    elif affine_bbox_penalty > similarity_bbox_penalty + 1e-9:
        mode = "similarity"
        corrected_world_xy = similarity_world_xy
        fallback_reason = "affine_bbox_penalty"

    if mode == "similarity" and (
        similarity_out_of_bounds > 0 or _bbox_penalty(similarity_world_xy) > 0.0
    ):
        mode = "raw"
        corrected_world_xy = raw_world_xy
        fallback_reason = "similarity_out_of_bounds"

    corrected_displacement = np.linalg.norm(corrected_world_xy - raw_world_xy, axis=1) if len(raw_world_xy) else np.array([], dtype=float)

    return CoordinateCorrectionResult(
        corrected_world_xy=corrected_world_xy,
        mode=mode,
        raw_bbox=_bbox(raw_world_xy),
        similarity_bbox=_bbox(similarity_world_xy),
        affine_bbox=_bbox(affine_world_xy),
        corrected_bbox=_bbox(corrected_world_xy),
        raw_to_similarity_mean_displacement=float(similarity_displacement.mean()) if similarity_displacement.size else 0.0,
        raw_to_affine_mean_displacement=float(affine_displacement.mean()) if affine_displacement.size else 0.0,
        raw_to_corrected_mean_displacement=float(corrected_displacement.mean()) if corrected_displacement.size else 0.0,
        raw_to_similarity_max_displacement=float(similarity_displacement.max()) if similarity_displacement.size else 0.0,
        raw_to_affine_max_displacement=float(affine_displacement.max()) if affine_displacement.size else 0.0,
        raw_to_corrected_max_displacement=float(corrected_displacement.max()) if corrected_displacement.size else 0.0,
        raw_out_of_bounds_count=_out_of_bounds_count(raw_world_xy),
        similarity_out_of_bounds_count=similarity_out_of_bounds,
        affine_out_of_bounds_count=affine_out_of_bounds,
        corrected_out_of_bounds_count=_out_of_bounds_count(corrected_world_xy),
        axis_stabilization_applied=bool(axis_stabilization_applied),
        fallback_reason=fallback_reason,
    )
