from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .geometry import fit_line_tls, intersect_lines


_NORMALIZED_AXIS_TEMPLATE = {
    "width": 1920.0,
    "height": 1080.0,
    "origin_x": 337.0004714434814,
    "origin_y": 999.5179968701095,
    "x_axis_end": 1587.0,
    "y_axis_top": 61.00000002924628,
}


@dataclass(slots=True)
class AxisCalibration:
    origin_x: float
    origin_y: float
    x_axis_end: float
    y_axis_top: float
    x_extent: float = 100.0
    y_extent: float = 75.0

    @property
    def x_scale(self) -> float:
        return self.x_extent / (self.x_axis_end - self.origin_x)

    @property
    def y_scale(self) -> float:
        return self.y_extent / (self.origin_y - self.y_axis_top)

    def pixel_to_world(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        x = (pts[:, 0] - self.origin_x) * self.x_scale
        y = (self.origin_y - pts[:, 1]) * self.y_scale
        return np.column_stack([x, y])

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        x = pts[:, 0] / self.x_scale + self.origin_x
        y = self.origin_y - pts[:, 1] / self.y_scale
        return np.column_stack([x, y])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AxisCalibrationDiagnostics:
    raw: AxisCalibration
    stabilized: AxisCalibration
    stabilization_applied: bool
    stabilization_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw.to_dict(),
            "stabilized": self.stabilized.to_dict(),
            "stabilization_applied": bool(self.stabilization_applied),
            "stabilization_reason": self.stabilization_reason,
        }


def magenta_axis_mask(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    return (
        (image[:, :, 0] > 170)
        & (image[:, :, 2] > 170)
        & (image[:, :, 1] < 120)
    )


def yellow_mesh_mask(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    return (
        (image[:, :, 0] > 160)
        & (image[:, :, 1] > 160)
        & (image[:, :, 2] < 110)
    )


def focus_mesh_region(
    image_rgb: np.ndarray,
    *,
    pad_ratio: float = 0.14,
    min_pad_px: int = 28,
) -> np.ndarray:
    image = np.array(image_rgb, copy=True)
    active_mask = np.any(image > 16, axis=2)
    if not np.any(active_mask):
        return image

    closed_mask = ndi.binary_closing(active_mask, structure=np.ones((3, 3), dtype=bool))
    labels, num_labels = ndi.label(closed_mask)
    if num_labels <= 0:
        return image

    component_areas = np.bincount(labels.ravel())[1:]
    if component_areas.size == 0:
        return image

    keep_label = int(np.argmax(component_areas)) + 1
    ys, xs = np.where(labels == keep_label)
    if xs.size == 0 or ys.size == 0:
        return image

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    pad = max(min_pad_px, int(round(max(x1 - x0, y1 - y0) * pad_ratio)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.shape[1], x1 + pad)
    y1 = min(image.shape[0], y1 + pad)

    focused = np.zeros_like(image)
    focused[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    return focused


def _detect_axis_calibration_raw(image_rgb: np.ndarray) -> AxisCalibration:
    mask = magenta_axis_mask(image_rgb)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Failed to detect the magenta axes in the input image.")

    band_radius = 3
    x_counts = np.bincount(xs)
    y_counts = np.bincount(ys)
    origin_x = int(np.argmax(x_counts))
    origin_y = int(np.argmax(y_counts))

    horizontal_slice = mask[max(0, origin_y - band_radius) : origin_y + band_radius + 1, :]
    vertical_slice = mask[:, max(0, origin_x - band_radius) : origin_x + band_radius + 1]

    horizontal_xs = np.where(horizontal_slice)[1]
    vertical_ys = np.where(vertical_slice)[0]
    if horizontal_xs.size == 0 or vertical_ys.size == 0:
        raise ValueError("Detected magenta pixels, but could not recover axis extents.")

    horizontal_points = np.column_stack(np.where(horizontal_slice)[::-1]).astype(float)
    horizontal_points[:, 1] += max(0, origin_y - band_radius)
    vertical_points = np.column_stack(np.where(vertical_slice)[::-1]).astype(float)
    vertical_points[:, 0] += max(0, origin_x - band_radius)

    horizontal_line = fit_line_tls(horizontal_points)
    vertical_line = fit_line_tls(vertical_points)
    origin = intersect_lines(horizontal_line, vertical_line)
    origin_x = float(origin[0])
    origin_y = float(origin[1])

    horizontal_direction = horizontal_line[1]
    if horizontal_direction[0] < 0:
        horizontal_direction = -horizontal_direction
    vertical_direction = vertical_line[1]
    if vertical_direction[1] > 0:
        vertical_direction = -vertical_direction

    full_points = np.column_stack([xs.astype(float), ys.astype(float)])
    horizontal_residual = np.abs(
        horizontal_direction[0] * (full_points[:, 1] - horizontal_line[0][1])
        - horizontal_direction[1] * (full_points[:, 0] - horizontal_line[0][0])
    )
    vertical_residual = np.abs(
        vertical_direction[0] * (full_points[:, 1] - vertical_line[0][1])
        - vertical_direction[1] * (full_points[:, 0] - vertical_line[0][0])
    )
    horizontal_mask = horizontal_residual <= max(3.5, band_radius + 0.5)
    vertical_mask = vertical_residual <= max(3.5, band_radius + 0.5)

    horizontal_support = full_points[horizontal_mask]
    vertical_support = full_points[vertical_mask]
    if len(horizontal_support) == 0 or len(vertical_support) == 0:
        raise ValueError("Axis support points were empty after line fitting.")

    horizontal_projection = (horizontal_support - origin) @ horizontal_direction
    vertical_projection = (vertical_support - origin) @ vertical_direction
    x_axis_end = float(origin_x + horizontal_direction[0] * horizontal_projection.max())
    y_axis_top = float(origin_y + vertical_direction[1] * vertical_projection.max())
    return AxisCalibration(
        origin_x=origin_x,
        origin_y=origin_y,
        x_axis_end=x_axis_end,
        y_axis_top=y_axis_top,
    )


def detect_axis_calibration_details(image_rgb: np.ndarray) -> AxisCalibrationDiagnostics:
    raw = _detect_axis_calibration_raw(image_rgb)
    stabilized, applied, reason = _stabilize_axis_calibration(raw, image_rgb.shape[:2])
    return AxisCalibrationDiagnostics(
        raw=raw,
        stabilized=stabilized,
        stabilization_applied=applied,
        stabilization_reason=reason,
    )


def detect_axis_calibration(image_rgb: np.ndarray) -> AxisCalibration:
    return detect_axis_calibration_details(image_rgb).stabilized


def _stabilize_axis_calibration(
    calibration: AxisCalibration,
    image_shape: tuple[int, int],
) -> tuple[AxisCalibration, bool, str | None]:
    height, width = image_shape
    template_width = _NORMALIZED_AXIS_TEMPLATE["width"]
    template_height = _NORMALIZED_AXIS_TEMPLATE["height"]

    expected_origin_x = width * (_NORMALIZED_AXIS_TEMPLATE["origin_x"] / template_width)
    expected_origin_y = height * (_NORMALIZED_AXIS_TEMPLATE["origin_y"] / template_height)
    expected_x_axis_end = width * (_NORMALIZED_AXIS_TEMPLATE["x_axis_end"] / template_width)
    expected_y_axis_top = height * (_NORMALIZED_AXIS_TEMPLATE["y_axis_top"] / template_height)

    expected_x_span = expected_x_axis_end - expected_origin_x
    expected_y_span = expected_origin_y - expected_y_axis_top
    actual_x_span = calibration.x_axis_end - calibration.origin_x
    actual_y_span = calibration.origin_y - calibration.y_axis_top

    origin_tol_x = max(4.0, width * 0.004)
    origin_tol_y = max(4.0, height * 0.004)
    span_tol_x = max(12.0, width * 0.008)
    span_tol_y = max(8.0, height * 0.008)

    origin_close = (
        abs(calibration.origin_x - expected_origin_x) <= origin_tol_x
        and abs(calibration.origin_y - expected_origin_y) <= origin_tol_y
    )
    x_span_outlier = abs(actual_x_span - expected_x_span) > span_tol_x
    y_span_outlier = abs(actual_y_span - expected_y_span) > span_tol_y

    if not origin_close:
        return calibration, False, None

    if not x_span_outlier and not y_span_outlier:
        return calibration, False, None

    reason_parts: list[str] = []
    if x_span_outlier:
        reason_parts.append("x_span_outlier")
    if y_span_outlier:
        reason_parts.append("y_span_outlier")

    return AxisCalibration(
        origin_x=float(expected_origin_x),
        origin_y=float(expected_origin_y),
        x_axis_end=float(expected_x_axis_end if x_span_outlier else calibration.x_axis_end),
        y_axis_top=float(expected_y_axis_top if y_span_outlier else calibration.y_axis_top),
        x_extent=calibration.x_extent,
        y_extent=calibration.y_extent,
    ), True, ",".join(reason_parts)


def strip_axis_artifacts(image_rgb: np.ndarray, calibration: AxisCalibration) -> np.ndarray:
    image = np.array(image_rgb, copy=True)
    mask = magenta_axis_mask(image)

    x0 = int(round(calibration.origin_x))
    y0 = int(round(calibration.origin_y))
    x1 = int(round(calibration.x_axis_end))
    y_top = int(round(calibration.y_axis_top))

    mask[:, max(0, x0 - 18) : min(image.shape[1], x0 + 18)] = True
    mask[max(0, y0 - 18) : min(image.shape[0], y0 + 18), :] = True

    # Remove the on-axis label boxes. This is intentionally conservative:
    # the FE mesh is centered far away from the axes in the normalized images.
    mask[
        max(0, y_top + 220) : min(image.shape[0], y0 - 220),
        max(0, x0 - 8) : min(image.shape[1], x0 + 90),
    ] = True
    mask[
        max(0, y0 - 40) : min(image.shape[0], y0 + 35),
        max(0, x0 + 220) : min(image.shape[1], x1 - 220),
    ] = True

    image[mask] = 0
    return image


def save_axis_debug(
    image_rgb: np.ndarray,
    calibration: AxisCalibration,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    origin = (calibration.origin_x, calibration.origin_y)
    x_end = (calibration.x_axis_end, calibration.origin_y)
    y_end = (calibration.origin_x, calibration.y_axis_top)

    draw.line([origin, x_end], fill=(0, 255, 255), width=3)
    draw.line([origin, y_end], fill=(0, 255, 255), width=3)
    draw.ellipse(
        (
            calibration.origin_x - 6,
            calibration.origin_y - 6,
            calibration.origin_x + 6,
            calibration.origin_y + 6,
        ),
        outline=(255, 255, 255),
        width=2,
    )
    draw.text((calibration.origin_x + 20, calibration.origin_y - 30), "origin", fill=(255, 255, 255))
    draw.text((calibration.x_axis_end - 130, calibration.origin_y - 32), "x=100", fill=(255, 255, 255))
    draw.text((calibration.origin_x + 12, calibration.y_axis_top + 10), "y=75", fill=(255, 255, 255))
    image.save(save_path)
