"""Pick a CBCT patch interactively and denoise it with a fresh Filter2Noise model."""

from __future__ import annotations

import numpy as np


def snap_patch(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    height: int,
    width: int,
    multiple: int = 8,
    min_size: int = 32,
) -> tuple[int, int, int, int]:
    """Turn a dragged rectangle into a valid ``(x, y, w, h)`` patch."""
    if height < multiple or width < multiple:
        raise ValueError(
            f"Image {width}x{height} is smaller than the {multiple}-pixel patch grid."
        )

    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    xa = min(max(xa, 0.0), float(width))
    xb = min(max(xb, 0.0), float(width))
    ya = min(max(ya, 0.0), float(height))
    yb = min(max(yb, 0.0), float(height))

    w = max(min_size, int(round((xb - xa) / multiple)) * multiple)
    h = max(min_size, int(round((yb - ya) / multiple)) * multiple)
    w = min(w, (width // multiple) * multiple)
    h = min(h, (height // multiple) * multiple)

    cx = (xa + xb) / 2.0
    cy = (ya + yb) / 2.0
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    x = min(max(x, 0), width - w)
    y = min(max(y, 0), height - h)
    return x, y, w, h


def hu_to_unit(image_hu: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Clip and normalise HU values to ``[0, 1]`` over ``window``."""
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        raise ValueError(f"Invalid HU window: {window}")
    scaled = (image_hu.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def unit_to_hu(image_unit: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Convert ``[0, 1]`` values back to HU over ``window``."""
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        raise ValueError(f"Invalid HU window: {window}")
    return (image_unit.astype(np.float32) * (hi - lo) + lo).astype(np.float32)
