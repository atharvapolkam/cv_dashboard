"""Shared helpers for catalog executors.

Also the single place that absorbs OpenCV-version differences. OpenCV 5 changed
several return shapes relative to 4.x (``HoughLinesP`` -> (N,4) instead of
(N,1,4), ``calcHist`` -> (256,) instead of (256,1), ``convexityDefects`` ->
(N,4)). Every catalog module goes through the normalisers below so the rest of
the app never branches on cv2.__version__.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from app.core.errors import OperationError
from app.ops.constants import cv_const


# --------------------------------------------------------------------- guards
def as_gray(img: np.ndarray) -> np.ndarray:
    """Single-channel view of any input, without mutating the caller's array."""
    if img.ndim == 2:
        return img
    ch = img.shape[2]
    if ch == 1:
        return img[:, :, 0]
    if ch == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def as_bgr(img: np.ndarray) -> np.ndarray:
    """3-channel BGR view — used for drawing overlays on masks."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def as_u8(img: np.ndarray) -> np.ndarray:
    """uint8 version, scaling floats sensibly (0-1 floats are common)."""
    if img.dtype == np.uint8:
        return img
    a = img.astype(np.float64)
    if np.issubdtype(img.dtype, np.floating):
        hi = float(np.nanmax(a)) if a.size else 1.0
        if hi <= 1.0 + 1e-6:
            a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def require_gray(img: np.ndarray, fn: str) -> np.ndarray:
    if img.ndim == 2 or img.shape[2] == 1:
        return as_gray(img)
    raise OperationError(
        f"{fn} needs a single-channel image, got {img.shape[2]} channels.",
        hint="Run 'BGR → Gray' (or a threshold) first. Tip: the Edge Detection "
             "pipeline button does Gray → Blur → Canny in one click.",
    )


def require_binary(img: np.ndarray, fn: str) -> np.ndarray:
    g = require_gray(img, fn)
    uniq = np.unique(g[:: max(1, g.shape[0] // 64), :: max(1, g.shape[1] // 64)])
    if uniq.size > 2:
        raise OperationError(
            f"{fn} expects a binary image (only 0 and 255).",
            hint="Apply Threshold, Adaptive Threshold or InRange first.",
        )
    return g


def ensure_odd(v: int, lo: int = 1) -> int:
    v = int(v)
    if v < lo:
        v = lo
    return v if v % 2 == 1 else v + 1


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def same_size(a: np.ndarray, b: np.ndarray, fn: str) -> np.ndarray:
    """Resize+match channels of ``b`` onto ``a`` so two-input ops always run."""
    if b is None:
        raise OperationError(f"{fn} needs a second image.", hint="Pick one in 'Second image'.")
    out = b
    if out.shape[:2] != a.shape[:2]:
        out = cv2.resize(out, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if a.ndim == 3 and out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if a.ndim == 2 and out.ndim == 3:
        out = as_gray(out)
    if a.ndim == 3 and out.ndim == 3 and a.shape[2] != out.shape[2]:
        out = as_bgr(out) if a.shape[2] == 3 else cv2.cvtColor(as_bgr(out), cv2.COLOR_BGR2BGRA)
    if out.dtype != a.dtype:
        out = out.astype(a.dtype)
    return out


# ------------------------------------------------------- shape normalisation
def norm_hough_lines_p(res: Any) -> list[list[int]]:
    """-> [[x1,y1,x2,y2], ...] for both (N,4) [cv5] and (N,1,4) [cv4]."""
    if res is None:
        return []
    arr = np.asarray(res)
    arr = arr.reshape(-1, 4)
    return [[int(v) for v in row] for row in arr]


def norm_hough_lines(res: Any) -> list[list[float]]:
    """-> [[rho, theta], ...]"""
    if res is None:
        return []
    arr = np.asarray(res).reshape(-1, 2)
    return [[float(r), float(t)] for r, t in arr]


def norm_hough_circles(res: Any) -> list[list[float]]:
    """-> [[x, y, r], ...]"""
    if res is None:
        return []
    arr = np.asarray(res).reshape(-1, 3)
    return [[float(x), float(y), float(r)] for x, y, r in arr]


def norm_hist(res: Any) -> list[float]:
    """calcHist gives (bins,) on cv5 and (bins,1) on cv4."""
    return [float(v) for v in np.asarray(res).reshape(-1)]


def norm_defects(res: Any) -> list[list[int]]:
    """convexityDefects gives (N,4) on cv5 and (N,1,4) on cv4."""
    if res is None:
        return []
    return [[int(v) for v in row] for row in np.asarray(res).reshape(-1, 4)]


def norm_contours(res: Any) -> tuple[list[np.ndarray], np.ndarray | None]:
    """findContours returns 2 values on cv4/cv5 and 3 on cv3."""
    if isinstance(res, tuple) and len(res) == 3:
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res
    return list(contours), hierarchy


def norm_points(res: Any) -> list[list[float]]:
    """goodFeaturesToTrack -> [[x, y], ...]"""
    if res is None:
        return []
    return [[float(x), float(y)] for x, y in np.asarray(res).reshape(-1, 2)]


# ------------------------------------------------------------ contour measure
def measure_contour(cnt: np.ndarray, index: int) -> dict[str, Any]:
    """One row of the contour results table."""
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    x, y, w, h = (int(v) for v in cv2.boundingRect(cnt))
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-9:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    (mc_x, mc_y), radius = cv2.minEnclosingCircle(cnt)
    (rc_x, rc_y), (rw, rh), angle = cv2.minAreaRect(cnt)
    return {
        "id": index,
        "area": round(area, 2),
        "perimeter": round(peri, 2),
        "x": x, "y": y, "w": w, "h": h,
        "x2": x + w, "y2": y + h,
        "cx": round(cx, 2), "cy": round(cy, 2),
        "points": int(len(cnt)),
        "convex": bool(cv2.isContourConvex(cnt)),
        "aspect": round(w / h, 4) if h else 0.0,
        "extent": round(area / (w * h), 4) if w * h else 0.0,
        "solidity": round(area / hull_area, 4) if hull_area > 1e-9 else 0.0,
        "hull_points": int(len(hull)),
        "min_circle": {"cx": round(float(mc_x), 2), "cy": round(float(mc_y), 2),
                       "r": round(float(radius), 2)},
        "min_rect": {"cx": round(float(rc_x), 2), "cy": round(float(rc_y), 2),
                     "w": round(float(rw), 2), "h": round(float(rh), 2),
                     "angle": round(float(angle), 2)},
    }


def contour_to_list(cnt: np.ndarray, limit: int = 400) -> list[list[int]]:
    """Flatten a contour to [[x,y],...], subsampled so JSON stays small."""
    pts = np.asarray(cnt).reshape(-1, 2)
    if len(pts) > limit:
        idx = np.linspace(0, len(pts) - 1, limit).astype(int)
        pts = pts[idx]
    return [[int(x), int(y)] for x, y in pts]


def structuring_element(shape_const: str, kw: int, kh: int) -> np.ndarray:
    return cv2.getStructuringElement(cv_const(shape_const), (max(1, int(kw)), max(1, int(kh))))


def kernel_preview(kernel: np.ndarray) -> list[list[int]]:
    return [[int(v) for v in row] for row in np.asarray(kernel)]


# ------------------------------------------------------------------- codegen
def py_tuple(vals: Iterable[Any]) -> str:
    items = list(vals)
    inner = ", ".join(fmt_num(v) for v in items)
    return f"({inner})" if len(items) != 1 else f"({inner},)"


def py_list(vals: Iterable[Any]) -> str:
    return "[" + ", ".join(fmt_num(v) for v in vals) + "]"


def fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f.is_integer() and abs(f) < 1e15:
            return f"{f:.1f}"
        return f"{f:g}"
    if isinstance(v, str):
        return v
    return repr(v)


def py_points(points: Sequence[Sequence[float]]) -> str:
    return "[" + ", ".join(f"[{int(p[0])}, {int(p[1])}]" for p in points) + "]"


def cvc(name: str) -> str:
    """Constant name -> generated-code token."""
    return f"cv2.{name}"


def image_stats(img: np.ndarray) -> dict[str, Any]:
    """Cheap min/max/mean/std used by the What-happened panel."""
    a = img if img.size < 4_000_000 else img[::2, ::2]
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": round(float(np.mean(a)), 3),
        "std": round(float(np.std(a)), 3),
    }


def shape_summary(img: np.ndarray) -> str:
    if img.ndim == 2:
        return f"{img.shape[1]} × {img.shape[0]} × 1 ({img.dtype})"
    return f"{img.shape[1]} × {img.shape[0]} × {img.shape[2]} ({img.dtype})"


def nonzero_ratio(mask: np.ndarray) -> dict[str, Any]:
    g = as_gray(mask)
    nz = int(cv2.countNonZero(g))
    total = int(g.size)
    return {"nonzero": nz, "total": total,
            "coverage_pct": round(100.0 * nz / total, 3) if total else 0.0}
