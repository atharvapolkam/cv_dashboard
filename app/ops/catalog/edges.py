"""Edge detection: Canny, Sobel, Scharr, Laplacian."""

from __future__ import annotations

import cv2
import numpy as np

from app.ops.constants import DDEPTHS, cv_const
from app.ops.helpers import (
    as_gray, as_u8, cvc, ensure_odd, fmt_num, nonzero_ratio, shape_summary,
)
from app.ops.params import Choice, p_bool, p_enum, p_float, p_int, p_odd
from app.ops.registry import Category, Cost, OpContext, OpResult, OutputKind, register


@register(
    id="canny", cv="cv2.Canny", label="Canny", category=Category.EDGES, tier=1,
    cost=Cost.MEDIUM, quick=True, output=OutputKind.MASK,
    tags=["canny", "edge", "edges", "gradient", "outline"],
    summary="Canny edges. Rule of thumb: threshold2 ≈ 2–3 × threshold1.",
    doc="dd/d1a/group__imgproc__feature.html",
    params=[
        p_int("threshold1", "threshold1 (low)", 50, 0, 500),
        p_int("threshold2", "threshold2 (high)", 150, 0, 1000),
        p_odd("aperture_size", "apertureSize", 3, 3, 7),
        p_bool("l2_gradient", "L2gradient", False,
               help="More accurate gradient magnitude, slightly slower."),
        p_bool("auto_gray", "Auto-convert to gray", True,
               help="Off: raises an error if the input is not single-channel."),
    ],
)
def _canny(ctx: OpContext) -> OpResult:
    t1, t2 = int(ctx.p["threshold1"]), int(ctx.p["threshold2"])
    ap = ensure_odd(int(ctx.p["aperture_size"]), 3)
    l2 = bool(ctx.p["l2_gradient"])
    pre: list[str] = []
    src = ctx.image
    if src.ndim != 2:
        if not ctx.p["auto_gray"]:
            from app.core.errors import OperationError
            raise OperationError("Canny needs a single-channel image.",
                                 hint="Enable 'Auto-convert to gray' or run BGR → Gray first.")
        src = as_gray(src)
        pre.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
    else:
        pre.append("gray = img")
    src = as_u8(src)
    out = cv2.Canny(src, t1, t2, apertureSize=ap, L2gradient=l2)
    extra = "".join([f", apertureSize={ap}" if ap != 3 else "",
                     ", L2gradient=True" if l2 else ""])
    nz = nonzero_ratio(out)
    return OpResult(
        image=out,
        code=[*pre, f"img = cv2.Canny(gray, {t1}, {t2}{extra})"],
        data={"threshold1": t1, "threshold2": t2, **nz},
        summary={"Input": shape_summary(ctx.image), "Output": shape_summary(out),
                 "Thresholds": f"{t1} / {t2}  (ratio {t2/max(1,t1):.2f})",
                 "Edge pixels": f"{nz['nonzero']:,}  ({nz['coverage_pct']:.2f} %)"},
        notes=(["No edges found — lower threshold1."] if nz["nonzero"] == 0 else
               ["Very noisy: >20 % of pixels are edges. Blur first or raise thresholds."]
               if nz["coverage_pct"] > 20 else []),
    )


def _grad(name: str, op_id: str, label: str, extra_params, fn, tier=2):
    @register(id=op_id, cv=f"cv2.{name}", label=label, category=Category.EDGES, tier=tier,
              cost=Cost.MEDIUM, params=extra_params,
              tags=["gradient", "edge", name.lower(), "derivative"],
              summary=f"cv2.{name} gradient, rendered via convertScaleAbs so negatives stay "
                      f"visible.")
    def _run(ctx: OpContext) -> OpResult:
        gray = as_u8(as_gray(ctx.image))
        pre = ["gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"] if ctx.channels != 1 else \
              ["gray = img"]
        raw, code = fn(ctx, gray)
        vis = cv2.convertScaleAbs(raw)
        return OpResult(
            image=vis,
            code=[*pre, *code, "img = cv2.convertScaleAbs(grad)"],
            data={"raw_min": round(float(raw.min()), 3), "raw_max": round(float(raw.max()), 3)},
            summary={"Function": f"cv2.{name}", "Raw range":
                     f"{raw.min():.1f} … {raw.max():.1f}", "Output": shape_summary(vis)},
        )
    return _run


def _sobel_fn(ctx: OpContext, gray: np.ndarray):
    dx, dy = int(ctx.p["dx"]), int(ctx.p["dy"])
    if dx == 0 and dy == 0:
        dx = 1
    k = ensure_odd(int(ctx.p["ksize"]), 1)
    dd = ctx.p["ddepth"]
    scale, delta = float(ctx.p["scale"]), float(ctx.p["delta"])
    raw = cv2.Sobel(gray, cv_const(dd), dx, dy, ksize=k, scale=scale, delta=delta)
    return raw, [f"grad = cv2.Sobel(gray, {cvc(dd)}, {dx}, {dy}, ksize={k}"
                 + (f", scale={fmt_num(scale)}" if scale != 1 else "")
                 + (f", delta={fmt_num(delta)}" if delta else "") + ")"]


_grad("Sobel", "sobel", "Sobel", [
    p_int("dx", "dx", 1, 0, 2), p_int("dy", "dy", 0, 0, 2),
    p_odd("ksize", "ksize", 3, 1, 31), p_enum("ddepth", "ddepth", "CV_64F", DDEPTHS),
    p_float("scale", "scale", 1.0, 0.01, 20.0, 0.1, advanced=True),
    p_float("delta", "delta", 0.0, -255.0, 255.0, 1.0, advanced=True),
], _sobel_fn)


def _scharr_fn(ctx: OpContext, gray: np.ndarray):
    axis = ctx.p["axis"]
    dx, dy = (1, 0) if axis == "x" else (0, 1)
    dd = ctx.p["ddepth"]
    raw = cv2.Scharr(gray, cv_const(dd), dx, dy)
    return raw, [f"grad = cv2.Scharr(gray, {cvc(dd)}, {dx}, {dy})"]


_grad("Scharr", "scharr", "Scharr", [
    p_enum("axis", "Axis", "x", (Choice("x", "X (dx=1, dy=0)"), Choice("y", "Y (dx=0, dy=1)"))),
    p_enum("ddepth", "ddepth", "CV_64F", DDEPTHS),
], _scharr_fn)


def _lap_fn(ctx: OpContext, gray: np.ndarray):
    k = ensure_odd(int(ctx.p["ksize"]), 1)
    dd = ctx.p["ddepth"]
    raw = cv2.Laplacian(gray, cv_const(dd), ksize=k)
    return raw, [f"grad = cv2.Laplacian(gray, {cvc(dd)}, ksize={k})"]


_grad("Laplacian", "laplacian", "Laplacian", [
    p_odd("ksize", "ksize", 3, 1, 31), p_enum("ddepth", "ddepth", "CV_64F", DDEPTHS),
], _lap_fn)


@register(
    id="sobel_magnitude", cv="cv2.Sobel + cv2.magnitude", label="Gradient magnitude",
    category=Category.EDGES, tier=2, cost=Cost.MEDIUM,
    params=[p_odd("ksize", "ksize", 3, 1, 31)],
    tags=["gradient", "magnitude", "sobel", "edge strength"],
    summary="sqrt(gx² + gy²) — direction-independent edge strength.",
)
def _sobel_mag(ctx: OpContext) -> OpResult:
    gray = as_u8(as_gray(ctx.image))
    k = ensure_odd(int(ctx.p["ksize"]), 1)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    vis = cv2.convertScaleAbs(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))
    pre = ["gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"] if ctx.channels != 1 else ["gray = img"]
    return OpResult(
        image=vis,
        code=[*pre,
              f"gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={k})",
              f"gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={k})",
              "mag = cv2.magnitude(gx, gy)",
              "img = cv2.convertScaleAbs(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))"],
        summary={"ksize": k, "Magnitude max": round(float(mag.max()), 2),
                 "Output": shape_summary(vis)},
    )
