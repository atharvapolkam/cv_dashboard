"""Thresholding: threshold, adaptiveThreshold, inRange (+ the HSV mask tester)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ops.constants import (
    ADAPTIVE_METHODS, ADAPTIVE_THRESH_TYPES, THRESH_AUTO, THRESH_TYPES, cv_const,
)
from app.ops.helpers import (
    as_bgr, as_gray, as_u8, cvc, ensure_odd, fmt_num, nonzero_ratio, py_list, require_gray,
    shape_summary,
)
from app.ops.params import Choice, ParamSpec, ParamType, p_bool, p_enum, p_float, p_int, p_odd
from app.ops.registry import Category, Cost, OpContext, OpResult, OutputKind, register


@register(
    id="threshold",
    cv="cv2.threshold",
    label="Threshold",
    category=Category.THRESHOLD,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    output=OutputKind.MASK,
    tags=["threshold", "binary", "otsu", "triangle", "mask"],
    summary="Global threshold with optional Otsu/Triangle auto-selection.",
    doc="d7/d1b/group__imgproc__misc.html",
    params=[
        p_int("thresh", "Threshold", 127, 0, 255),
        p_int("maxval", "Max value", 255, 0, 255),
        p_enum("type", "Type", "THRESH_BINARY", THRESH_TYPES),
        p_enum("auto", "Auto method", "none", THRESH_AUTO,
               help="Otsu/Triangle compute the threshold from the histogram and ignore the "
                    "slider."),
    ],
)
def _threshold(ctx: OpContext) -> OpResult:
    gray = as_gray(as_u8(ctx.image))
    t_const = ctx.p["type"]
    auto = ctx.p["auto"]
    flags = cv_const(t_const)
    code_flags = cvc(t_const)
    if auto != "none":
        flags |= cv_const(auto)
        code_flags += f" + {cvc(auto)}"
    thresh_in = 0 if auto != "none" else int(ctx.p["thresh"])
    used, out = cv2.threshold(gray, thresh_in, int(ctx.p["maxval"]), flags)
    pre = ["gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"] if ctx.channels != 1 else ["gray = img"]
    return OpResult(
        image=out,
        code=[*pre,
              f"_, img = cv2.threshold(gray, {thresh_in}, {int(ctx.p['maxval'])}, {code_flags})"],
        data={"threshold_used": round(float(used), 3), **nonzero_ratio(out)},
        summary={
            "Threshold used": f"{used:.1f}" + (f"  (auto: {auto})" if auto != "none" else ""),
            "Type": t_const,
            "Output": shape_summary(out),
            "White pixels": f"{nonzero_ratio(out)['coverage_pct']:.2f} %",
        },
        notes=[f"{auto} picked {used:.0f}; the slider value was ignored."]
        if auto != "none" else [],
    )


@register(
    id="adaptive_threshold",
    cv="cv2.adaptiveThreshold",
    label="Adaptive threshold",
    category=Category.THRESHOLD,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    output=OutputKind.MASK,
    tags=["adaptive", "threshold", "local", "gaussian", "mean", "uneven lighting"],
    summary="Per-neighbourhood threshold — the fix for uneven lighting across a frame.",
    params=[
        p_int("maxval", "Max value", 255, 0, 255),
        p_enum("method", "Method", "ADAPTIVE_THRESH_GAUSSIAN_C", ADAPTIVE_METHODS),
        p_enum("type", "Type", "THRESH_BINARY", ADAPTIVE_THRESH_TYPES),
        p_odd("block_size", "blockSize", 11, 3, 199,
              help="Neighbourhood size. Must be odd and > 1."),
        p_float("c", "C (subtracted)", 2.0, -50.0, 50.0, 0.5),
    ],
)
def _adaptive(ctx: OpContext) -> OpResult:
    gray = as_gray(as_u8(ctx.image))
    block = ensure_odd(int(ctx.p["block_size"]), 3)
    c = float(ctx.p["c"])
    out = cv2.adaptiveThreshold(
        gray, int(ctx.p["maxval"]), cv_const(ctx.p["method"]), cv_const(ctx.p["type"]), block, c
    )
    pre = ["gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"] if ctx.channels != 1 else ["gray = img"]
    return OpResult(
        image=out,
        code=[*pre,
              "img = cv2.adaptiveThreshold(",
              f"    gray, {int(ctx.p['maxval'])}, {cvc(ctx.p['method'])},",
              f"    {cvc(ctx.p['type'])}, {block}, {fmt_num(c)}",
              ")"],
        data=nonzero_ratio(out),
        summary={"Method": ctx.p["method"], "blockSize": block, "C": c,
                 "White pixels": f"{nonzero_ratio(out)['coverage_pct']:.2f} %"},
    )


# ------------------------------------------------------------------- inRange
_RANGE_SPACE = (
    Choice("hsv", "HSV (recommended for colour)", None, "Converts BGR → HSV first."),
    Choice("bgr", "BGR (as-is)"),
    Choice("lab", "LAB"),
    Choice("gray", "Grayscale (single band)"),
)

# Common starting points. These are HSV (OpenCV scale) ranges that work on
# typical camera frames; the user tunes from there.
HSV_PRESETS: dict[str, dict[str, object]] = {
    "red_low":    {"label": "Red (low hue)",  "lower": [0, 120, 70],   "upper": [10, 255, 255]},
    "red_high":   {"label": "Red (high hue)", "lower": [170, 120, 70], "upper": [179, 255, 255]},
    "orange":     {"label": "Orange",         "lower": [10, 120, 100], "upper": [22, 255, 255]},
    "yellow":     {"label": "Yellow",         "lower": [22, 100, 100], "upper": [35, 255, 255]},
    "green":      {"label": "Green",          "lower": [36, 80, 60],   "upper": [86, 255, 255]},
    "cyan":       {"label": "Cyan",           "lower": [86, 80, 60],   "upper": [100, 255, 255]},
    "blue":       {"label": "Blue",           "lower": [100, 120, 60], "upper": [130, 255, 255]},
    "purple":     {"label": "Purple",         "lower": [130, 60, 60],  "upper": [160, 255, 255]},
    "white":      {"label": "White / bright", "lower": [0, 0, 200],    "upper": [179, 40, 255]},
    "black":      {"label": "Black / dark",   "lower": [0, 0, 0],      "upper": [179, 255, 55]},
    "gray":       {"label": "Gray (low sat)", "lower": [0, 0, 60],     "upper": [179, 40, 200]},
    "skin":       {"label": "Skin tone",      "lower": [0, 30, 60],    "upper": [25, 170, 255]},
    "hi_vis":     {"label": "Hi-vis vest",    "lower": [15, 120, 120], "upper": [40, 255, 255]},
}


@register(
    id="in_range",
    cv="cv2.inRange",
    label="InRange (colour mask)",
    category=Category.THRESHOLD,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    output=OutputKind.MASK,
    tags=["inrange", "hsv", "mask", "colour", "color", "detect", "range", "hue"],
    summary="The colour-mask workhorse. Returns the mask plus the masked result and the "
            "HSV source so you can see all three at once.",
    params=[
        p_enum("space", "Colour space", "hsv", _RANGE_SPACE),
        p_int("l0", "H min", 35, 0, 255, group="Lower"),
        p_int("l1", "S min", 80, 0, 255, group="Lower"),
        p_int("l2", "V min", 60, 0, 255, group="Lower"),
        p_int("u0", "H max", 85, 0, 255, group="Upper"),
        p_int("u1", "S max", 255, 0, 255, group="Upper"),
        p_int("u2", "V max", 255, 0, 255, group="Upper"),
        p_bool("wrap_hue", "Wrap hue (red)", False,
               help="When H min > H max, OR two ranges together — needed for red."),
        p_bool("apply_mask", "Output masked image", False,
               help="Off: the mask becomes the working image. On: the colour image masked."),
    ],
)
def _in_range(ctx: OpContext) -> OpResult:
    space = ctx.p["space"]
    bgr = as_bgr(as_u8(ctx.image))
    if space == "hsv":
        conv, cconst = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), "COLOR_BGR2HSV"
        labels = ("H", "S", "V")
    elif space == "lab":
        conv, cconst = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB), "COLOR_BGR2LAB"
        labels = ("L", "A", "B")
    elif space == "gray":
        conv, cconst = as_gray(bgr), "COLOR_BGR2GRAY"
        labels = ("I", "-", "-")
    else:
        conv, cconst = bgr, None
        labels = ("B", "G", "R")

    lo = [int(ctx.p["l0"]), int(ctx.p["l1"]), int(ctx.p["l2"])]
    hi = [int(ctx.p["u0"]), int(ctx.p["u1"]), int(ctx.p["u2"])]

    code: list[str] = []
    if cconst:
        code.append(f"src = cv2.cvtColor(img, {cvc(cconst)})")
    else:
        code.append("src = img")

    if space == "gray":
        mask = cv2.inRange(conv, lo[0], hi[0])
        code.append(f"mask = cv2.inRange(src, {lo[0]}, {hi[0]})")
    elif ctx.p["wrap_hue"] and lo[0] > hi[0]:
        a = cv2.inRange(conv, np.array([lo[0], lo[1], lo[2]]), np.array([179, hi[1], hi[2]]))
        b = cv2.inRange(conv, np.array([0, lo[1], lo[2]]), np.array([hi[0], hi[1], hi[2]]))
        mask = cv2.bitwise_or(a, b)
        code += [
            f"lower_a = np.array([{lo[0]}, {lo[1]}, {lo[2]}])",
            f"upper_a = np.array([179, {hi[1]}, {hi[2]}])",
            f"lower_b = np.array([0, {lo[1]}, {lo[2]}])",
            f"upper_b = np.array([{hi[0]}, {hi[1]}, {hi[2]}])",
            "mask = cv2.bitwise_or(cv2.inRange(src, lower_a, upper_a),",
            "                      cv2.inRange(src, lower_b, upper_b))",
        ]
    else:
        mask = cv2.inRange(conv, np.array(lo), np.array(hi))
        code += [
            f"lower = np.array({py_list(lo)})",
            f"upper = np.array({py_list(hi)})",
            "mask = cv2.inRange(src, lower, upper)",
        ]

    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    if ctx.p["apply_mask"]:
        out = masked
        code.append("img = cv2.bitwise_and(img, img, mask=mask)")
    else:
        out = mask
        code.append("img = mask")

    ratio = nonzero_ratio(mask)
    extras = {"Mask": mask, "Masked result": masked}
    if space == "hsv":
        extras["HSV source"] = conv
    return OpResult(
        image=out,
        extras=extras,
        code=code,
        imports=["import numpy as np"],
        data={"lower": lo, "upper": hi, "space": space, **ratio,
              "labels": list(labels)},
        summary={
            "Space": space.upper(),
            "Lower": f"[{lo[0]}, {lo[1]}, {lo[2]}]",
            "Upper": f"[{hi[0]}, {hi[1]}, {hi[2]}]",
            "Mask coverage": f"{ratio['coverage_pct']:.2f} %  ({ratio['nonzero']:,} px)",
        },
        notes=(["Mask is empty — widen the range or lower S/V minimums."]
               if ratio["nonzero"] == 0 else
               ["Mask covers almost the whole frame — tighten the range."]
               if ratio["coverage_pct"] > 95 else []),
        carry={"mask": mask},
    )


@register(
    id="normalize",
    cv="cv2.normalize",
    label="Normalize",
    category=Category.THRESHOLD,
    tier=3,
    cost=Cost.CHEAP,
    tags=["normalize", "minmax", "stretch", "contrast"],
    summary="Stretch values to a range — makes faint gradients visible.",
    params=[
        p_float("alpha", "Min (alpha)", 0.0, -1000.0, 1000.0, 1.0),
        p_float("beta", "Max (beta)", 255.0, -1000.0, 1000.0, 1.0),
    ],
)
def _normalize(ctx: OpContext) -> OpResult:
    out = cv2.normalize(ctx.image.astype(np.float32), None, float(ctx.p["alpha"]),
                        float(ctx.p["beta"]), cv2.NORM_MINMAX)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return OpResult(
        image=out_u8,
        code=[f"img = cv2.normalize(img.astype(np.float32), None, {fmt_num(ctx.p['alpha'])}, "
              f"{fmt_num(ctx.p['beta'])}, cv2.NORM_MINMAX)",
              "img = np.clip(img, 0, 255).astype(np.uint8)"],
        imports=["import numpy as np"],
        summary={"Range": f"{ctx.p['alpha']} → {ctx.p['beta']}"},
    )
