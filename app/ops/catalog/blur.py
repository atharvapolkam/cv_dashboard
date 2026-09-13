"""Blur, denoise and generic convolution."""

from __future__ import annotations

import cv2
import numpy as np

from app.ops.constants import BORDER_TYPES, cv_const
from app.ops.helpers import (
    as_u8, cvc, ensure_odd, fmt_num, image_stats, kernel_preview, py_tuple, shape_summary,
)
from app.ops.params import Choice, p_bool, p_enum, p_float, p_int, p_kernel, p_odd
from app.ops.registry import Category, Cost, OpContext, OpResult, register


@register(
    id="gaussian_blur",
    cv="cv2.GaussianBlur",
    label="Gaussian blur",
    category=Category.BLUR,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    tags=["gaussian", "blur", "smooth", "denoise", "preprocess"],
    summary="The standard pre-Canny / pre-threshold smoothing step.",
    doc="d4/d86/group__imgproc__filter.html",
    params=[
        p_kernel("ksize", "Kernel (w × h)", (5, 5), hi=99,
                 help="Must be odd. Larger = blurrier and slower."),
        p_float("sigma_x", "sigmaX", 0.0, 0.0, 50.0, 0.1,
                help="0 lets OpenCV derive sigma from the kernel size."),
        p_float("sigma_y", "sigmaY", 0.0, 0.0, 50.0, 0.1,
                help="0 means 'same as sigmaX'."),
        p_enum("border_type", "Border", "BORDER_DEFAULT",
               (Choice("BORDER_DEFAULT", "BORDER_DEFAULT", "BORDER_DEFAULT"), *BORDER_TYPES),
               advanced=True),
    ],
)
def _gaussian(ctx: OpContext) -> OpResult:
    kw = ensure_odd(int(ctx.p["ksize"][0]))
    kh = ensure_odd(int(ctx.p["ksize"][1]))
    sx, sy = float(ctx.p["sigma_x"]), float(ctx.p["sigma_y"])
    bt = ctx.p["border_type"]
    out = cv2.GaussianBlur(ctx.image, (kw, kh), sx, sigmaY=sy, borderType=cv_const(bt))
    extra = f",\n    borderType={cvc(bt)}" if bt != "BORDER_DEFAULT" else ""
    return OpResult(
        image=out,
        code=[
            "img = cv2.GaussianBlur(",
            "    img,",
            f"    ({kw}, {kh}),",
            f"    {fmt_num(sx)}" + (f",\n    sigmaY={fmt_num(sy)}" if sy else "") + extra,
            ")",
        ],
        data={"kernel": [kw, kh], "sigma_x": sx, "sigma_y": sy, "stats": image_stats(out)},
        summary={"Kernel": f"{kw} × {kh}", "sigmaX": sx or "auto", "sigmaY": sy or "= sigmaX",
                 "Output": shape_summary(out)},
    )


@register(
    id="median_blur",
    cv="cv2.medianBlur",
    label="Median blur",
    category=Category.BLUR,
    tier=1,
    cost=Cost.MEDIUM,
    quick=True,
    tags=["median", "blur", "salt", "pepper", "denoise", "speckle"],
    summary="Kills salt-and-pepper noise and mask speckle without blurring edges much.",
    params=[p_odd("ksize", "Kernel size", 5, 1, 99,
                  help="Single odd number. >5 uses a slower code path on large images.")],
)
def _median(ctx: OpContext) -> OpResult:
    k = ensure_odd(int(ctx.p["ksize"]))
    src = as_u8(ctx.image) if k > 5 else ctx.image
    out = cv2.medianBlur(src, k)
    return OpResult(
        image=out,
        code=[f"img = cv2.medianBlur(img, {k})"],
        data={"kernel": k, "stats": image_stats(out)},
        summary={"Kernel": f"{k} × {k}", "Output": shape_summary(out)},
        notes=["ksize > 5 requires uint8 input; the image was converted."]
        if k > 5 and ctx.image.dtype != np.uint8 else [],
    )


@register(
    id="blur",
    cv="cv2.blur",
    label="Box blur (average)",
    category=Category.BLUR,
    tier=2,
    cost=Cost.CHEAP,
    tags=["blur", "average", "box", "mean"],
    summary="Plain box average — fastest blur, slightly boxy artefacts.",
    params=[p_kernel("ksize", "Kernel (w × h)", (5, 5), odd=False, hi=99)],
)
def _blur(ctx: OpContext) -> OpResult:
    kw, kh = max(1, int(ctx.p["ksize"][0])), max(1, int(ctx.p["ksize"][1]))
    out = cv2.blur(ctx.image, (kw, kh))
    return OpResult(
        image=out,
        code=[f"img = cv2.blur(img, ({kw}, {kh}))"],
        summary={"Kernel": f"{kw} × {kh}", "Output": shape_summary(out)},
    )


@register(
    id="bilateral_filter",
    cv="cv2.bilateralFilter",
    label="Bilateral filter",
    category=Category.BLUR,
    tier=2,
    cost=Cost.EXPENSIVE,
    tags=["bilateral", "edge preserving", "denoise", "smooth"],
    summary="Smooths flat regions while keeping edges sharp. Noticeably slow — this op "
            "uses the downscaled live-preview path.",
    params=[
        p_int("d", "d (diameter)", 9, 1, 31,
              help="Pixel neighbourhood diameter. 5 = fast, 9 = good, >9 = slow."),
        p_float("sigma_color", "sigmaColor", 75.0, 1.0, 300.0, 1.0),
        p_float("sigma_space", "sigmaSpace", 75.0, 1.0, 300.0, 1.0),
    ],
)
def _bilateral(ctx: OpContext) -> OpResult:
    d = int(ctx.p["d"])
    sc, ss = float(ctx.p["sigma_color"]), float(ctx.p["sigma_space"])
    out = cv2.bilateralFilter(as_u8(ctx.image), d, sc, ss)
    return OpResult(
        image=out,
        code=["img = cv2.bilateralFilter(",
              f"    img, {d}, {fmt_num(sc)}, {fmt_num(ss)}", ")"],
        summary={"d": d, "sigmaColor": sc, "sigmaSpace": ss, "Output": shape_summary(out)},
    )


_K2D_PRESETS = (
    Choice("sharpen", "Sharpen 3×3"),
    Choice("edge_enhance", "Edge enhance"),
    Choice("emboss", "Emboss"),
    Choice("box5", "Box 5×5"),
    Choice("identity", "Identity"),
    Choice("custom", "Custom (below)"),
)

_KERNELS: dict[str, list[list[float]]] = {
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "edge_enhance": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
    "emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    "box5": [[1 / 25] * 5 for _ in range(5)],
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}


@register(
    id="filter_2d",
    cv="cv2.filter2D",
    label="Custom kernel (filter2D)",
    category=Category.BLUR,
    tier=3,
    cost=Cost.MEDIUM,
    tags=["filter2d", "convolution", "kernel", "sharpen", "emboss", "custom"],
    summary="Arbitrary convolution kernel with useful presets.",
    params=[
        p_enum("preset", "Kernel", "sharpen", _K2D_PRESETS),
        p_int("size", "Custom size", 3, 3, 9, 2, visible_when={"preset": ["custom"]}),
        p_float("center", "Custom centre weight", 5.0, -50.0, 50.0, 0.5,
                visible_when={"preset": ["custom"]}),
        p_float("neighbour", "Custom neighbour weight", -1.0, -50.0, 50.0, 0.5,
                visible_when={"preset": ["custom"]}),
        p_bool("normalize", "Normalise kernel sum to 1", False),
    ],
)
def _filter2d(ctx: OpContext) -> OpResult:
    preset = ctx.p["preset"]
    if preset == "custom":
        n = ensure_odd(int(ctx.p["size"]), 3)
        k = np.full((n, n), float(ctx.p["neighbour"]), np.float32)
        k[n // 2, n // 2] = float(ctx.p["center"])
    else:
        k = np.array(_KERNELS[preset], np.float32)
    if ctx.p["normalize"]:
        s = float(k.sum())
        if abs(s) > 1e-9:
            k = k / s
    out = cv2.filter2D(ctx.image, -1, k)
    rows = ",\n    ".join("[" + ", ".join(f"{v:g}" for v in row) + "]" for row in k)
    return OpResult(
        image=out,
        code=[f"kernel = np.array([\n    {rows}\n], dtype=np.float32)",
              "img = cv2.filter2D(img, -1, kernel)"],
        imports=["import numpy as np"],
        data={"kernel": [[round(float(v), 4) for v in row] for row in k],
              "kernel_sum": round(float(k.sum()), 4)},
        summary={"Kernel": f"{k.shape[0]} × {k.shape[1]}",
                 "Sum": round(float(k.sum()), 4), "Preset": preset},
    )
