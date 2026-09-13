"""Resize, crop, flip, rotate, borders — the highest-traffic ops."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.errors import OperationError
from app.ops.constants import BORDER_TYPES, FLIP_CODES, INTERPOLATIONS, ROTATE_CODES, cv_const
from app.ops.helpers import (
    as_bgr, cvc, fmt_num, image_stats, py_tuple, shape_summary,
)
from app.ops.params import Choice, p_bool, p_color, p_enum, p_float, p_int
from app.ops.registry import Category, Cost, OpContext, OpResult, OutputKind, register

# --------------------------------------------------------------------- resize
_RESIZE_MODES = (
    Choice("dims", "Target width × height"),
    Choice("scale", "Scale factor (fx, fy)"),
)


@register(
    id="resize",
    cv="cv2.resize",
    label="Resize",
    category=Category.GEOMETRY,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    tags=["resize", "scale", "yolo", "640", "letterbox", "dimensions", "downscale"],
    summary="Change image dimensions. The #1 source of coordinate bugs in CV pipelines.",
    doc="d4/d86/group__imgproc__filter.html",
    params=[
        p_enum("mode", "Mode", "dims", _RESIZE_MODES),
        p_int("width", "Width", 640, 1, 10000, visible_when={"mode": ["dims"]}),
        p_int("height", "Height", 640, 1, 10000, visible_when={"mode": ["dims"]}),
        p_float("fx", "Scale X", 0.5, 0.01, 8.0, 0.01, visible_when={"mode": ["scale"]}),
        p_float("fy", "Scale Y", 0.5, 0.01, 8.0, 0.01, visible_when={"mode": ["scale"]}),
        p_bool("keep_aspect", "Lock aspect ratio", False, visible_when={"mode": ["dims"]},
               help="Fits inside width×height without distorting."),
        p_enum("interpolation", "Interpolation", "INTER_LINEAR", INTERPOLATIONS),
    ],
)
def _resize(ctx: OpContext) -> OpResult:
    src_h, src_w = ctx.image.shape[:2]
    interp = ctx.p["interpolation"]

    if ctx.p["mode"] == "scale":
        fx, fy = float(ctx.p["fx"]), float(ctx.p["fy"])
        tw, th = max(1, int(round(src_w * fx))), max(1, int(round(src_h * fy)))
        out = cv2.resize(ctx.image, None, fx=fx, fy=fy, interpolation=cv_const(interp))
        code = [
            f"img = cv2.resize(img, None, fx={fmt_num(fx)}, fy={fmt_num(fy)}, "
            f"interpolation={cvc(interp)})"
        ]
    else:
        tw, th = int(ctx.p["width"]), int(ctx.p["height"])
        if ctx.p["keep_aspect"]:
            r = min(tw / src_w, th / src_h)
            tw, th = max(1, int(round(src_w * r))), max(1, int(round(src_h * r)))
        out = cv2.resize(ctx.image, (tw, th), interpolation=cv_const(interp))
        code = [f"img = cv2.resize(img, ({tw}, {th}), interpolation={cvc(interp)})"]

    sx, sy = tw / src_w, th / src_h
    return OpResult(
        image=out,
        code=code,
        data={
            "original": {"width": src_w, "height": src_h},
            "output": {"width": tw, "height": th},
            "scale_x": round(sx, 6),
            "scale_y": round(sy, 6),
            "aspect_original": round(src_w / src_h, 6),
            "aspect_output": round(tw / th, 6),
            "aspect_preserved": abs(sx - sy) < 1e-6,
        },
        summary={
            "Input": f"{src_w} × {src_h}",
            "Output": f"{tw} × {th}",
            "Scale": f"{sx:.4f} × {sy:.4f}",
            "Aspect": "preserved" if abs(sx - sy) < 1e-6 else "DISTORTED",
        },
        notes=([] if abs(sx - sy) < 1e-6 else
               [f"Aspect ratio changed ({src_w/src_h:.4f} → {tw/th:.4f}). "
                f"Detections trained on square-letterboxed input may shift."]),
    )


# --------------------------------------------------------------------- crop
@register(
    id="crop_rect",
    cv="numpy slice",
    label="Crop (pixel rect)",
    category=Category.CROP,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    tags=["crop", "roi", "slice", "rectangle", "region"],
    summary="img[y1:y2, x1:x2] — pixel-coordinate rectangle crop.",
    params=[
        p_int("x1", "x1", 0, 0, 10000, bind_dim="width"),
        p_int("y1", "y1", 0, 0, 10000, bind_dim="height"),
        p_int("x2", "x2", 100, 1, 10000, bind_dim="width"),
        p_int("y2", "y2", 100, 1, 10000, bind_dim="height"),
    ],
)
def _crop_rect(ctx: OpContext) -> OpResult:
    h, w = ctx.image.shape[:2]
    x1, y1, x2, y2 = (int(ctx.p[k]) for k in ("x1", "y1", "x2", "y2"))
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise OperationError(
            f"Empty crop: x {x1}→{x2}, y {y1}→{y2}.",
            hint="Width and height must be at least 1 px. Draw a rectangle ROI and "
                 "press 'Send to crop' to fill these fields.",
        )
    out = ctx.image[y1:y2, x1:x2].copy()
    return OpResult(
        image=out,
        code=[f"x1, y1, x2, y2 = {x1}, {y1}, {x2}, {y2}", "img = img[y1:y2, x1:x2]"],
        data={
            "rect": {"x": x1, "y": y1, "x2": x2, "y2": y2,
                     "width": x2 - x1, "height": y2 - y1,
                     "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2},
            "normalized": {"x": round(x1 / w, 6), "y": round(y1 / h, 6),
                           "x2": round(x2 / w, 6), "y2": round(y2 / h, 6),
                           "w": round((x2 - x1) / w, 6), "h": round((y2 - y1) / h, 6)},
            "output": {"width": x2 - x1, "height": y2 - y1},
        },
        summary={
            "Input": f"{w} × {h}",
            "Crop": f"x {x1}→{x2}, y {y1}→{y2}",
            "Output": f"{x2 - x1} × {y2 - y1}",
            "Area kept": f"{100.0 * (x2-x1) * (y2-y1) / (w*h):.2f} %",
        },
    )


@register(
    id="crop_norm",
    cv="numpy slice",
    label="Crop (normalised)",
    category=Category.CROP,
    tier=2,
    cost=Cost.CHEAP,
    tags=["crop", "normalized", "yolo", "roi", "relative"],
    summary="Crop using 0–1 normalised coordinates — resolution independent.",
    params=[
        p_float("nx1", "x1 (0-1)", 0.0, 0.0, 1.0, 0.001),
        p_float("ny1", "y1 (0-1)", 0.0, 0.0, 1.0, 0.001),
        p_float("nx2", "x2 (0-1)", 0.5, 0.0, 1.0, 0.001),
        p_float("ny2", "y2 (0-1)", 0.5, 0.0, 1.0, 0.001),
    ],
)
def _crop_norm(ctx: OpContext) -> OpResult:
    h, w = ctx.image.shape[:2]
    x1 = int(round(ctx.p["nx1"] * w)); x2 = int(round(ctx.p["nx2"] * w))
    y1 = int(round(ctx.p["ny1"] * h)); y2 = int(round(ctx.p["ny2"] * h))
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise OperationError("Normalised crop is empty.", hint="Increase x2/y2.")
    out = ctx.image[y1:y2, x1:x2].copy()
    return OpResult(
        image=out,
        code=[
            "h, w = img.shape[:2]",
            f"x1, y1 = int({fmt_num(ctx.p['nx1'])} * w), int({fmt_num(ctx.p['ny1'])} * h)",
            f"x2, y2 = int({fmt_num(ctx.p['nx2'])} * w), int({fmt_num(ctx.p['ny2'])} * h)",
            "img = img[y1:y2, x1:x2]",
        ],
        data={"pixel": {"x": x1, "y": y1, "x2": x2, "y2": y2,
                        "width": x2 - x1, "height": y2 - y1}},
        summary={"Input": f"{w} × {h}", "Pixel rect": f"({x1}, {y1}) → ({x2}, {y2})",
                 "Output": f"{x2 - x1} × {y2 - y1}"},
    )


# ---------------------------------------------------------------- flip/rotate
@register(
    id="flip",
    cv="cv2.flip",
    label="Flip",
    category=Category.GEOMETRY,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    tags=["flip", "mirror", "augment"],
    summary="Mirror horizontally, vertically or both.",
    params=[p_enum("flip_code", "Direction", "1", FLIP_CODES)],
)
def _flip(ctx: OpContext) -> OpResult:
    code_val = int(ctx.p["flip_code"])
    names = {1: "horizontal", 0: "vertical", -1: "both axes"}
    return OpResult(
        image=cv2.flip(ctx.image, code_val),
        code=[f"img = cv2.flip(img, {code_val})"],
        summary={"Direction": names[code_val], "Size": shape_summary(ctx.image)},
    )


@register(
    id="rotate",
    cv="cv2.rotate",
    label="Rotate 90/180",
    category=Category.GEOMETRY,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    tags=["rotate", "90", "180", "orientation"],
    summary="Lossless 90° multiples. Swaps width/height for the 90° cases.",
    params=[p_enum("rotate_code", "Rotation", "ROTATE_90_CLOCKWISE", ROTATE_CODES)],
)
def _rotate(ctx: OpContext) -> OpResult:
    const = ctx.p["rotate_code"]
    out = cv2.rotate(ctx.image, cv_const(const))
    return OpResult(
        image=out,
        code=[f"img = cv2.rotate(img, {cvc(const)})"],
        summary={"Input": f"{ctx.width} × {ctx.height}",
                 "Output": f"{out.shape[1]} × {out.shape[0]}"},
    )


@register(
    id="transpose",
    cv="cv2.transpose",
    label="Transpose",
    category=Category.GEOMETRY,
    tier=2,
    cost=Cost.CHEAP,
    tags=["transpose", "swap axes"],
    summary="Swap rows and columns (matrix transpose).",
)
def _transpose(ctx: OpContext) -> OpResult:
    out = cv2.transpose(ctx.image)
    return OpResult(
        image=out,
        code=["img = cv2.transpose(img)"],
        summary={"Input": f"{ctx.width} × {ctx.height}",
                 "Output": f"{out.shape[1]} × {out.shape[0]}"},
    )


@register(
    id="copy_make_border",
    cv="cv2.copyMakeBorder",
    label="Pad / border",
    category=Category.GEOMETRY,
    tier=3,
    cost=Cost.CHEAP,
    tags=["border", "pad", "letterbox", "yolo"],
    summary="Pad the image — the letterbox step of most detector preprocessors.",
    params=[
        p_int("top", "Top", 20, 0, 2000),
        p_int("bottom", "Bottom", 20, 0, 2000),
        p_int("left", "Left", 20, 0, 2000),
        p_int("right", "Right", 20, 0, 2000),
        p_enum("border_type", "Border type", "BORDER_CONSTANT", BORDER_TYPES),
        p_color("value", "Fill colour", (0, 0, 0),
                visible_when={"border_type": ["BORDER_CONSTANT"]}),
    ],
)
def _border(ctx: OpContext) -> OpResult:
    t, b, l, r = (int(ctx.p[k]) for k in ("top", "bottom", "left", "right"))
    bt = ctx.p["border_type"]
    color = [int(c) for c in ctx.p["value"]]
    out = cv2.copyMakeBorder(ctx.image, t, b, l, r, cv_const(bt),
                             value=color if bt == "BORDER_CONSTANT" else None)
    extra = f", value={py_tuple(color)}" if bt == "BORDER_CONSTANT" else ""
    return OpResult(
        image=out,
        code=[f"img = cv2.copyMakeBorder(img, {t}, {b}, {l}, {r}, {cvc(bt)}{extra})"],
        summary={"Input": f"{ctx.width} × {ctx.height}",
                 "Output": f"{out.shape[1]} × {out.shape[0]}",
                 "Padding": f"T{t} B{b} L{l} R{r}"},
    )


# ------------------------------------------------------------------ letterbox
@register(
    id="letterbox",
    cv="cv2.resize + cv2.copyMakeBorder",
    label="Letterbox (YOLO)",
    category=Category.GEOMETRY,
    tier=2,
    cost=Cost.CHEAP,
    tags=["letterbox", "yolo", "640", "preprocess", "pad", "aspect"],
    summary="Aspect-preserving resize + centre pad to a square — YOLO-style preprocessing, "
            "with the exact scale/pad offsets you need to map detections back.",
    params=[
        p_int("size", "Target size", 640, 32, 2048, 32),
        p_color("pad_color", "Pad colour", (114, 114, 114)),
        p_enum("interpolation", "Interpolation", "INTER_LINEAR", INTERPOLATIONS),
    ],
)
def _letterbox(ctx: OpContext) -> OpResult:
    size = int(ctx.p["size"])
    h, w = ctx.image.shape[:2]
    r = min(size / w, size / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    interp = ctx.p["interpolation"]
    resized = cv2.resize(ctx.image, (nw, nh), interpolation=cv_const(interp))
    dw, dh = size - nw, size - nh
    top, left = dh // 2, dw // 2
    bottom, right = dh - top, dw - left
    color = [int(c) for c in ctx.p["pad_color"]]
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return OpResult(
        image=out,
        code=[
            f"size = {size}",
            "h, w = img.shape[:2]",
            "r = min(size / w, size / h)",
            "nw, nh = int(round(w * r)), int(round(h * r))",
            f"resized = cv2.resize(img, (nw, nh), interpolation={cvc(interp)})",
            "dw, dh = size - nw, size - nh",
            "top, left = dh // 2, dw // 2",
            "img = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,",
            f"                        cv2.BORDER_CONSTANT, value={py_tuple(color)})",
        ],
        data={"scale": round(r, 6), "pad_x": left, "pad_y": top,
              "resized": {"width": nw, "height": nh},
              "unmap_hint": "x_orig = (x_letterbox - pad_x) / scale"},
        summary={"Input": f"{w} × {h}", "Output": f"{size} × {size}",
                 "Scale r": f"{r:.6f}", "Pad (x, y)": f"{left}, {top}",
                 "Un-map": "x_orig = (x_lb - pad_x) / r"},
    )
