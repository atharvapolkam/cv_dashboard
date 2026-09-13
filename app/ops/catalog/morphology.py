"""Morphology: erode, dilate, morphologyEx, structuring elements."""

from __future__ import annotations

import cv2
import numpy as np

from app.ops.constants import MORPH_OPS, MORPH_SHAPES, cv_const
from app.ops.helpers import (
    as_u8, cvc, kernel_preview, nonzero_ratio, py_tuple, shape_summary, structuring_element,
)
from app.ops.params import p_enum, p_int, p_kernel, p_point
from app.ops.registry import Category, Cost, OpContext, OpResult, register

_KP = [
    p_enum("shape", "Kernel shape", "MORPH_RECT", MORPH_SHAPES),
    p_kernel("ksize", "Kernel (w × h)", (5, 5), odd=False, hi=99),
    p_int("iterations", "Iterations", 1, 1, 20),
]


def _kernel_and_code(ctx: OpContext) -> tuple[np.ndarray, list[str], dict]:
    shape = ctx.p["shape"]
    kw, kh = max(1, int(ctx.p["ksize"][0])), max(1, int(ctx.p["ksize"][1]))
    kernel = structuring_element(shape, kw, kh)
    code = [f"kernel = cv2.getStructuringElement({cvc(shape)}, ({kw}, {kh}))"]
    info = {"kernel_shape": shape, "kernel_size": [kw, kh],
            "kernel_matrix": kernel_preview(kernel) if kw * kh <= 625 else None,
            "kernel_area": int(kernel.sum())}
    return kernel, code, info


@register(
    id="erode", cv="cv2.erode", label="Erode", category=Category.MORPHOLOGY, tier=1,
    cost=Cost.CHEAP, quick=True, params=_KP,
    tags=["erode", "morphology", "shrink", "thin"],
    summary="Shrinks white regions. Removes thin noise but eats real detail too.",
)
def _erode(ctx: OpContext) -> OpResult:
    kernel, code, info = _kernel_and_code(ctx)
    it = int(ctx.p["iterations"])
    out = cv2.erode(ctx.image, kernel, iterations=it)
    return OpResult(image=out, code=[*code, f"img = cv2.erode(img, kernel, iterations={it})"],
                    data={**info, **nonzero_ratio(out)},
                    summary={"Kernel": f"{info['kernel_size'][0]} × {info['kernel_size'][1]} "
                                       f"{info['kernel_shape']}", "Iterations": it,
                             "White pixels": f"{nonzero_ratio(out)['coverage_pct']:.2f} %"})


@register(
    id="dilate", cv="cv2.dilate", label="Dilate", category=Category.MORPHOLOGY, tier=1,
    cost=Cost.CHEAP, quick=True, params=_KP,
    tags=["dilate", "morphology", "grow", "thicken"],
    summary="Grows white regions. Joins nearby blobs before contour finding.",
)
def _dilate(ctx: OpContext) -> OpResult:
    kernel, code, info = _kernel_and_code(ctx)
    it = int(ctx.p["iterations"])
    out = cv2.dilate(ctx.image, kernel, iterations=it)
    return OpResult(image=out, code=[*code, f"img = cv2.dilate(img, kernel, iterations={it})"],
                    data={**info, **nonzero_ratio(out)},
                    summary={"Kernel": f"{info['kernel_size'][0]} × {info['kernel_size'][1]} "
                                       f"{info['kernel_shape']}", "Iterations": it,
                             "White pixels": f"{nonzero_ratio(out)['coverage_pct']:.2f} %"})


@register(
    id="morphology_ex", cv="cv2.morphologyEx", label="Morphology (open/close/…)",
    category=Category.MORPHOLOGY, tier=1, cost=Cost.CHEAP,
    params=[p_enum("op", "Operation", "MORPH_OPEN", MORPH_OPS), *_KP],
    tags=["morphologyex", "open", "close", "gradient", "tophat", "blackhat", "morphology"],
    summary="Open removes specks, Close fills holes, Gradient outlines. The standard "
            "mask-cleanup step after InRange.",
)
def _morph_ex(ctx: OpContext) -> OpResult:
    kernel, code, info = _kernel_and_code(ctx)
    op = ctx.p["op"]
    it = int(ctx.p["iterations"])
    before = nonzero_ratio(ctx.image)
    out = cv2.morphologyEx(ctx.image, cv_const(op), kernel, iterations=it)
    after = nonzero_ratio(out)
    return OpResult(
        image=out,
        code=[*code, f"img = cv2.morphologyEx(img, {cvc(op)}, kernel, iterations={it})"],
        data={**info, "op": op, "before": before, "after": after},
        summary={"Operation": op, "Kernel": f"{info['kernel_size'][0]} × "
                                           f"{info['kernel_size'][1]} {info['kernel_shape']}",
                 "Iterations": it,
                 "White before": f"{before['coverage_pct']:.2f} %",
                 "White after": f"{after['coverage_pct']:.2f} %"},
    )


def _quick_morph(op_id: str, const: str, label: str):
    @register(id=op_id, cv="cv2.morphologyEx", label=label, category=Category.MORPHOLOGY,
              tier=1, cost=Cost.CHEAP, quick=True,
              params=[p_kernel("ksize", "Kernel", (5, 5), odd=False, hi=99),
                      p_int("iterations", "Iterations", 1, 1, 20)],
              tags=["morphology", label.lower(), const.lower()],
              summary=f"One-click cv2.morphologyEx(img, cv2.{const}, kernel).")
    def _fn(ctx: OpContext, _c: str = const) -> OpResult:
        kw, kh = max(1, int(ctx.p["ksize"][0])), max(1, int(ctx.p["ksize"][1]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        it = int(ctx.p["iterations"])
        out = cv2.morphologyEx(ctx.image, cv_const(_c), kernel, iterations=it)
        return OpResult(
            image=out,
            code=[f"kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({kw}, {kh}))",
                  f"img = cv2.morphologyEx(img, {cvc(_c)}, kernel, iterations={it})"],
            data=nonzero_ratio(out),
            summary={"Operation": _c, "Kernel": f"{kw} × {kh}", "Iterations": it,
                     "White pixels": f"{nonzero_ratio(out)['coverage_pct']:.2f} %"},
        )
    return _fn


_quick_morph("morph_open", "MORPH_OPEN", "Open")
_quick_morph("morph_close", "MORPH_CLOSE", "Close")


@register(
    id="get_structuring_element", cv="cv2.getStructuringElement", label="Inspect kernel",
    category=Category.MORPHOLOGY, tier=3, cost=Cost.CHEAP,
    params=[p_enum("shape", "Shape", "MORPH_ELLIPSE", MORPH_SHAPES),
            p_kernel("ksize", "Size", (7, 7), odd=False, hi=51)],
    tags=["kernel", "structuring", "element", "visualise"],
    summary="Render the structuring element itself so you can see what a shape/size does.",
)
def _get_se(ctx: OpContext) -> OpResult:
    kernel, code, info = _kernel_and_code(ctx)
    vis = cv2.resize((kernel * 255).astype(np.uint8), (kernel.shape[1] * 24, kernel.shape[0] * 24),
                     interpolation=cv2.INTER_NEAREST)
    return OpResult(image=None, extras={"Kernel": vis}, code=code, data=info,
                    summary={"Shape": info["kernel_shape"],
                             "Size": f"{info['kernel_size'][0]} × {info['kernel_size'][1]}",
                             "Active cells": info["kernel_area"]})
