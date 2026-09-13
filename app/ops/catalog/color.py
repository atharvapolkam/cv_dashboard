"""Colour-space conversion and channel work."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.errors import OperationError
from app.ops.constants import COLOR_CONVERSIONS, cv_const
from app.ops.helpers import as_bgr, as_gray, cvc, image_stats, shape_summary
from app.ops.params import Choice, p_bool, p_enum, p_image, p_int
from app.ops.registry import Category, Cost, OpContext, OpResult, OutputKind, register

_CH_COUNT = {"COLOR_BGR2GRAY": 1, "COLOR_RGB2GRAY": 1, "COLOR_GRAY2BGR": 3,
             "COLOR_GRAY2BGRA": 4, "COLOR_BGR2BGRA": 4, "COLOR_BGRA2BGR": 3}


@register(
    id="cvt_color",
    cv="cv2.cvtColor",
    label="Convert colour space",
    category=Category.COLOR,
    tier=1,
    cost=Cost.CHEAP,
    quick=True,
    tags=["cvtcolor", "bgr", "rgb", "gray", "hsv", "lab", "hls", "ycrcb", "colour", "color"],
    summary="cv2.cvtColor with every conversion you actually use, emitting the real constant.",
    doc="d8/d01/group__imgproc__color__conversions.html",
    params=[p_enum("code", "Conversion", "COLOR_BGR2GRAY", COLOR_CONVERSIONS)],
)
def _cvt(ctx: OpContext) -> OpResult:
    const = ctx.p["code"]
    src_ch = ctx.channels
    want_in = 1 if const.startswith(("COLOR_GRAY2",)) else (4 if "BGRA2" in const else 3)
    img = ctx.image
    note = []
    if want_in == 1 and src_ch != 1:
        img = as_gray(img)
        note.append(f"Input had {src_ch} channels; converted to gray first.")
    elif want_in == 3 and src_ch != 3:
        img = as_bgr(img)
        note.append(f"Input had {src_ch} channels; treated as BGR.")
    elif want_in == 4 and src_ch != 4:
        img = cv2.cvtColor(as_bgr(img), cv2.COLOR_BGR2BGRA)
        note.append(f"Input had {src_ch} channels; added an alpha channel first.")

    try:
        out = cv2.cvtColor(img, cv_const(const))
    except cv2.error as exc:  # pragma: no cover - guarded above
        raise OperationError(f"cvtColor failed: {exc}",
                             hint="Check the current channel count in the Inspector.") from exc

    out_ch = 1 if out.ndim == 2 else out.shape[2]
    return OpResult(
        image=out,
        code=[f"img = cv2.cvtColor(img, {cvc(const)})"],
        data={"input_channels": src_ch, "output_channels": out_ch,
              "stats": image_stats(out)},
        summary={
            "Conversion": const.replace("COLOR_", "").replace("2", " → "),
            "Input": shape_summary(ctx.image),
            "Output": shape_summary(out),
            "Channels": f"{src_ch} → {out_ch}",
        },
        notes=note + (["H is 0–179 in OpenCV (not 0–359). S and V are 0–255."]
                      if "2HSV" in const and "FULL" not in const else []),
    )


# Dedicated one-click conversions for the Quick Test strip -------------------
def _make_quick_cvt(op_id: str, const: str, label: str, quick: bool = True):
    @register(
        id=op_id, cv="cv2.cvtColor", label=label, category=Category.COLOR,
        tier=1, cost=Cost.CHEAP, quick=quick,
        tags=["cvtcolor", "colour", "color", label.lower()],
        summary=f"One-click cv2.cvtColor(img, cv2.{const}).",
    )
    def _fn(ctx: OpContext, _const: str = const) -> OpResult:
        src = ctx.image
        pre: list[str] = []
        if _const.startswith("COLOR_GRAY2") and ctx.channels != 1:
            src = as_gray(src)
            pre.append("img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        elif not _const.startswith("COLOR_GRAY2") and ctx.channels == 1:
            src = as_bgr(src)
            pre.append("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)")
        out = cv2.cvtColor(src, cv_const(_const))
        return OpResult(
            image=out,
            code=[*pre, f"img = cv2.cvtColor(img, {cvc(_const)})"],
            summary={"Conversion": _const.replace("COLOR_", "").replace("2", " → "),
                     "Input": shape_summary(ctx.image), "Output": shape_summary(out)},
        )

    return _fn


_make_quick_cvt("bgr2rgb", "COLOR_BGR2RGB", "BGR → RGB")
_make_quick_cvt("bgr2gray", "COLOR_BGR2GRAY", "BGR → Gray")
_make_quick_cvt("bgr2hsv", "COLOR_BGR2HSV", "BGR → HSV")
_make_quick_cvt("rgb2bgr", "COLOR_RGB2BGR", "RGB → BGR")
_make_quick_cvt("gray2bgr", "COLOR_GRAY2BGR", "Gray → BGR")
_make_quick_cvt("bgr2lab", "COLOR_BGR2LAB", "BGR → LAB", quick=False)
_make_quick_cvt("bgr2hls", "COLOR_BGR2HLS", "BGR → HLS", quick=False)
_make_quick_cvt("bgr2ycrcb", "COLOR_BGR2YCrCb", "BGR → YCrCb", quick=False)


# ------------------------------------------------------------------ channels
_CHANNEL_PICK = (
    Choice("all", "All channels (grid)"),
    Choice("0", "Channel 0"),
    Choice("1", "Channel 1"),
    Choice("2", "Channel 2"),
    Choice("3", "Channel 3 (alpha)"),
)


@register(
    id="split",
    cv="cv2.split",
    label="Split channels",
    category=Category.CHANNELS,
    tier=2,
    cost=Cost.CHEAP,
    output=OutputKind.DATA,
    tags=["split", "channels", "b", "g", "r", "bgr"],
    summary="Show each channel separately. Also confirms channel order at a glance.",
    params=[
        p_enum("pick", "Show", "all", _CHANNEL_PICK),
        p_bool("colorize", "Tint channels", False,
               help="Render B/G/R channels in their own colour instead of grayscale."),
    ],
)
def _split(ctx: OpContext) -> OpResult:
    img = ctx.image
    if img.ndim == 2:
        return OpResult(
            image=img,
            extras={"channel_0 (gray)": img},
            code=["# single-channel image: cv2.split(img) returns one plane",
                  "planes = cv2.split(img)"],
            summary={"Channels": "1 (already single-channel)"},
            notes=["This image has one channel; nothing to split."],
        )

    planes = cv2.split(img)
    n = len(planes)
    names = {3: ["B", "G", "R"], 4: ["B", "G", "R", "A"]}.get(n, [str(i) for i in range(n)])
    tint = {"B": (255, 0, 0), "G": (0, 255, 0), "R": (0, 0, 255)}
    extras: dict[str, np.ndarray] = {}
    pick = ctx.p["pick"]
    for i, plane in enumerate(planes):
        if pick != "all" and int(pick) != i:
            continue
        label = f"{names[i]} channel"
        if ctx.p["colorize"] and names[i] in tint:
            canvas = np.zeros((*plane.shape, 3), np.uint8)
            ch_idx = {"B": 0, "G": 1, "R": 2}[names[i]]
            canvas[:, :, ch_idx] = plane
            extras[label] = canvas
        else:
            extras[label] = plane

    stats = {names[i]: image_stats(p) for i, p in enumerate(planes)}
    return OpResult(
        image=None if pick == "all" else planes[int(pick)],
        extras=extras,
        code=[f"{', '.join(n.lower() for n in names)} = cv2.split(img)"]
              + ([] if pick == "all" else [f"img = {names[int(pick)].lower()}"]),
        data={"channel_count": n, "names": names, "stats": stats},
        summary={"Channels": n, "Order": " / ".join(names),
                 **{f"{k} mean": v["mean"] for k, v in stats.items()}},
    )


@register(
    id="merge",
    cv="cv2.merge",
    label="Merge / reorder channels",
    category=Category.CHANNELS,
    tier=2,
    cost=Cost.CHEAP,
    needs_color=True,
    tags=["merge", "channels", "reorder", "swap"],
    summary="Rebuild an image from channels in any order — the cheapest way to see a "
            "BGR/RGB mix-up.",
    params=[
        p_int("c0", "Output ch 0 ← source ch", 0, 0, 3),
        p_int("c1", "Output ch 1 ← source ch", 1, 0, 3),
        p_int("c2", "Output ch 2 ← source ch", 2, 0, 3),
        p_bool("zero_c0", "Zero out ch 0", False),
        p_bool("zero_c1", "Zero out ch 1", False),
        p_bool("zero_c2", "Zero out ch 2", False),
    ],
)
def _merge(ctx: OpContext) -> OpResult:
    img = as_bgr(ctx.image)
    planes = cv2.split(img)
    idx = [min(int(ctx.p[f"c{i}"]), len(planes) - 1) for i in range(3)]
    chosen = []
    for i, src_i in enumerate(idx):
        plane = planes[src_i]
        if ctx.p[f"zero_c{i}"]:
            plane = np.zeros_like(plane)
        chosen.append(plane)
    out = cv2.merge(chosen)
    parts = []
    for i, src_i in enumerate(idx):
        parts.append("np.zeros_like(planes[0])" if ctx.p[f"zero_c{i}"] else f"planes[{src_i}]")
    return OpResult(
        image=out,
        code=["planes = cv2.split(img)", f"img = cv2.merge([{', '.join(parts)}])"],
        imports=["import numpy as np"] if any(ctx.p[f"zero_c{i}"] for i in range(3)) else [],
        summary={"Mapping": f"out[0]←src[{idx[0]}], out[1]←src[{idx[1]}], out[2]←src[{idx[2]}]",
                 "Output": shape_summary(out)},
    )


@register(
    id="extract_hsv_channel",
    cv="cv2.cvtColor + cv2.split",
    label="HSV channel viewer",
    category=Category.CHANNELS,
    tier=2,
    cost=Cost.CHEAP,
    output=OutputKind.DATA,
    tags=["hsv", "hue", "saturation", "value", "channels"],
    summary="Hue / Saturation / Value planes side by side — pick thresholds by eye.",
)
def _hsv_channels(ctx: OpContext) -> OpResult:
    hsv = cv2.cvtColor(as_bgr(ctx.image), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Hue rendered as a colour wheel slice reads far better than raw 0-179 gray.
    hue_vis = cv2.cvtColor(cv2.merge([h, np.full_like(s, 255), np.full_like(v, 255)]),
                           cv2.COLOR_HSV2BGR)
    return OpResult(
        image=None,
        extras={"Hue (colourised)": hue_vis, "Hue (raw 0-179)": h,
                "Saturation": s, "Value": v},
        code=["hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)", "h, s, v = cv2.split(hsv)"],
        data={"hue": image_stats(h), "sat": image_stats(s), "val": image_stats(v)},
        summary={"Hue range": f"{int(h.min())} – {int(h.max())} (of 0–179)",
                 "Sat mean": image_stats(s)["mean"],
                 "Val mean": image_stats(v)["mean"]},
    )
