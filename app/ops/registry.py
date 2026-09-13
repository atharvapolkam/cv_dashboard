"""The operation registry.

To add an OpenCV operation you write **one** decorated function in
``app/ops/catalog/<topic>.py``:

    @register(
        id="my_op", cv="cv2.myOp", label="My op", category=Category.FILTER,
        tier=2, cost=Cost.CHEAP, tags=["thing"], params=[p_int("k", "K", 3)],
    )
    def _my_op(ctx: OpContext) -> OpResult:
        out = cv2.myOp(ctx.image, ctx.p["k"])
        return OpResult(image=out, code=[f"img = cv2.myOp(img, {ctx.p['k']})"])

Everything else — validation, the UI widget set, live preview debounce class,
search indexing, favourites, history, generated code, the "what happened"
panel and the REST surface — is derived from that declaration. There is no
second place to edit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from app.core.errors import NotFoundError, ValidationError
from app.ops.params import ParamSpec


class Category(str, Enum):
    INSPECT = "Inspect"
    GEOMETRY = "Resize & Geometry"
    CROP = "Crop & ROI"
    COLOR = "Colour"
    THRESHOLD = "Threshold"
    BLUR = "Blur & Denoise"
    MORPHOLOGY = "Morphology"
    EDGES = "Edges"
    CONTOURS = "Contours"
    BITWISE = "Bitwise"
    ARITHMETIC = "Arithmetic"
    DRAWING = "Drawing"
    TRANSFORM = "Perspective & Affine"
    DETECT = "Geometric Detection"
    MATCH = "Template Matching"
    HISTOGRAM = "Histogram"
    CHANNELS = "Channels"
    NUMPY = "NumPy / Array"


class Cost(str, Enum):
    """Drives the frontend live-preview strategy (see static/js/core/preview.js).

    CHEAP      -> 40 ms debounce, full resolution
    MEDIUM     -> 120 ms debounce, full resolution
    EXPENSIVE  -> 260 ms debounce + downscaled preview pass
    """

    CHEAP = "cheap"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"


class OutputKind(str, Enum):
    IMAGE = "image"      # returns a new image that replaces the working image
    MASK = "mask"        # returns a single-channel mask
    DATA = "data"        # returns measurements only (image passes through)
    OVERLAY = "overlay"  # draws on top of the working image


@dataclass
class OpContext:
    """Everything an executor is allowed to touch."""

    image: np.ndarray
    p: dict[str, Any]
    # Second operand for two-input ops (arithmetic, bitwise, matchTemplate).
    image2: np.ndarray | None = None
    # Session artefacts an op may consume, e.g. contours found by a prior step.
    carry: dict[str, Any] = field(default_factory=dict)
    # True when running a throwaway live-preview pass (possibly downscaled).
    preview: bool = False
    # Scale applied to the source before this call (1.0 == original pixels).
    scale: float = 1.0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def channels(self) -> int:
        return 1 if self.image.ndim == 2 else int(self.image.shape[2])


@dataclass
class OpResult:
    """What an executor returns.

    ``image``      new working image (None keeps the input unchanged)
    ``extras``     named side images the UI shows as thumbnails (HSV, mask, ...)
    ``data``       JSON-safe measurements (contour table, stats, best match ...)
    ``code``       generated Python lines, already indented, using ``img``
    ``imports``    extra import lines the generated script needs
    ``summary``    ordered key/value rows for the "What happened?" panel
    ``carry``      artefacts stored on the session for later ops
    ``notes``      warnings/hints surfaced as a subtle line under the result
    """

    image: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    code: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    carry: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


Executor = Callable[[OpContext], OpResult]


@dataclass(frozen=True)
class OpSpec:
    id: str
    cv: str
    label: str
    category: Category
    tier: int                       # 1 = always visible, 2 = frequent, 3 = specialised
    cost: Cost
    run: Executor
    params: tuple[ParamSpec, ...] = ()
    tags: tuple[str, ...] = ()
    summary: str = ""
    quick: bool = False             # appears in the Quick Test strip
    output: OutputKind = OutputKind.IMAGE
    inputs: int = 1                 # 2 => needs a second image
    # Input requirements — checked before running so the user gets a real hint
    # instead of an OpenCV assertion dump.
    needs_gray: bool = False
    needs_binary: bool = False
    needs_color: bool = False
    doc: str = ""                   # link fragment on docs.opencv.org

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cv": self.cv,
            "label": self.label,
            "category": self.category.value,
            "tier": self.tier,
            "cost": self.cost.value,
            "tags": list(self.tags),
            "summary": self.summary,
            "quick": self.quick,
            "output": self.output.value,
            "inputs": self.inputs,
            "needs_gray": self.needs_gray,
            "needs_binary": self.needs_binary,
            "needs_color": self.needs_color,
            "doc": self.doc,
            "params": [p.to_json() for p in self.params],
            "search": " ".join([self.id, self.cv, self.label, *self.tags,
                                self.category.value, self.summary]).lower(),
        }

    # ------------------------------------------------------------- validation
    def coerce_params(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        raw = raw or {}
        unknown = set(raw) - {p.name for p in self.params}
        # Unknown keys are ignored rather than fatal: the UI may send stale
        # fields while the user switches ops mid-request.
        out: dict[str, Any] = {}
        for spec in self.params:
            out[spec.name] = spec.coerce(raw.get(spec.name))
        if unknown:
            out["_ignored"] = sorted(unknown)
            out.pop("_ignored")
        return out


class _Registry:
    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}

    def add(self, spec: OpSpec) -> None:
        if spec.id in self._ops:
            raise RuntimeError(f"Duplicate operation id '{spec.id}'")
        self._ops[spec.id] = spec

    def get(self, op_id: str) -> OpSpec:
        try:
            return self._ops[op_id]
        except KeyError:
            raise NotFoundError(f"Unknown operation '{op_id}'") from None

    def all(self) -> list[OpSpec]:
        return sorted(self._ops.values(), key=lambda s: (s.tier, s.category.value, s.label))

    def by_category(self) -> dict[str, list[OpSpec]]:
        out: dict[str, list[OpSpec]] = {}
        for spec in self.all():
            out.setdefault(spec.category.value, []).append(spec)
        return out

    def quick(self) -> list[OpSpec]:
        return [s for s in self._ops.values() if s.quick]

    def __len__(self) -> int:
        return len(self._ops)


REGISTRY = _Registry()


def register(
    *,
    id: str,
    cv: str,
    label: str,
    category: Category,
    tier: int = 2,
    cost: Cost = Cost.CHEAP,
    params: Sequence[ParamSpec] = (),
    tags: Sequence[str] = (),
    summary: str = "",
    quick: bool = False,
    output: OutputKind = OutputKind.IMAGE,
    inputs: int = 1,
    needs_gray: bool = False,
    needs_binary: bool = False,
    needs_color: bool = False,
    doc: str = "",
) -> Callable[[Executor], Executor]:
    """Decorator that registers an executor as a dashboard operation."""

    def deco(fn: Executor) -> Executor:
        REGISTRY.add(
            OpSpec(
                id=id, cv=cv, label=label, category=category, tier=tier, cost=cost,
                run=fn, params=tuple(params), tags=tuple(tags), summary=summary,
                quick=quick, output=output, inputs=inputs, needs_gray=needs_gray,
                needs_binary=needs_binary, needs_color=needs_color, doc=doc,
            )
        )
        return fn

    return deco


def load_catalog() -> int:
    """Import every catalog module so the decorators run. Idempotent."""
    from app.ops import catalog  # noqa: F401  (import triggers registration)

    return len(REGISTRY)


def require_param(p: dict[str, Any], name: str) -> Any:
    v = p.get(name)
    if v is None:
        raise ValidationError(f"Parameter '{name}' is required")
    return v
