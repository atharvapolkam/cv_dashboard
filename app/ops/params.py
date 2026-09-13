"""Parameter specification language for the operation registry.

A ``ParamSpec`` is the single source of truth for one operation argument:

* the backend uses it to **coerce and validate** incoming JSON,
* the frontend uses the same JSON to **render the widget** (slider, select,
  checkbox, point picker ...),
* the code generator uses it to **emit the literal** in Python.

Adding a parameter therefore never requires touching the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from app.core.errors import ValidationError


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"              # single choice, value is a name from ops.constants
    TEXT = "text"
    COLOR = "color"            # -> (B, G, R)
    POINT = "point"            # -> [x, y]
    POINTS = "points"          # -> [[x, y], ...]
    KERNEL = "kernel"          # -> odd (w, h) pair rendered as two linked ints
    IMAGE_REF = "image_ref"    # id of a second image in the session
    RANGE3 = "range3"          # triple lower/upper pair, used by inRange


@dataclass(frozen=True)
class Choice:
    """One option of an ENUM param.

    ``const`` is the *OpenCV constant name* (e.g. ``INTER_AREA``) which is what
    we emit in generated code; ``value`` is resolved at runtime from cv2.
    """

    value: str
    label: str
    const: str | None = None
    help: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label, "const": self.const, "help": self.help}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: ParamType
    label: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: Sequence[Choice] = field(default_factory=tuple)
    odd: bool = False              # force odd values (kernel sizes)
    help: str = ""
    group: str | None = None       # renders inside a labelled sub-group
    # Show this widget only when another param has one of these values.
    visible_when: dict[str, list[Any]] | None = None
    # Bind slider to image dimensions: "width" | "height" -> max auto-filled
    bind_dim: str | None = None
    optional: bool = False         # None is an accepted value
    advanced: bool = False         # collapsed by default in the UI

    # ------------------------------------------------------------------ json
    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["choices"] = [c.to_json() for c in self.choices]
        return d

    # -------------------------------------------------------------- coercion
    def coerce(self, raw: Any) -> Any:
        """Validate + normalise one incoming value. Raises ValidationError."""
        if raw is None:
            if self.optional:
                return None
            raw = self.default
        if raw is None and not self.optional:
            raise ValidationError(f"Parameter '{self.name}' is required")

        t = self.type
        try:
            if t is ParamType.INT:
                v = int(round(float(raw)))
                if self.odd and v % 2 == 0:
                    v += 1
                return self._clamp(v)
            if t is ParamType.FLOAT:
                return float(self._clamp(float(raw)))
            if t is ParamType.BOOL:
                if isinstance(raw, str):
                    return raw.strip().lower() in {"1", "true", "yes", "on"}
                return bool(raw)
            if t is ParamType.ENUM:
                s = str(raw)
                allowed = {c.value for c in self.choices}
                if allowed and s not in allowed:
                    raise ValidationError(
                        f"Parameter '{self.name}' must be one of {sorted(allowed)}, got '{s}'"
                    )
                return s
            if t is ParamType.TEXT:
                return str(raw)
            if t is ParamType.COLOR:
                return _coerce_color(raw)
            if t is ParamType.POINT:
                return _coerce_point(raw)
            if t is ParamType.POINTS:
                pts = [_coerce_point(p) for p in raw]
                if len(pts) < 2:
                    raise ValidationError(f"Parameter '{self.name}' needs at least 2 points")
                return pts
            if t is ParamType.KERNEL:
                k = _coerce_point(raw)
                w = max(1, int(k[0]))
                h = max(1, int(k[1]))
                if self.odd:
                    w += (w + 1) % 2
                    h += (h + 1) % 2
                return [w, h]
            if t is ParamType.IMAGE_REF:
                return str(raw)
            if t is ParamType.RANGE3:
                vals = [int(round(float(x))) for x in raw]
                if len(vals) != 6:
                    raise ValidationError(
                        f"Parameter '{self.name}' expects 6 numbers [l0,l1,l2,u0,u1,u2]"
                    )
                return vals
        except ValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Parameter '{self.name}': {exc}") from exc
        raise ValidationError(f"Unhandled param type {t}")

    def _clamp(self, v: float) -> float:
        if self.min is not None and v < self.min:
            return self.min
        if self.max is not None and v > self.max:
            return self.max
        return v


def _coerce_point(raw: Any) -> list[int]:
    if isinstance(raw, dict):
        return [int(round(float(raw.get("x", 0)))), int(round(float(raw.get("y", 0))))]
    seq = list(raw)
    if len(seq) < 2:
        raise ValidationError("Point needs 2 values")
    return [int(round(float(seq[0]))), int(round(float(seq[1])))]


def _coerce_color(raw: Any) -> list[int]:
    """Accept ``#rrggbb``, ``[b,g,r]`` or ``{r,g,b}`` -> BGR triple."""
    if isinstance(raw, str):
        s = raw.strip().lstrip("#")
        if len(s) == 6:
            r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
            return [b, g, r]
        raise ValidationError(f"Bad colour '{raw}'")
    if isinstance(raw, dict):
        return [int(raw.get("b", 0)), int(raw.get("g", 0)), int(raw.get("r", 0))]
    seq = [int(round(float(x))) for x in raw][:4]
    while len(seq) < 3:
        seq.append(0)
    return [max(0, min(255, c)) for c in seq]


# ----------------------------------------------------------------- shortcuts
def p_int(name: str, label: str, default: int, lo: int | None = None, hi: int | None = None,
          step: int = 1, **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.INT, label, default, lo, hi, step, **kw)


def p_odd(name: str, label: str, default: int, lo: int = 1, hi: int = 99, **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.INT, label, default, lo, hi, 2, odd=True, **kw)


def p_float(name: str, label: str, default: float, lo: float | None = None,
            hi: float | None = None, step: float = 0.1, **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.FLOAT, label, default, lo, hi, step, **kw)


def p_bool(name: str, label: str, default: bool = False, **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.BOOL, label, default, **kw)


def p_enum(name: str, label: str, default: str, choices: Sequence[Choice], **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.ENUM, label, default, choices=tuple(choices), **kw)


def p_kernel(name: str = "ksize", label: str = "Kernel", default: tuple[int, int] = (5, 5),
             odd: bool = True, hi: int = 99, **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.KERNEL, label, list(default), 1, hi, 2, odd=odd, **kw)


def p_color(name: str = "color", label: str = "Colour",
            default: tuple[int, int, int] = (0, 255, 0), **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.COLOR, label, list(default), **kw)


def p_point(name: str, label: str, default: tuple[int, int] = (0, 0), **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.POINT, label, list(default), **kw)


def p_points(name: str = "points", label: str = "Points", **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.POINTS, label, [], **kw)


def p_text(name: str, label: str, default: str = "", **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.TEXT, label, default, **kw)


def p_image(name: str = "image2", label: str = "Second image", **kw: Any) -> ParamSpec:
    return ParamSpec(name, ParamType.IMAGE_REF, label, None, optional=True, **kw)
