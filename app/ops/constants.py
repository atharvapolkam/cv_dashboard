"""OpenCV constant tables.

Every enum parameter references constants **by name**. That gives us two
things at once: correct runtime values via ``getattr(cv2, name)`` and exact,
copy-pasteable generated code (``cv2.INTER_AREA``, not ``3``).
"""

from __future__ import annotations

import cv2

from app.core.errors import ValidationError
from app.ops.params import Choice


def cv_const(name: str) -> int:
    """Resolve an OpenCV constant name to its int value."""
    if not hasattr(cv2, name):
        raise ValidationError(f"cv2.{name} is not available in OpenCV {cv2.__version__}")
    return int(getattr(cv2, name))


def _c(const: str, label: str, help_: str | None = None) -> Choice:
    return Choice(value=const, label=label, const=const, help=help_)


# ------------------------------------------------------------------ resize
INTERPOLATIONS = (
    _c("INTER_NEAREST", "INTER_NEAREST", "Fastest, blocky. Keeps exact label values — use for masks."),
    _c("INTER_LINEAR", "INTER_LINEAR", "Default. Good speed/quality balance."),
    _c("INTER_CUBIC", "INTER_CUBIC", "Slower, sharper when upscaling."),
    _c("INTER_AREA", "INTER_AREA", "Best for downscaling — averages the source region."),
    _c("INTER_LANCZOS4", "INTER_LANCZOS4", "Highest quality, slowest."),
)

# ------------------------------------------------------------------ colour
COLOR_CONVERSIONS = (
    _c("COLOR_BGR2RGB", "BGR → RGB"),
    _c("COLOR_RGB2BGR", "RGB → BGR"),
    _c("COLOR_BGR2GRAY", "BGR → GRAY"),
    _c("COLOR_RGB2GRAY", "RGB → GRAY"),
    _c("COLOR_BGR2HSV", "BGR → HSV", "H in 0-179, S/V in 0-255."),
    _c("COLOR_RGB2HSV", "RGB → HSV"),
    _c("COLOR_BGR2HSV_FULL", "BGR → HSV_FULL", "H scaled to 0-255."),
    _c("COLOR_BGR2LAB", "BGR → LAB"),
    _c("COLOR_RGB2LAB", "RGB → LAB"),
    _c("COLOR_BGR2HLS", "BGR → HLS"),
    _c("COLOR_BGR2YCrCb", "BGR → YCrCb"),
    _c("COLOR_BGR2YUV", "BGR → YUV"),
    _c("COLOR_GRAY2BGR", "GRAY → BGR"),
    _c("COLOR_GRAY2BGRA", "GRAY → BGRA"),
    _c("COLOR_BGR2BGRA", "BGR → BGRA"),
    _c("COLOR_BGRA2BGR", "BGRA → BGR"),
    _c("COLOR_HSV2BGR", "HSV → BGR"),
    _c("COLOR_LAB2BGR", "LAB → BGR"),
)

# --------------------------------------------------------------- threshold
THRESH_TYPES = (
    _c("THRESH_BINARY", "THRESH_BINARY"),
    _c("THRESH_BINARY_INV", "THRESH_BINARY_INV"),
    _c("THRESH_TRUNC", "THRESH_TRUNC"),
    _c("THRESH_TOZERO", "THRESH_TOZERO"),
    _c("THRESH_TOZERO_INV", "THRESH_TOZERO_INV"),
)

THRESH_AUTO = (
    Choice("none", "Manual", None, "Use the threshold slider value."),
    _c("THRESH_OTSU", "+ THRESH_OTSU", "Auto threshold from a bimodal histogram."),
    _c("THRESH_TRIANGLE", "+ THRESH_TRIANGLE", "Auto threshold, triangle method."),
)

ADAPTIVE_METHODS = (
    _c("ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_MEAN_C"),
    _c("ADAPTIVE_THRESH_GAUSSIAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"),
)

ADAPTIVE_THRESH_TYPES = (
    _c("THRESH_BINARY", "THRESH_BINARY"),
    _c("THRESH_BINARY_INV", "THRESH_BINARY_INV"),
)

# -------------------------------------------------------------- morphology
MORPH_SHAPES = (
    _c("MORPH_RECT", "MORPH_RECT"),
    _c("MORPH_ELLIPSE", "MORPH_ELLIPSE"),
    _c("MORPH_CROSS", "MORPH_CROSS"),
)

MORPH_OPS = (
    _c("MORPH_ERODE", "MORPH_ERODE"),
    _c("MORPH_DILATE", "MORPH_DILATE"),
    _c("MORPH_OPEN", "MORPH_OPEN", "Erode then dilate — removes small white specks."),
    _c("MORPH_CLOSE", "MORPH_CLOSE", "Dilate then erode — fills small black holes."),
    _c("MORPH_GRADIENT", "MORPH_GRADIENT", "Dilate − erode — outlines."),
    _c("MORPH_TOPHAT", "MORPH_TOPHAT", "src − open."),
    _c("MORPH_BLACKHAT", "MORPH_BLACKHAT", "close − src."),
)

# ----------------------------------------------------------------- contours
RETR_MODES = (
    _c("RETR_EXTERNAL", "RETR_EXTERNAL", "Outermost contours only."),
    _c("RETR_LIST", "RETR_LIST", "All contours, no hierarchy."),
    _c("RETR_CCOMP", "RETR_CCOMP", "Two-level hierarchy."),
    _c("RETR_TREE", "RETR_TREE", "Full nested hierarchy."),
)

CHAIN_APPROX = (
    _c("CHAIN_APPROX_NONE", "CHAIN_APPROX_NONE", "Store every boundary point."),
    _c("CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_SIMPLE", "Compress straight runs to endpoints."),
    _c("CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_L1"),
    _c("CHAIN_APPROX_TC89_KCOS", "CHAIN_APPROX_TC89_KCOS"),
)

# ------------------------------------------------------------------ drawing
LINE_TYPES = (
    _c("LINE_8", "LINE_8"),
    _c("LINE_4", "LINE_4"),
    _c("LINE_AA", "LINE_AA", "Anti-aliased."),
)

FONTS = (
    _c("FONT_HERSHEY_SIMPLEX", "SIMPLEX"),
    _c("FONT_HERSHEY_PLAIN", "PLAIN"),
    _c("FONT_HERSHEY_DUPLEX", "DUPLEX"),
    _c("FONT_HERSHEY_COMPLEX", "COMPLEX"),
    _c("FONT_HERSHEY_TRIPLEX", "TRIPLEX"),
    _c("FONT_HERSHEY_COMPLEX_SMALL", "COMPLEX_SMALL"),
    _c("FONT_HERSHEY_SCRIPT_SIMPLEX", "SCRIPT_SIMPLEX"),
    _c("FONT_HERSHEY_SCRIPT_COMPLEX", "SCRIPT_COMPLEX"),
)

MARKERS = (
    _c("MARKER_CROSS", "MARKER_CROSS"),
    _c("MARKER_TILTED_CROSS", "MARKER_TILTED_CROSS"),
    _c("MARKER_STAR", "MARKER_STAR"),
    _c("MARKER_DIAMOND", "MARKER_DIAMOND"),
    _c("MARKER_SQUARE", "MARKER_SQUARE"),
    _c("MARKER_TRIANGLE_UP", "MARKER_TRIANGLE_UP"),
    _c("MARKER_TRIANGLE_DOWN", "MARKER_TRIANGLE_DOWN"),
)

# ------------------------------------------------------------------ borders
BORDER_TYPES = (
    _c("BORDER_CONSTANT", "BORDER_CONSTANT"),
    _c("BORDER_REPLICATE", "BORDER_REPLICATE"),
    _c("BORDER_REFLECT", "BORDER_REFLECT"),
    _c("BORDER_REFLECT_101", "BORDER_REFLECT_101"),
    _c("BORDER_WRAP", "BORDER_WRAP"),
    _c("BORDER_ISOLATED", "BORDER_ISOLATED"),
)

# --------------------------------------------------------------- transforms
FLIP_CODES = (
    Choice("1", "Horizontal (flipCode=1)"),
    Choice("0", "Vertical (flipCode=0)"),
    Choice("-1", "Both (flipCode=-1)"),
)

ROTATE_CODES = (
    _c("ROTATE_90_CLOCKWISE", "90° clockwise"),
    _c("ROTATE_90_COUNTERCLOCKWISE", "90° counter-clockwise"),
    _c("ROTATE_180", "180°"),
)

# ---------------------------------------------------------- template match
TM_METHODS = (
    _c("TM_CCOEFF_NORMED", "TM_CCOEFF_NORMED", "Best default — higher is better."),
    _c("TM_CCORR_NORMED", "TM_CCORR_NORMED", "Higher is better."),
    _c("TM_SQDIFF_NORMED", "TM_SQDIFF_NORMED", "Lower is better."),
    _c("TM_CCOEFF", "TM_CCOEFF"),
    _c("TM_CCORR", "TM_CCORR"),
    _c("TM_SQDIFF", "TM_SQDIFF", "Lower is better."),
)

MIN_IS_BEST = {"TM_SQDIFF", "TM_SQDIFF_NORMED"}

# ------------------------------------------------------------------- depths
DDEPTHS = (
    Choice("CV_64F", "CV_64F (recommended)", "CV_64F", "Signed — keeps negative gradients."),
    Choice("CV_32F", "CV_32F", "CV_32F"),
    Choice("CV_16S", "CV_16S", "CV_16S"),
    Choice("CV_8U", "CV_8U", "CV_8U", "Clips negatives to 0 — loses half the edges."),
)

# ------------------------------------------------------------------- dtypes
NP_DTYPES = (
    Choice("uint8", "uint8"),
    Choice("float32", "float32"),
    Choice("float64", "float64"),
    Choice("int32", "int32"),
    Choice("int16", "int16"),
)
