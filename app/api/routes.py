"""REST surface. One router per concern; add new routers in app.main."""
from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response

from app.config import settings
from app.core.errors import PayloadTooLargeError, UnsupportedMediaError, ValidationError
from app.ops.catalog.threshold import HSV_PRESETS
from app.ops.helpers import as_bgr, as_gray
from app.ops.registry import REGISTRY
from app.services.executor import execute, generate_script
from app.services.store import STORE

router = APIRouter(prefix="/api")


# ------------------------------------------------------------------ registry
@router.get("/ops")
def list_ops() -> dict[str, Any]:
    ops = [s.to_json() for s in REGISTRY.all()]
    return {
        "ok": True, "count": len(ops), "ops": ops,
        "quick": [s.id for s in REGISTRY.quick()],
        "categories": {k: [s.id for s in v] for k, v in REGISTRY.by_category().items()},
        "hsv_presets": HSV_PRESETS,
        "pipelines": PIPELINES,
        "cv_version": cv2.__version__,
    }


# -------------------------------------------------------------------- images
@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...),
                 session_id: str | None = Query(None)) -> dict[str, Any]:
    name = (file.filename or "frame.png").lower()
    if not any(name.endswith(e) for e in settings.allowed_upload_ext):
        raise UnsupportedMediaError(f"Unsupported file type: {name}")
    blob = await file.read()
    if len(blob) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds {settings.max_upload_mb} MB")
    arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise UnsupportedMediaError("cv2.imdecode could not read this file")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = as_bgr(arr)
    m = settings.ingest_max_side
    if m and max(arr.shape[:2]) > m:
        r = m / max(arr.shape[:2])
        arr = cv2.resize(arr, (int(arr.shape[1] * r), int(arr.shape[0] * r)),
                         interpolation=cv2.INTER_AREA)
    s = STORE.session(session_id)
    s.images.clear(); s.history.clear(); s.cursor = -1; s.carry.clear()
    iid = STORE.put(s, arr)
    from app.services.store import Entry
    STORE.commit(s, Entry(op_id="source", label=f"Loaded {file.filename}", params={},
                          image_id=iid, summary={"Source": file.filename or "upload"}))
    return {"ok": True, "session_id": s.id, "image_id": iid,
            "info": STORE.inspect(arr), "filename": file.filename,
            "upload_bytes": len(blob)}


@router.get("/img/{session_id}/{image_id}")
def render(session_id: str, image_id: str, max: int = Query(0, ge=0, le=8000)) -> Response:
    s = STORE.session(session_id)
    data, mime = STORE.render(s, image_id, max)
    return Response(data, media_type=mime,
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/state/{session_id}")
def state(session_id: str) -> dict[str, Any]:
    s = STORE.session(session_id)
    if s.cursor < 0:
        return {"ok": True, "session_id": s.id, "empty": True}
    cur = s.current
    return {"ok": True, "session_id": s.id, "empty": False, "cursor": s.cursor,
            "image_id": cur.image_id, "info": STORE.inspect(STORE.img(s, cur.image_id)),
            "source_info": STORE.inspect(STORE.img(s, s.history[0].image_id)),
            "meta": s.meta,
            "history": [{"i": i, "op": e.op_id, "label": e.label, "params": e.params,
                         "image_id": e.image_id, "summary": e.summary,
                         "metrics": e.metrics} for i, e in enumerate(s.history)]}


# ----------------------------------------------------------------- execution
@router.post("/execute")
async def do_execute(request: Request) -> dict[str, Any]:
    body = await request.json()
    s = STORE.session(body.get("session_id"))
    return execute(STORE, s, body["op_id"], body.get("params"),
                   image_id=body.get("image_id"), preview=bool(body.get("preview")),
                   image2_id=body.get("image2_id"))


@router.post("/compare")
async def compare(request: Request) -> dict[str, Any]:
    """Run the same op twice with different params and diff the results."""
    body = await request.json()
    s = STORE.session(body.get("session_id"))
    op = body["op_id"]
    a = execute(STORE, s, op, body.get("params_a"), preview=True)
    b = execute(STORE, s, op, body.get("params_b"), preview=True)
    ia, ib = STORE.img(s, a["image_id"]), STORE.img(s, b["image_id"])
    if ia.shape == ib.shape and ia.dtype == ib.dtype:
        diff = cv2.absdiff(ia, ib)
        d = as_gray(diff)
        stats = {"changed_px": int(cv2.countNonZero(d)),
                 "max_delta": int(d.max()), "mean_delta": round(float(d.mean()), 4),
                 "identical": bool(d.max() == 0)}
        diff_id = STORE.put(s, cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=4),
                                                 cv2.COLORMAP_INFERNO))
    else:
        stats, diff_id = {"identical": False, "note": "different shapes"}, None
    return {"ok": True, "a": a, "b": b, "diff_image_id": diff_id, "diff": stats}


@router.post("/history/{action}")
async def history(action: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    s = STORE.session(body.get("session_id"))
    fn = {"undo": STORE.undo, "redo": STORE.redo, "reset": STORE.reset}.get(action)
    if fn is None:
        raise ValidationError("action must be undo, redo or reset")
    e = fn(s)
    return {"ok": True, "cursor": s.cursor, "image_id": e.image_id,
            "info": STORE.inspect(STORE.img(s, e.image_id)),
            "label": e.label, "history_len": len(s.history)}


@router.get("/code/{session_id}", response_class=PlainTextResponse)
def code(session_id: str) -> str:
    return generate_script(STORE.session(session_id))


# ---------------------------------------------------------- pixel inspection
@router.get("/pixel/{session_id}/{image_id}")
def pixel(session_id: str, image_id: str, x: int, y: int,
          n: int = Query(3, ge=1, le=9)) -> dict[str, Any]:
    s = STORE.session(session_id)
    arr = STORE.img(s, image_id)
    h, w = arr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValidationError(f"({x}, {y}) is outside {w}×{h}")
    px = arr[y, x]
    out: dict[str, Any] = {"ok": True, "x": x, "y": y, "width": w, "height": h,
                           "norm_x": round(x / w, 6), "norm_y": round(y / h, 6)}
    if arr.ndim == 2:
        out["intensity"] = int(px)
        out["gray"] = int(px)
    else:
        b, g, r = (int(v) for v in px[:3])
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0][0]
        out.update({"b": b, "g": g, "r": r, "rgb": [r, g, b],
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "hsv": [int(v) for v in hsv], "lab": [int(v) for v in lab],
                    "gray": int(round(0.114 * b + 0.587 * g + 0.299 * r))})
    half = n // 2
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    patch = arr[y0:y1, x0:x1]
    out["neighbourhood"] = {
        "size": n, "origin": [x0, y0],
        "values": (as_gray(patch) if patch.ndim == 3 else patch).tolist(),
        "colors": [[f"#{int(p[2]):02x}{int(p[1]):02x}{int(p[0]):02x}" for p in row]
                   for row in patch] if patch.ndim == 3 else None,
    }
    return out


@router.get("/histogram/{session_id}/{image_id}")
def histogram(session_id: str, image_id: str, bins: int = Query(64, ge=8, le=256)):
    s = STORE.session(session_id)
    arr = STORE.img(s, image_id)
    from app.ops.helpers import norm_hist
    out: dict[str, Any] = {"ok": True, "bins": bins}
    if arr.ndim == 2:
        out["gray"] = norm_hist(cv2.calcHist([arr], [0], None, [bins], [0, 256]))
    else:
        names = ["b", "g", "r"]
        for i, nm in enumerate(names[: arr.shape[2]]):
            out[nm] = norm_hist(cv2.calcHist([arr], [i], None, [bins], [0, 256]))
        out["gray"] = norm_hist(cv2.calcHist([as_gray(arr)], [0], None, [bins], [0, 256]))
    return out


# ------------------------------------------------------------------ ROI math
@router.post("/roi/analyze")
async def roi_analyze(request: Request) -> dict[str, Any]:
    """Geometry + code for a rect / polygon / line / circle ROI."""
    b = await request.json()
    w, h = int(b["width"]), int(b["height"])
    kind = b.get("kind", "polygon")
    pts = [[float(p[0]), float(p[1])] for p in b.get("points", [])]
    nx = lambda v: round(v / w, 6)   # noqa: E731
    ny = lambda v: round(v / h, 6)   # noqa: E731
    out: dict[str, Any] = {"ok": True, "kind": kind, "width": w, "height": h,
                           "pixel": [[int(p[0]), int(p[1])] for p in pts],
                           "normalized": [[nx(p[0]), ny(p[1])] for p in pts]}

    if kind == "rect" and len(pts) >= 2:
        x1, y1 = min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1])
        x2, y2 = max(pts[0][0], pts[1][0]), max(pts[0][1], pts[1][1])
        bw, bh = x2 - x1, y2 - y1
        out["rect"] = {"x": int(x1), "y": int(y1), "x2": int(x2), "y2": int(y2),
                       "width": int(bw), "height": int(bh),
                       "cx": int(x1 + bw / 2), "cy": int(y1 + bh / 2),
                       "area": int(bw * bh), "aspect": round(bw / bh, 4) if bh else 0}
        out["rect_norm"] = {"x": nx(x1), "y": ny(y1), "x2": nx(x2), "y2": ny(y2),
                            "w": nx(bw), "h": ny(bh),
                            "cx": nx(x1 + bw / 2), "cy": ny(y1 + bh / 2)}
        out["code"] = {
            "crop": f"x1, y1, x2, y2 = {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}\n"
                    "crop = img[y1:y2, x1:x2]",
            "draw": f"cv2.rectangle(img, ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)}), "
                    "(0, 255, 0), 2)",
            "yolo": f"# class cx cy w h (normalised)\n0 {nx(x1+bw/2)} {ny(y1+bh/2)} "
                    f"{nx(bw)} {ny(bh)}",
        }
    elif kind == "line" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(-dy, dx)))
        out["line"] = {"start": [int(x1), int(y1)], "end": [int(x2), int(y2)],
                       "length": round(length, 3), "angle_deg": round(angle, 3),
                       "slope": round(dy / dx, 6) if abs(dx) > 1e-9 else None,
                       "cx": int((x1 + x2) / 2), "cy": int((y1 + y2) / 2)}
        out["line_norm"] = {"start": [nx(x1), ny(y1)], "end": [nx(x2), ny(y2)]}
        out["code"] = {
            "draw": f"line_start = ({int(x1)}, {int(y1)})\nline_end = ({int(x2)}, {int(y2)})\n"
                    "cv2.line(img, line_start, line_end, (0, 255, 0), 2)",
            "normalized": f"line = [[{nx(x1)}, {ny(y1)}], [{nx(x2)}, {ny(y2)}]]",
        }
    elif kind == "circle" and len(pts) >= 2:
        (cx, cy), (px, py) = pts[0], pts[1]
        r = float(np.hypot(px - cx, py - cy))
        out["circle"] = {"cx": int(cx), "cy": int(cy), "r": int(r),
                         "area": round(float(np.pi * r * r), 2)}
        out["circle_norm"] = {"cx": nx(cx), "cy": ny(cy), "r": round(r / w, 6)}
        out["code"] = {"draw": f"cv2.circle(img, ({int(cx)}, {int(cy)}), {int(r)}, "
                               "(0, 255, 0), 2)",
                       "mask": f"mask = np.zeros(img.shape[:2], dtype=np.uint8)\n"
                               f"cv2.circle(mask, ({int(cx)}, {int(cy)}), {int(r)}, 255, -1)"}
    elif len(pts) >= 3:
        cnt = np.array([[int(p[0]), int(p[1])] for p in pts], np.int32).reshape(-1, 1, 2)
        area = float(cv2.contourArea(cnt))
        peri = float(cv2.arcLength(cnt, True))
        x, y, bw, bh = (int(v) for v in cv2.boundingRect(cnt))
        m = cv2.moments(cnt)
        cx = m["m10"] / m["m00"] if abs(m["m00"]) > 1e-9 else x + bw / 2
        cy = m["m01"] / m["m00"] if abs(m["m00"]) > 1e-9 else y + bh / 2
        out["polygon"] = {"num_points": len(pts), "area": round(area, 2),
                          "perimeter": round(peri, 2),
                          "bounding_rect": {"x": x, "y": y, "w": bw, "h": bh,
                                            "x2": x + bw, "y2": y + bh},
                          "centroid": [round(cx, 2), round(cy, 2)],
                          "convex": bool(cv2.isContourConvex(cnt))}
        norm = [[nx(p[0]), ny(p[1])] for p in pts]
        py_pix = "roi = np.array([\n" + "".join(
            f"    [{int(p[0])}, {int(p[1])}],\n" for p in pts) + "], dtype=np.int32)"
        out["code"] = {
            "pixel_list": "roi = [\n" + "".join(f"    [{int(p[0])}, {int(p[1])}],\n"
                                                for p in pts) + "]",
            "normalized_list": "roi = [\n" + "".join(f"    [{v[0]}, {v[1]}],\n"
                                                     for v in norm) + "]",
            "numpy": py_pix,
            "mask": py_pix + "\nmask = np.zeros(img.shape[:2], dtype=np.uint8)\n"
                             "cv2.fillPoly(mask, [roi], 255)\n"
                             "masked = cv2.bitwise_and(img, img, mask=mask)",
            "draw": "cv2.polylines(img, [roi], True, (0, 255, 0), 2)",
            "json": json.dumps({"pixel": [[int(p[0]), int(p[1])] for p in pts],
                                "normalized": norm}, indent=2),
        }
    return out


@router.post("/roi/mask")
async def roi_mask(request: Request) -> dict[str, Any]:
    """Build a polygon/rect/circle mask and apply it to the current image."""
    b = await request.json()
    s = STORE.session(b.get("session_id"))
    arr = STORE.img(s, b.get("image_id") or s.current.image_id)
    h, w = arr.shape[:2]
    kind = b.get("kind", "polygon")
    pts = [[int(p[0]), int(p[1])] for p in b.get("points", [])]
    mask = np.zeros((h, w), np.uint8)
    if kind == "rect" and len(pts) >= 2:
        cv2.rectangle(mask, tuple(pts[0]), tuple(pts[1]), 255, -1)
        code = f"cv2.rectangle(mask, {tuple(pts[0])}, {tuple(pts[1])}, 255, -1)"
    elif kind == "circle" and len(pts) >= 2:
        r = int(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
        cv2.circle(mask, tuple(pts[0]), r, 255, -1)
        code = f"cv2.circle(mask, {tuple(pts[0])}, {r}, 255, -1)"
    else:
        if len(pts) < 3:
            raise ValidationError("A polygon mask needs at least 3 points")
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        code = "cv2.fillPoly(mask, [roi], 255)"
    masked = cv2.bitwise_and(arr, arr, mask=mask)
    if bool(b.get("commit")):
        from app.services.store import Entry
        out_id = STORE.put(s, masked)
        STORE.commit(s, Entry(
            op_id="roi_mask", label="Mask ROI", params={"kind": kind, "points": pts},
            image_id=out_id,
            code=["roi = np.array(" + str(pts) + ", dtype=np.int32)",
                  "mask = np.zeros(img.shape[:2], dtype=np.uint8)", code,
                  "img = cv2.bitwise_and(img, img, mask=mask)"],
            imports=["import numpy as np"],
            summary={"ROI kind": kind, "Points": len(pts)}))
    else:
        out_id = STORE.put(s, masked)
    nz = int(cv2.countNonZero(mask))
    return {"ok": True, "mask_image_id": STORE.put(s, mask), "masked_image_id": out_id,
            "coverage_pct": round(100.0 * nz / (w * h), 3), "nonzero": nz,
            "cursor": s.cursor}


@router.post("/coords")
async def coords(request: Request) -> dict[str, Any]:
    """Bidirectional pixel <-> normalised converter."""
    b = await request.json()
    w, h = int(b["width"]), int(b["height"])
    if w <= 0 or h <= 0:
        raise ValidationError("width and height must be positive")
    out: dict[str, Any] = {"ok": True, "width": w, "height": h}
    if b.get("direction", "to_norm") == "to_norm":
        x, y = float(b.get("x", 0)), float(b.get("y", 0))
        out.update({"pixel": [int(x), int(y)],
                    "normalized": [round(x / w, 6), round(y / h, 6)]})
    else:
        nx_, ny_ = float(b.get("nx", 0)), float(b.get("ny", 0))
        out.update({"normalized": [nx_, ny_],
                    "pixel": [int(round(nx_ * w)), int(round(ny_ * h))]})
    return out


@router.post("/camera/meta")
async def camera_meta(request: Request) -> dict[str, Any]:
    b = await request.json()
    s = STORE.session(b.get("session_id"))
    s.meta.update({k: b.get(k) for k in ("camera_name", "fps", "resolution", "source")})
    return {"ok": True, "meta": s.meta}


# ---------------------------------------------------------------- pipelines
PIPELINES: list[dict[str, Any]] = [
    {"id": "preproc", "label": "Basic detection preprocessing",
     "desc": "BGR → Gray → GaussianBlur → Canny",
     "steps": [{"op": "bgr2gray"}, {"op": "gaussian_blur", "params": {"ksize": [5, 5]}},
               {"op": "canny", "params": {"threshold1": 50, "threshold2": 150}}]},
    {"id": "binary_mask", "label": "Binary colour mask",
     "desc": "BGR → HSV → InRange → Open → Close",
     "steps": [{"op": "in_range"}, {"op": "morph_open", "params": {"ksize": [5, 5]}},
               {"op": "morph_close", "params": {"ksize": [5, 5]}}]},
    {"id": "contour_prep", "label": "Contour detection prep",
     "desc": "Gray → Blur → Threshold",
     "steps": [{"op": "bgr2gray"}, {"op": "gaussian_blur", "params": {"ksize": [5, 5]}},
               {"op": "threshold", "params": {"thresh": 127, "auto": "THRESH_OTSU"}}]},
    {"id": "edges", "label": "Edge detection",
     "desc": "Gray → GaussianBlur → Canny",
     "steps": [{"op": "bgr2gray"}, {"op": "gaussian_blur", "params": {"ksize": [3, 3]}},
               {"op": "canny"}]},
    {"id": "yolo_prep", "label": "YOLO input prep",
     "desc": "Letterbox 640 → BGR→RGB",
     "steps": [{"op": "letterbox", "params": {"size": 640}}, {"op": "bgr2rgb"}]},
    {"id": "color_detect", "label": "Colour object detection",
     "desc": "HSV → InRange → Morphology",
     "steps": [{"op": "in_range"}, {"op": "morphology_ex",
                                    "params": {"op": "MORPH_CLOSE", "ksize": [7, 7]}}]},
]


@router.post("/pipeline/{pipeline_id}")
async def run_pipeline(pipeline_id: str, request: Request) -> dict[str, Any]:
    b = await request.json()
    s = STORE.session(b.get("session_id"))
    spec = next((p for p in PIPELINES if p["id"] == pipeline_id), None)
    if spec is None:
        raise ValidationError(f"Unknown pipeline '{pipeline_id}'")
    results = []
    for step in spec["steps"]:
        results.append(execute(STORE, s, step["op"], step.get("params")))
    return {"ok": True, "pipeline": spec, "steps": results,
            "image_id": results[-1]["image_id"] if results else None,
            "cursor": s.cursor}
