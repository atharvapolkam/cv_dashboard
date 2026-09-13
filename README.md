# CV Dashboard — interactive OpenCV testing lab

FastAPI + vanilla JS dashboard for real CV development: ROI/coordinate work, preprocessing
tuning, and exact Python code generation. Verified on OpenCV 5.0 / Python 3.14 in a local venv.

## Run

```bash
./run.sh
```
Then open http://localhost:8000 — API docs at `/api/docs`, health at `/health`.

Manual:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

```
app/
  config.py            all tunables, env-overridable via CVD_*
  core/                typed errors -> HTTP, logging, Timings stopwatch
  ops/
    params.py          ParamSpec — validates server-side AND renders the UI widget
    constants.py       OpenCV constant tables (emitted by name in generated code)
    helpers.py         input guards + OpenCV-5 return-shape normalisers
    registry.py        @register -> OpSpec; the single extension point
    catalog/           one module per topic — 40 operations registered
  services/
    store.py           sessions, image store, undo/redo history, render cache
    executor.py        guards, timing, preview downscale, script generation
  api/routes.py        REST surface
static/ templates/     frontend, no build step
```

### Adding an operation
One decorated function. UI widgets, validation, search, favourites, history and codegen all
derive from the declaration — there is no second place to edit:

```python
@register(id="sobel_x", cv="cv2.Sobel", label="Sobel X", category=Category.EDGES,
          tier=2, cost=Cost.MEDIUM, params=[p_odd("ksize", "ksize", 3)])
def _sobel_x(ctx):
    out = cv2.Sobel(ctx.image, cv2.CV_64F, 1, 0, ksize=ctx.p["ksize"])
    return OpResult(image=cv2.convertScaleAbs(out),
                    code=[f'g = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize={ctx.p["ksize"]})',
                          "img = cv2.convertScaleAbs(g)"])
```
Then import the module in `app/ops/catalog/__init__.py`.

### Live-preview strategy
Every op declares a `cost`. The frontend debounces by class (cheap 40 ms, medium 120 ms,
expensive 260 ms), aborts in-flight requests, and drops stale responses with a generation
counter — so dragging a Canny slider never queues 100 round-trips. For `expensive` ops the
backend also runs the preview on a downscaled copy (`CVD_PREVIEW_MAX_SIDE`, default 900 px) and
reports the `scale` it used. Previews are never committed; only **Apply** writes to history,
which keeps undo/redo exact.

### OpenCV 5 notes
OpenCV 5 changed several return shapes vs 4.x (`HoughLinesP` → `(N,4)`, `calcHist` → `(bins,)`,
`convexityDefects` → `(N,4)`). `ops/helpers.py` has `norm_*` functions that accept both, so
catalog code never branches on version.

## Implemented
Quick-test strip · function search · favourites + recents (localStorage) · collapsible
categories by tier · image inspection (w/h/channels/dtype/min/max/mean/std/aspect/MP) ·
resize with scale readout and distortion warning · YOLO letterbox with un-map maths · pixel and
normalised crop · full cvtColor set + one-click conversions · InRange with 13 HSV presets, hue
wrap for red, and mask/masked/HSV thumbnails · threshold (+Otsu/Triangle) · adaptive threshold ·
box/Gaussian/median/bilateral/filter2D · erode/dilate/morphologyEx (+kernel visualiser) ·
Canny/Sobel/Scharr/Laplacian/gradient-magnitude · channel split/merge/HSV viewer · ROI drawing
(rect/polygon/line/circle) with pixel + normalised coordinates, area/perimeter/centroid/bbox,
length/angle/slope for lines, and copy-as Python-list / normalised / `np.array` / fillPoly-mask /
JSON · polygon mask applied via `fillPoly` + `bitwise_and` · coordinate converter both ways ·
pixel inspector with BGR/RGB/HSV/LAB/gray and 5×5 neighbourhood · A/B compare with difference
heat-map · 6 pipeline presets · undo/redo/reset + history list · what-happened and performance
panels · generated runnable Python script with `.py`/`.png`/`.json` download.

## Deferred (registry slots ready)
Contour table + area filtering, drawing ops, bitwise/arithmetic, perspective/affine warp, Hough
lines/circles, template matching, histogram/CLAHE charts, NumPy mask builders, RTSP source. Each
is one catalog module following the pattern above. `services/store.py` already carries a `carry`
dict for contour hand-off between ops, and `config.enable_rtsp` reserves the source flag.
