"""Session store: images, undo/redo history, render cache. Disk-free (RAM, LRU)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.core.errors import NotFoundError
from app.ops.helpers import as_bgr, image_stats


@dataclass
class Entry:
    """One committed history step. index 0 is the uploaded source."""
    op_id: str
    label: str
    params: dict[str, Any]
    image_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    code: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    id: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    history: list[Entry] = field(default_factory=list)
    cursor: int = -1
    carry: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)   # camera name / fps / resolution
    touched: float = field(default_factory=time.time)

    @property
    def current(self) -> Entry:
        if self.cursor < 0:
            raise NotFoundError("No image in this session yet — upload one first.")
        return self.history[self.cursor]


class Store:
    def __init__(self) -> None:
        self._s: dict[str, Session] = {}
        self._render: dict[str, bytes] = {}

    def session(self, sid: str | None) -> Session:
        sid = sid or uuid.uuid4().hex
        s = self._s.get(sid)
        if s is None:
            self._gc()
            s = self._s[sid] = Session(id=sid)
        s.touched = time.time()
        return s

    def _gc(self) -> None:
        ttl = settings.session_ttl_minutes * 60
        now = time.time()
        for k, v in list(self._s.items()):
            if now - v.touched > ttl:
                self._s.pop(k, None)

    # -------------------------------------------------------------- images
    def put(self, s: Session, arr: np.ndarray) -> str:
        iid = uuid.uuid4().hex[:16]
        s.images[iid] = arr
        if len(s.images) > settings.max_images_per_session:
            keep = {e.image_id for e in s.history}
            for k in list(s.images):
                if k not in keep and len(s.images) > settings.max_images_per_session // 2:
                    s.images.pop(k, None)
        return iid

    def img(self, s: Session, iid: str) -> np.ndarray:
        a = s.images.get(iid)
        if a is None:
            raise NotFoundError(f"Image '{iid}' is not in this session")
        return a

    # ------------------------------------------------------------- history
    def commit(self, s: Session, e: Entry) -> Entry:
        del s.history[s.cursor + 1:]          # redo tail is discarded on new work
        s.history.append(e)
        if len(s.history) > settings.max_history_per_session:
            s.history.pop(1 if len(s.history) > 1 else 0)
        s.cursor = len(s.history) - 1
        return e

    def undo(self, s: Session) -> Entry:
        if s.cursor > 0:
            s.cursor -= 1
        return s.current

    def redo(self, s: Session) -> Entry:
        if s.cursor < len(s.history) - 1:
            s.cursor += 1
        return s.current

    def reset(self, s: Session) -> Entry:
        s.cursor = 0
        del s.history[1:]
        return s.current

    # -------------------------------------------------------------- render
    def render(self, s: Session, iid: str, max_side: int = 0) -> tuple[bytes, str]:
        key = f"{s.id}:{iid}:{max_side}"
        if key in self._render:
            return self._render[key], "image/webp"
        arr = self.img(s, iid)
        vis = as_bgr(arr) if arr.ndim == 3 and arr.shape[2] not in (1, 3) else arr
        if vis.dtype != np.uint8:
            vis = cv2.normalize(vis.astype(np.float32), None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
        if max_side:
            h, w = vis.shape[:2]
            if max(h, w) > max_side:
                r = max_side / max(h, w)
                vis = cv2.resize(vis, (max(1, int(w * r)), max(1, int(h * r))),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".webp", vis, [cv2.IMWRITE_WEBP_QUALITY,
                                              settings.render_quality])
        if not ok:
            ok, buf = cv2.imencode(".png", vis)
        data = buf.tobytes()
        if len(self._render) > settings.render_cache_entries:
            self._render.clear()
        self._render[key] = data
        return data, "image/webp" if ok else "image/png"

    def inspect(self, arr: np.ndarray) -> dict[str, Any]:
        h, w = arr.shape[:2]
        ch = 1 if arr.ndim == 2 else int(arr.shape[2])
        st = image_stats(arr)
        return {"width": w, "height": h, "channels": ch, "dtype": str(arr.dtype),
                "aspect_ratio": round(w / h, 6) if h else 0,
                "megapixels": round(w * h / 1e6, 3),
                "nbytes": int(arr.nbytes), "shape": list(arr.shape), **st}


STORE = Store()
