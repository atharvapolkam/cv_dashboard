"""Typed application errors mapped to HTTP responses by ``app.main``."""

from __future__ import annotations

from typing import Any


class CVDashboardError(Exception):
    """Base class. ``status`` is the HTTP status the API layer will emit."""

    status: int = 400
    code: str = "error"

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "code": self.code, "message": self.message}
        if self.detail is not None:
            out["detail"] = self.detail
        return out


class NotFoundError(CVDashboardError):
    status = 404
    code = "not_found"


class ValidationError(CVDashboardError):
    status = 422
    code = "validation_error"


class UnsupportedMediaError(CVDashboardError):
    status = 415
    code = "unsupported_media"


class PayloadTooLargeError(CVDashboardError):
    status = 413
    code = "payload_too_large"


class OperationError(CVDashboardError):
    """Raised when an OpenCV op cannot run on the given input.

    Carries a human-readable hint so the UI can tell the user *why*
    (e.g. "Canny needs a single-channel image; run BGR -> Gray first").
    """

    status = 400
    code = "operation_error"

    def __init__(self, message: str, *, hint: str | None = None, detail: Any = None) -> None:
        super().__init__(message, detail=detail)
        self.hint = hint

    def payload(self) -> dict[str, Any]:
        out = super().payload()
        if self.hint:
            out["hint"] = self.hint
        return out


class ConflictError(CVDashboardError):
    status = 409
    code = "conflict"
