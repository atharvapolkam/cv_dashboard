"""FastAPI application factory."""
from __future__ import annotations

import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.config import settings
from app.core.errors import CVDashboardError
from app.core.logging import get_logger, setup_logging
from app.ops.registry import REGISTRY, load_catalog

setup_logging(settings.debug)
log = get_logger("cvd")


def create_app() -> FastAPI:
    n = load_catalog()
    log.info("registered %d operations (OpenCV %s)", n, cv2.__version__)

    app = FastAPI(title=settings.app_name, version=settings.version,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    templates = Jinja2Templates(directory=str(settings.templates_dir))

    @app.exception_handler(CVDashboardError)
    async def _app_error(_: Request, exc: CVDashboardError) -> JSONResponse:
        return JSONResponse(exc.payload(), status_code=exc.status)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "version": settings.version, "cv": cv2.__version__,
                "ops": len(REGISTRY)}

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html",
                                          {"version": settings.version,
                                           "cv_version": cv2.__version__,
                                           "op_count": len(REGISTRY)})

    return app


app = create_app()
