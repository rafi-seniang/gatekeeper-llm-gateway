"""The app factory: settings, logging, middleware, error handlers, then one router per feature."""

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import VERSION
from app.features.gate import router as gate
from app.features.health import router as health
from app.platform.dependencies import enforce_rate_limit
from app.platform.errors import register_error_handlers
from app.platform.middleware import register_middleware
from app.platform.resources import lifespan
from app.platform.security import require_api_key
from config.logging import configure_logging
from config.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Gatekeeper",
        version=VERSION,
        lifespan=lifespan(settings),
        docs_url=None if settings.environment == "production" else "/docs",
    )
    app.state.settings = settings
    register_middleware(app)
    register_error_handlers(app)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    # Key first, then the rate limit, so an unknown caller never spends a known caller's budget
    api = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
    api.include_router(gate.router)
    app.include_router(health.router)
    app.include_router(api)
    return app
