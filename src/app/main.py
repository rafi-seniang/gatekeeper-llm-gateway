"""App factory. `uvicorn app.main:app` for the default settings, create_app(settings) in tests."""

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import VERSION
from app.api.middleware import register_middleware
from app.api.routes import chat, health
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.resources import lifespan
from app.core.security import require_api_key


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

    api = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
    api.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(api)
    return app


app = create_app()
