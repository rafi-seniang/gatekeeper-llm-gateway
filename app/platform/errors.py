"""Domain errors and their HTTP mapping as RFC 9457 problem details (application/problem+json)."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config.logging import request_id

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base for errors the caller may see. `detail` must be safe to show."""

    status = 500
    title = "Internal error"

    def __init__(self, detail: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra or {}


class UpstreamUnavailable(AppError):
    """A dependency (the upstream LLM, the decision endpoint) failed; the request may be retried later."""

    status, title = 503, "Upstream unavailable"


class RequestBlocked(AppError):
    """The guard judged the request a prompt injection."""

    status, title = 403, "Request blocked"


class RateLimited(AppError):
    status, title = 429, "Too many requests"

    def __init__(self, detail: str, retry_after_seconds: int) -> None:
        super().__init__(detail)
        self.retry_after_seconds = retry_after_seconds


def problem(
    status: int, title: str, detail: str, request: Request, extra: dict[str, Any] | None = None
) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "request_id": request_id.get(),
        **(extra or {}),
    }
    return JSONResponse(body, status_code=status, media_type="application/problem+json")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        response = problem(exc.status, exc.title, exc.detail, request, exc.extra)
        if isinstance(exc, RateLimited):
            response.headers["Retry-After"] = str(exc.retry_after_seconds)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return problem(422, "Invalid request", detail, request)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")  # the detail stays in the log, not in the response
        return problem(500, "Internal error", "Unexpected error. Quote the request id.", request)
