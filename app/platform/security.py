"""API key check for /v1 routes. Constant-time comparison; several keys allow rotation."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.platform.dependencies import get_settings
from config.settings import Settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    key: Annotated[str | None, Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    accepted = settings.api_key_list
    if not accepted:  # auth disabled (local)
        return
    presented = (key or "").encode()
    if not any(secrets.compare_digest(presented, candidate.encode()) for candidate in accepted):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key")
