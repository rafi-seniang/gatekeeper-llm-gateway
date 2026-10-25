"""The Lab backend contract."""

from pydantic import BaseModel


class ServiceMeta(BaseModel):
    """What the lab frontend's lobby shows about this backend (the Lab backend contract)."""

    id: str
    name: str
    summary: str
    version: str
    engine: str
    own_ui_path: str
    docs_path: str
