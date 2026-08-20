"""Small standard-library client for the DE-owned FastAPI service.

Keeping HTTP access outside the Streamlit module makes the API contract easy to
test without starting a browser or importing Streamlit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiError(RuntimeError):
    """A network, HTTP, or malformed-response error safe to show in the UI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _decode_json(raw: bytes, *, endpoint: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(f"{endpoint} trả về dữ liệu không phải JSON hợp lệ") from exc
    if not isinstance(value, dict):
        raise ApiError(f"{endpoint} phải trả về một JSON object")
    return value


@dataclass(frozen=True)
class LzdApiClient:
    base_url: str = "http://localhost:18000"
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        normalized = self.base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("API URL phải bắt đầu bằng http:// hoặc https://")
        object.__setattr__(self, "base_url", normalized)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not path.startswith("/"):
            raise ValueError("API path phải bắt đầu bằng /")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return _decode_json(response.read(), endpoint=path)
        except HTTPError as exc:
            raw = exc.read()
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = {"detail": raw.decode("utf-8", errors="replace")}
            detail = parsed.get("detail", parsed) if isinstance(parsed, dict) else parsed
            raise ApiError(
                f"API trả HTTP {exc.code} tại {path}",
                status_code=exc.code,
                detail=detail,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ApiError(f"Không kết nối được {self.base_url}{path}: {reason}") from exc

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/ready")

    def store_info(self) -> dict[str, Any]:
        return self._request("GET", "/store/info")

    def campaign_decide(
        self,
        user_ids: Sequence[str],
        *,
        budget: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/campaign/decide",
            payload={
                "user_ids": list(user_ids),
                "budget": budget,
                "context": {},
            },
        )

    def decide_debug(self, user_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/decide",
            payload={"user_id": user_id, "context": {}, "debug": True},
        )

