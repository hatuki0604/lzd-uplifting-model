"""Translate inference API fields into stable BA-facing labels."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionView:
    label: str
    short_label: str
    icon: str
    background: str


ACTION_VIEWS: dict[str, ActionView] = {
    "SEND_VOUCHER": ActionView(
        "Gửi voucher", "Gửi", "🟢", "#dcfce7"
    ),
    "WAIT_FOR_INTENT": ActionView(
        "Chờ thêm ý định mua", "Chờ", "🟡", "#fef3c7"
    ),
    "SUPPRESS_ALREADY_PURCHASED": ActionView(
        "Không gửi vì khách đã mua", "Đã mua", "🔴", "#fee2e2"
    ),
    "NOT_SELECTED_BUDGET": ActionView(
        "Ngoài nhóm được xét ưu tiên", "Ngoài Top-K", "⚪", "#f3f4f6"
    ),
    "SKIP_NON_POSITIVE_UPLIFT": ActionView(
        "Bỏ qua vì voucher không được dự đoán mang lại hiệu quả",
        "Uplift không dương",
        "⚫",
        "#e5e7eb",
    ),
    "NO_MODEL_SCORE": ActionView(
        "Không chấm được điểm", "Không có điểm", "⚠️", "#ffedd5"
    ),
}

UNKNOWN_ACTION = ActionView(
    "Trạng thái chưa được hỗ trợ", "Không xác định", "❔", "#f3f4f6"
)


def action_view(action: str) -> ActionView:
    return ACTION_VIEWS.get(action, UNKNOWN_ACTION)


def action_display(action: str) -> str:
    view = action_view(action)
    return f"{view.icon} {view.label}"


def format_uplift_pp(value: Any) -> str:
    """Format the API's already-converted percentage-point field.

    Do not multiply by 100 here: `/campaign/decide` already returns
    `uplift_percentage_point`.
    """
    if value is None:
        return "Không có điểm"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Không có điểm"
    return f"{number:+.3f}".replace(".", ",") + " điểm %"


def format_integer_vi(value: Any) -> str:
    return f"{int(value):,}".replace(",", ".")


def format_decimal_vi(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class RealtimeFreshness:
    status: str
    age_seconds: float | None

    @property
    def is_fresh(self) -> bool:
        return self.status == "fresh"


def realtime_freshness(
    row: Mapping[str, Any],
    *,
    now_epoch: float | None = None,
    max_age_seconds: float = 3600.0,
) -> RealtimeFreshness:
    realtime = row.get("realtime")
    if not isinstance(realtime, Mapping):
        return RealtimeFreshness("missing", None)
    raw = realtime.get("rt_last_event_ts")
    try:
        timestamp = float(raw)
    except (TypeError, ValueError):
        return RealtimeFreshness("missing", None)
    if timestamp <= 0:
        return RealtimeFreshness("missing", None)

    now = time.time() if now_epoch is None else now_epoch
    age = now - timestamp
    if age < -300:
        return RealtimeFreshness("future", age)
    if age > max_age_seconds:
        return RealtimeFreshness("stale", age)
    return RealtimeFreshness("fresh", max(age, 0.0))


def format_age(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "Không có tín hiệu"
    if age_seconds < 0:
        return "Sai lệch đồng hồ"
    minutes = int(age_seconds // 60)
    if minutes < 1:
        return "Vừa cập nhật"
    if minutes < 60:
        return f"{minutes} phút trước"
    return f"{minutes // 60} giờ {minutes % 60} phút trước"
