"""Load and validate the committed offline RCT policy evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_POLICY_PATH = Path(__file__).with_name("offline_policy.json")
REQUIRED_SCENARIO_FIELDS = {
    "target_percent",
    "selected_customers",
    "measured_uplift_pp",
    "uplift_ci_95_low_pp",
    "uplift_ci_95_high_pp",
    "n_treatment",
    "n_control",
    "incremental_orders",
    "effect_retained_percent",
    "voucher_saved_percent",
    "campaign_cvr_percent",
}


class PolicyEvidenceError(ValueError):
    """Raised when the committed BA evidence is incomplete or inconsistent."""


def load_policy_evidence(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyEvidenceError(f"Không đọc được bằng chứng policy: {exc}") from exc

    if not isinstance(payload, dict):
        raise PolicyEvidenceError("Bằng chứng policy phải là một JSON object")
    source = payload.get("source")
    baseline = payload.get("baseline")
    scenarios = payload.get("scenarios")
    if not isinstance(source, dict) or not isinstance(baseline, dict):
        raise PolicyEvidenceError("Thiếu source hoặc baseline")
    if source.get("dataset") != "rct_holdout" or not source.get("randomized_treatment"):
        raise PolicyEvidenceError("Bằng chứng cấp nhóm phải đến từ RCT holdout")
    try:
        population = int(source["population"])
        broadcast_orders = int(baseline["broadcast_incremental_orders"])
        default_view = int(payload["default_view_target_percent"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyEvidenceError("Population, baseline hoặc default view không hợp lệ") from exc
    if population <= 0 or not isinstance(scenarios, list) or not scenarios:
        raise PolicyEvidenceError("Population/scenarios không hợp lệ")

    seen: set[int] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not REQUIRED_SCENARIO_FIELDS <= scenario.keys():
            raise PolicyEvidenceError("Scenario policy thiếu trường bắt buộc")
        try:
            target = int(scenario["target_percent"])
            selected = int(scenario["selected_customers"])
            uplift_pp = float(scenario["measured_uplift_pp"])
            ci_low = float(scenario["uplift_ci_95_low_pp"])
            ci_high = float(scenario["uplift_ci_95_high_pp"])
            n_treatment = int(scenario["n_treatment"])
            n_control = int(scenario["n_control"])
            incremental = int(scenario["incremental_orders"])
            retained = float(scenario["effect_retained_percent"])
            saved = float(scenario["voucher_saved_percent"])
        except (TypeError, ValueError) as exc:
            raise PolicyEvidenceError("Scenario policy có giá trị không phải số") from exc
        if target in seen or target < 1 or target > 100:
            raise PolicyEvidenceError("Target percent bị trùng hoặc ngoài khoảng 1–100")
        seen.add(target)
        if selected != round(population * target / 100):
            raise PolicyEvidenceError("Số khách được chọn không khớp target percent")
        if n_treatment + n_control != selected or min(n_treatment, n_control) <= 0:
            raise PolicyEvidenceError("Cỡ mẫu treatment/control không khớp nhóm policy")
        if not all(math.isfinite(value) for value in (uplift_pp, ci_low, ci_high)):
            raise PolicyEvidenceError("Uplift hoặc khoảng tin cậy không hữu hạn")
        if not ci_low <= uplift_pp <= ci_high:
            raise PolicyEvidenceError("Khoảng tin cậy không chứa uplift point estimate")
        if abs(saved - (100 - target)) > 0.01:
            raise PolicyEvidenceError("Tỷ lệ voucher tiết kiệm không khớp target percent")
        estimated_incremental = selected * uplift_pp / 100
        if abs(estimated_incremental - incremental) > 1.0:
            raise PolicyEvidenceError("Đơn tăng thêm không khớp uplift và số khách")
        if retained < 0 or retained > 100:
            raise PolicyEvidenceError("Hiệu quả giữ được phải nằm trong 0–100%")
        if target < 100:
            profile = scenario.get("feature_profile")
            if not isinstance(profile, list) or not profile:
                raise PolicyEvidenceError("Policy dưới 100% thiếu feature profile")
            for item in profile:
                if not isinstance(item, dict) or not {
                    "feature", "smd", "selected_mean", "remainder_mean"
                } <= item.keys():
                    raise PolicyEvidenceError("Feature profile thiếu trường bắt buộc")
                try:
                    numbers = (
                        float(item["smd"]),
                        float(item["selected_mean"]),
                        float(item["remainder_mean"]),
                    )
                except (TypeError, ValueError) as exc:
                    raise PolicyEvidenceError(
                        "Feature profile có giá trị không phải số"
                    ) from exc
                if not str(item["feature"]) or not all(map(math.isfinite, numbers)):
                    raise PolicyEvidenceError("Feature profile không hợp lệ")

    if default_view not in seen or 100 not in seen:
        raise PolicyEvidenceError("Thiếu scenario mặc định hoặc phát đại trà")
    broadcast = next(row for row in scenarios if int(row["target_percent"]) == 100)
    if int(broadcast["incremental_orders"]) != broadcast_orders:
        raise PolicyEvidenceError("Baseline phát đại trà không khớp scenario 100%")
    return payload
