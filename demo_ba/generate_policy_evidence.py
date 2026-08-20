"""Regenerate BA policy evidence from the committed model and RCT holdout.

This script intentionally mirrors the ranking, tie-breaking, uplift and SMD
formulas in ``notebooks/04_benchmark_evaluation.ipynb``.  The generated JSON is
small enough for Streamlit to load without importing pandas or LightGBM at
runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("offline_policy.json")
TARGET_FRACTIONS = (0.1, 0.2, 0.3, 0.5, 1.0)
Z_95 = 1.959963984540054


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_in_proportions(
    outcomes: np.ndarray,
    treatment: np.ndarray,
) -> dict[str, float | int]:
    treated = outcomes[treatment == 1]
    control = outcomes[treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Mỗi nhóm policy phải có cả treatment và control")

    p_treated = float(treated.mean())
    p_control = float(control.mean())
    standard_error = float(
        np.sqrt(
            p_treated * (1 - p_treated) / len(treated)
            + p_control * (1 - p_control) / len(control)
        )
    )
    uplift = p_treated - p_control
    return {
        "uplift": uplift,
        "standard_error": standard_error,
        "ci_low": uplift - Z_95 * standard_error,
        "ci_high": uplift + Z_95 * standard_error,
        "n_treatment": int(len(treated)),
        "n_control": int(len(control)),
    }


def feature_profile(
    frame: pd.DataFrame,
    selected: np.ndarray,
    feature_names: list[str],
    *,
    limit: int = 10,
) -> list[dict[str, float | str]]:
    selected_frame = frame.iloc[selected][feature_names]
    remainder_mask = np.ones(len(frame), dtype=bool)
    remainder_mask[selected] = False
    remainder_frame = frame.loc[remainder_mask, feature_names]

    selected_mean = selected_frame.mean()
    remainder_mean = remainder_frame.mean()
    pooled_sd = np.sqrt((selected_frame.var() + remainder_frame.var()) / 2)
    smd = ((selected_mean - remainder_mean) / pooled_sd.replace(0, np.nan)).dropna()
    ordered = smd.reindex(smd.abs().sort_values(ascending=False).index).head(limit)

    return [
        {
            "feature": str(name),
            "smd": round(float(value), 6),
            "selected_mean": round(float(selected_mean[name]), 6),
            "remainder_mean": round(float(remainder_mean[name]), 6),
        }
        for name, value in ordered.items()
    ]


def generate_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    dataset_path = project_root / "dataset" / "rct_holdout.parquet"
    model_path = project_root / "artifacts" / "model_booster.txt"
    contract_path = project_root / "artifacts" / "feature_contract.json"
    metadata_path = project_root / "artifacts" / "metadata.json"

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = [
        item["ten"]
        for item in sorted(
            contract["dac_trung"],
            key=lambda item: int(item["thu_tu_dua_vao_mo_hinh"]),
        )
    ]
    seed = int(metadata["seed"])

    frame = pd.read_parquet(dataset_path)
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"RCT holdout thiếu feature: {missing}")

    booster = lgb.Booster(model_file=str(model_path))
    if booster.num_feature() != len(feature_names):
        raise ValueError(
            f"Model nhận {booster.num_feature()} cột nhưng contract có {len(feature_names)}"
        )
    scores = np.asarray(booster.predict(frame[feature_names]), dtype=float)
    outcomes = frame["label"].to_numpy(dtype=float)
    treatment = frame["is_treat"].to_numpy(dtype=int)
    population = len(frame)

    rng = np.random.RandomState(seed)
    order = np.lexsort((rng.rand(population), -scores))

    full_measurement = difference_in_proportions(outcomes, treatment)
    broadcast_incremental = float(full_measurement["uplift"]) * population
    control_cvr = float(outcomes[treatment == 0].mean())

    scenarios: list[dict[str, Any]] = []
    for fraction in TARGET_FRACTIONS:
        selected_count = max(1, int(round(fraction * population)))
        selected = order[:selected_count]
        measurement = difference_in_proportions(
            outcomes[selected], treatment[selected]
        )
        uplift = float(measurement["uplift"])
        incremental_orders = uplift * selected_count
        target_percent = int(round(fraction * 100))

        scenario: dict[str, Any] = {
            "target_percent": target_percent,
            "selected_customers": selected_count,
            "measured_uplift_pp": round(uplift * 100, 6),
            "uplift_ci_95_low_pp": round(float(measurement["ci_low"]) * 100, 6),
            "uplift_ci_95_high_pp": round(float(measurement["ci_high"]) * 100, 6),
            "uplift_standard_error_pp": round(
                float(measurement["standard_error"]) * 100, 6
            ),
            "n_treatment": int(measurement["n_treatment"]),
            "n_control": int(measurement["n_control"]),
            "incremental_orders": int(round(incremental_orders)),
            "effect_retained_percent": round(
                incremental_orders / broadcast_incremental * 100, 6
            ),
            "voucher_saved_percent": int(round((1 - fraction) * 100)),
            "campaign_cvr_percent": round(
                (control_cvr + incremental_orders / population) * 100, 6
            ),
        }
        if target_percent < 100:
            scenario["feature_profile"] = feature_profile(
                frame, selected, feature_names
            )
        scenarios.append(scenario)

    return {
        "schema_version": "offline_policy_evidence_v2",
        "source": {
            "dataset": "rct_holdout",
            "population": population,
            "randomized_treatment": True,
            "notebook": "notebooks/04_benchmark_evaluation.ipynb",
            "notebook_section": "Bước 37 và Bước 40",
            "model": str(metadata["ten_mo_hinh"]),
            "seed": seed,
            "feature_count": len(feature_names),
            "artifact_sha256": {
                "rct_holdout.parquet": sha256(dataset_path),
                "model_booster.txt": sha256(model_path),
                "feature_contract.json": sha256(contract_path),
            },
        },
        "baseline": {
            "control_cvr_percent": round(control_cvr * 100, 6),
            "broadcast_incremental_orders": int(round(broadcast_incremental)),
            "broadcast_uplift_ci_95_low_pp": round(
                float(full_measurement["ci_low"]) * 100, 6
            ),
            "broadcast_uplift_ci_95_high_pp": round(
                float(full_measurement["ci_high"]) * 100, 6
            ),
        },
        "default_view_target_percent": 20,
        "confidence_interval": {
            "level_percent": 95,
            "method": "Wald normal interval for difference of two proportions",
            "interpretation": (
                "Nếu lặp lại cách lấy mẫu RCT nhiều lần, khoảng 95% khoảng tính theo "
                "cách này sẽ chứa uplift thật của nhóm. Khoảng chứa 0 nghĩa là chưa đủ "
                "bằng chứng rằng uplift của riêng nhóm đó khác 0 ở mức 5%."
            ),
        },
        "scenarios": scenarios,
        "caveat": (
            "Đây là mô phỏng policy và uplift thực đo trên RCT holdout, không phải "
            "kết quả realtime của cohort 10 khách. Qini holdout có KTC 95% chứa 0 "
            "nên không được diễn giải là mô hình đã chứng minh xếp hạng tốt ở mức "
            "ý nghĩa 95%. Feature profile là tương quan mô tả, không phải nguyên nhân."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = generate_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Đã ghi {args.output} với {len(payload['scenarios'])} policy scenario")


if __name__ == "__main__":
    main()
