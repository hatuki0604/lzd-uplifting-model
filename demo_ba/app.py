"""Streamlit dashboard that presents uplift decisions in BA language."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

# Streamlit executes the target as a script and may put only `demo_ba/` on
# sys.path. Add the repository root so package imports work regardless of the
# current shell directory or Streamlit's runner implementation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_ba.api_client import ApiError, LzdApiClient
from demo_ba.artifact_check import compare_artifacts
from demo_ba.campaign_economics import (
    CampaignAssumptions,
    ScenarioEconomics,
    evaluate_scenarios,
    recommend_scenario,
)
from demo_ba.policy_evidence import PolicyEvidenceError, load_policy_evidence
from demo_ba.presentation import (
    action_display,
    action_view,
    format_decimal_vi,
    format_age,
    format_integer_vi,
    format_uplift_pp,
    realtime_freshness,
)


COHORT_PATH = Path(__file__).with_name("cohort.json")
REFRESH_COMMAND = (
    'cd "/Users/hatrungkien/MacDrive/GreenSM/nhung-lala/LZD"\n'
    "make track-b-online-demo"
)


def load_default_cohort() -> dict[str, Any]:
    return json.loads(COHORT_PATH.read_text(encoding="utf-8"))


def stop_with_api_error(error: ApiError) -> None:
    st.error(f"Không thể tải dữ liệu demo: {error}")
    if error.detail:
        st.code(json.dumps(error.detail, ensure_ascii=False, indent=2), language="json")
    st.info("Hãy kiểm tra hoặc chạy lại stack demo DE:")
    st.code(REFRESH_COMMAND, language="bash")
    st.stop()


def render_html_table(
    headers: list[str],
    rows: list[list[str]],
    row_colors: list[str] | None = None,
) -> None:
    """Render a small table without pandas/pyarrow.

    PyArrow 23 can segfault on the second Streamlit rerun in this Python 3.10
    environment. A native HTML table is sufficient for the bounded BA demo and
    keeps widget interactions from terminating the whole server.
    """
    colors = row_colors or ["#ffffff"] * len(rows)
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row, color in zip(rows, colors):
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        body.append(f'<tr style="background:{escape(color)}">{cells}</tr>')
    st.markdown(
        '<div class="demo-table-wrap"><table class="demo-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


def format_vnd(value: float) -> str:
    return f"{format_integer_vi(round(value))} ₫"


def format_million_vnd(value: float) -> str:
    return f"{format_decimal_vi(value / 1_000_000)} triệu ₫"


def policy_label(target_percent: int) -> str:
    return f"Top {target_percent}%" if target_percent < 100 else "Phát đại trà"


def format_profit_interval(row: ScenarioEconomics) -> str:
    return (
        f"{format_million_vnd(row.net_profit)} "
        f"[{format_million_vnd(row.net_profit_low)}; "
        f"{format_million_vnd(row.net_profit_high)}]"
    )


def render_offline_policy(evidence: dict[str, Any]) -> None:
    source = evidence["source"]
    scenarios = evidence["scenarios"]

    st.subheader("1. Hiệu quả campaign theo nhóm — bằng chứng RCT offline")
    st.markdown(
        f"Trên **{format_integer_vi(source['population'])} khách hàng của RCT holdout**, "
        "mô hình chấm điểm từng người rồi chọn nhóm có uplift dự đoán cao nhất. "
        "Treatment/control ngẫu nhiên được dùng để đo uplift thực tế của từng nhóm Top%. "
        "Các dòng dưới đây là những kịch bản để so sánh, chưa phải policy được chọn sẵn."
    )

    policy_rows: list[list[str]] = []
    for row in scenarios:
        target = int(row["target_percent"])
        policy_rows.append([
            policy_label(target),
            format_integer_vi(row["selected_customers"]),
            format_uplift_pp(row["measured_uplift_pp"]),
            (
                f"[{format_uplift_pp(row['uplift_ci_95_low_pp'])}; "
                f"{format_uplift_pp(row['uplift_ci_95_high_pp'])}]"
            ),
            f"{format_integer_vi(row['n_treatment'])} / {format_integer_vi(row['n_control'])}",
            f"≈ {format_integer_vi(row['incremental_orders'])}",
            f"{format_decimal_vi(row['effect_retained_percent'])}%",
            f"{format_decimal_vi(row['voucher_saved_percent'], 0)}%",
        ])
    render_html_table(
        [
            "Chính sách",
            "Số khách",
            "Uplift thực đo",
            "KTC 95% của uplift",
            "Treatment / Control",
            "Đơn tăng thêm",
            "Hiệu quả giữ được",
            "Voucher tiết kiệm",
        ],
        policy_rows,
    )
    st.caption(
        "KTC 95% biểu diễn độ bất định của uplift đo trong từng nhóm; khoảng chứa 0 "
        "nghĩa là riêng nhóm đó chưa đủ bằng chứng uplift khác 0 ở mức ý nghĩa 5%."
    )

    st.markdown("#### 1.1. Tính hiệu quả kinh doanh")
    st.info(
        "Các số bên dưới là giả định demo. Hãy thay bằng chi phí voucher, tỷ lệ sử "
        "dụng và biên lợi nhuận thật của doanh nghiệp."
    )
    input_cols = st.columns(3)
    voucher_unit_cost = input_cols[0].number_input(
        "Chi phí mỗi voucher được dùng (₫)",
        min_value=0,
        value=20_000,
        step=5_000,
        key="voucher_unit_cost",
    )
    redemption_percent = input_cols[1].number_input(
        "Tỷ lệ khách sử dụng voucher (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="redemption_percent",
    )
    margin_per_order = input_cols[2].number_input(
        "Biên lợi nhuận mỗi đơn tăng thêm (₫)",
        min_value=0,
        value=200_000,
        step=10_000,
        key="margin_per_order",
    )
    budget_cols = st.columns(2)
    maximum_budget = budget_cols[0].number_input(
        "Ngân sách campaign tối đa (₫)",
        min_value=0,
        value=100_000_000,
        step=10_000_000,
        key="maximum_budget",
    )
    fixed_cost = budget_cols[1].number_input(
        "Chi phí campaign cố định (₫)",
        min_value=0,
        value=0,
        step=5_000_000,
        key="fixed_campaign_cost",
    )

    assumptions = CampaignAssumptions(
        voucher_unit_cost=float(voucher_unit_cost),
        redemption_rate=float(redemption_percent) / 100,
        contribution_margin_per_order=float(margin_per_order),
        maximum_budget=float(maximum_budget),
        fixed_campaign_cost=float(fixed_cost),
    )
    evaluations = evaluate_scenarios(scenarios, assumptions)
    recommendation = recommend_scenario(evaluations)

    if recommendation is None:
        st.warning(
            "Đề xuất theo các giả định hiện tại: **không chạy campaign**. Không có "
            "kịch bản nào vừa nằm trong ngân sách vừa có lợi nhuận kỳ vọng dương."
        )
        profile_default_target = int(evidence["default_view_target_percent"])
    else:
        st.success(
            f"Đề xuất theo các giả định hiện tại: **{policy_label(recommendation.target_percent)}** "
            "vì có lợi nhuận kỳ vọng cao nhất trong các kịch bản không vượt ngân sách."
        )
        result_cols = st.columns(5)
        result_cols[0].metric(
            "Nhóm mục tiêu",
            f"{format_integer_vi(recommendation.selected_customers)} khách",
        )
        result_cols[1].metric("Chi phí", format_million_vnd(recommendation.total_cost))
        result_cols[2].metric(
            "Đơn tăng thêm",
            f"≈ {format_integer_vi(round(recommendation.incremental_orders))}",
        )
        result_cols[3].metric(
            "Lãi ròng kỳ vọng", format_million_vnd(recommendation.net_profit)
        )
        result_cols[4].metric(
            "ROI kỳ vọng",
            "Không xác định"
            if recommendation.roi is None
            else f"{format_decimal_vi(recommendation.roi * 100)}%",
        )
        if recommendation.net_profit_low <= 0 <= recommendation.net_profit_high:
            st.warning(
                "Khoảng lợi nhuận suy ra từ KTC uplift vẫn chứa 0: point estimate có "
                "lãi, nhưng dữ liệu hiện tại chưa loại trừ khả năng campaign bị lỗ."
            )
        profile_default_target = recommendation.target_percent

    economics_rows: list[list[str]] = []
    economics_colors: list[str] = []
    for row in evaluations:
        economics_rows.append([
            policy_label(row.target_percent),
            format_vnd(row.total_cost),
            f"≈ {format_integer_vi(round(row.incremental_orders))}",
            format_million_vnd(row.incremental_margin),
            format_profit_interval(row),
            "—" if row.roi is None else f"{format_decimal_vi(row.roi * 100)}%",
            "Đủ" if row.within_budget else "Vượt",
        ])
        is_recommended = (
            recommendation is not None
            and row.target_percent == recommendation.target_percent
        )
        economics_colors.append(
            "#dcfce7"
            if is_recommended
            else ("#fee2e2" if not row.within_budget else "#ffffff")
        )
    render_html_table(
        [
            "Chính sách",
            "Tổng chi phí",
            "Đơn tăng thêm",
            "Biên lợi nhuận tăng thêm",
            "Lãi ròng kỳ vọng [KTC]",
            "ROI",
            "Ngân sách",
        ],
        economics_rows,
        economics_colors,
    )
    st.caption(
        "Tổng chi phí = số khách × tỷ lệ dùng voucher × chi phí/voucher + chi phí cố định. "
        "Lãi ròng = đơn tăng thêm × biên lợi nhuận/đơn − tổng chi phí. "
        "Kịch bản được đề xuất tối đa hóa lãi ròng kỳ vọng trong ngân sách và luôn được "
        "so với phương án không chạy campaign (lãi = 0)."
    )

    st.markdown("#### 1.2. Nhóm khách mục tiêu khác phần còn lại ở đâu?")
    profiled_scenarios = [
        row for row in scenarios if isinstance(row.get("feature_profile"), list)
    ]
    profile_targets = [int(row["target_percent"]) for row in profiled_scenarios]
    if profile_default_target not in profile_targets:
        profile_default_target = int(evidence["default_view_target_percent"])
    profile_target = st.selectbox(
        "Chọn nhóm Top% để xem hồ sơ",
        options=profile_targets,
        index=profile_targets.index(profile_default_target),
        format_func=lambda value: f"Top {value}%",
        key="profile_target_percent",
    )
    profile_scenario = next(
        row for row in profiled_scenarios if int(row["target_percent"]) == profile_target
    )
    st.markdown(
        f"Nhóm **Top {profile_target}%** gồm "
        f"**{format_integer_vi(profile_scenario['selected_customers'])} khách** có uplift "
        "dự đoán cao nhất. Dưới đây là 10 feature có chênh lệch chuẩn hóa lớn nhất "
        "so với phần khách còn lại."
    )
    profile_rows: list[list[str]] = []
    for item in profile_scenario["feature_profile"]:
        smd = float(item["smd"])
        profile_rows.append([
            str(item["feature"]),
            format_decimal_vi(item["selected_mean"], 3),
            format_decimal_vi(item["remainder_mean"], 3),
            f"{smd:+.3f}".replace(".", ","),
            "Cao hơn" if smd > 0 else "Thấp hơn",
        ])
    render_html_table(
        ["Feature", "TB nhóm mục tiêu", "TB phần còn lại", "SMD", "Khác biệt"],
        profile_rows,
    )
    st.caption(
        "Tên f* là feature đã ẩn danh trong dữ liệu nguồn. SMD dương nghĩa là giá trị "
        "trung bình của nhóm mục tiêu cao hơn; SMD âm nghĩa là thấp hơn. Đây là mô tả "
        "tương quan của nhóm model chọn, không phải feature importance và không chứng "
        "minh feature đó gây ra uplift."
    )
    st.caption(
        "Nguồn được tái tạo từ model_booster.txt, feature_contract.json và "
        "dataset/rct_holdout.parquet theo notebook Bước 37/40. " + str(evidence["caveat"])
    )


def render_uplift_chart(rows: list[dict[str, Any]]) -> None:
    """Render a compact horizontal uplift chart without Arrow conversion."""
    numeric = [abs(float(row["uplift"])) for row in rows if row.get("uplift") is not None]
    scale = max(numeric, default=1.0) or 1.0
    bars = []
    for row in rows:
        raw = row.get("uplift")
        value = float(raw) if raw is not None else 0.0
        width = max(2.0, abs(value) / scale * 100.0)
        color = "#2563eb" if value >= 0 else "#6b7280"
        bars.append(
            '<div class="uplift-row">'
            f'<span class="uplift-user">{escape(str(row["user_id"]))}</span>'
            '<div class="uplift-track">'
            f'<div class="uplift-bar" style="width:{width:.2f}%;background:{color}"></div>'
            "</div>"
            f'<span class="uplift-value">{escape(format_uplift_pp(raw))}</span>'
            "</div>"
        )
    st.markdown(f'<div class="uplift-chart">{"".join(bars)}</div>', unsafe_allow_html=True)


def render_artifact_evidence(api_model_version: str) -> None:
    rows = compare_artifacts()
    matched = sum(bool(row["byte_identical"]) for row in rows)
    if matched == len(rows):
        st.success(f"Handoff repo DS → DE: {matched}/{len(rows)} artifact trùng byte.")
    else:
        st.warning(
            f"Chỉ đối chiếu được {matched}/{len(rows)} artifact. "
            "Kiểm tra lại đường dẫn repo DE hoặc artifact mới."
        )
    evidence: list[list[str]] = []
    for row in rows:
        ds_hashes = row.get("ds_hashes", {})
        evidence.append([
            str(row["file"]),
            "Có" if row["byte_identical"] else "Không",
            str(ds_hashes.get("sha256", "Không tìm thấy")),
        ])
    render_html_table(["File", "Trùng byte", "SHA-256 DS"], evidence)
    st.caption(
        f"Runtime API tự báo model version: {api_model_version}. Đối chiếu trên "
        "chứng minh handoff giữa hai repo; nó không phải attestation của Docker image."
    )


st.set_page_config(
    page_title="Voucher Decision Demo",
    page_icon="🎟️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.45rem;}
      .demo-subtitle {color: #4b5563; margin-top: -0.5rem;}
      .demo-table-wrap {overflow-x: auto; margin: .5rem 0 1rem;}
      .demo-table {border-collapse: collapse; width: 100%; font-size: .92rem;}
      .demo-table th, .demo-table td {
        border-bottom: 1px solid #d1d5db; padding: .55rem .65rem;
        text-align: left; vertical-align: top;
      }
      .demo-table th {background: #f8fafc; font-weight: 650; white-space: nowrap;}
      .uplift-chart {display: grid; gap: .48rem; margin: .75rem 0 1rem;}
      .uplift-row {
        display: grid; grid-template-columns: 7rem minmax(12rem, 1fr) 8rem;
        gap: .65rem; align-items: center;
      }
      .uplift-user {font-family: monospace; font-weight: 600;}
      .uplift-track {height: 1.05rem; background: #e5e7eb; border-radius: .28rem; overflow: hidden;}
      .uplift-bar {height: 100%; border-radius: .28rem;}
      .uplift-value {text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎟️ Voucher Decision Demo")
st.markdown(
    '<p class="demo-subtitle">Bằng chứng RCT cho thấy hiệu quả ở cấp nhóm; '
    "API live minh họa cách hệ thống xếp hạng và ra quyết định cho từng khách.</p>",
    unsafe_allow_html=True,
)

try:
    offline_policy = load_policy_evidence()
except PolicyEvidenceError as exc:
    st.warning(f"Không hiển thị được bằng chứng policy offline: {exc}")
else:
    render_offline_policy(offline_policy)

st.divider()
st.subheader("2. API live — cách hệ thống chọn và xử lý từng khách")
st.caption(
    "Cohort 10 khách dưới đây là dữ liệu synthetic để demo vận hành API; "
    "không dùng nó để đo uplift thực tế ở cấp nhóm."
)

default_cohort = load_default_cohort()
user_ids = list(default_cohort["user_ids"])
configured_budget = int(default_cohort["default_budget"])

with st.sidebar:
    st.header("Thiết lập demo")
    api_url = st.text_input(
        "FastAPI URL",
        value=os.environ.get("LZD_API_URL", "http://localhost:18000"),
    )
    st.caption("Cohort demo cố định: U0000000–U0000009 (10 khách hàng).")

    default_budget = min(configured_budget, len(user_ids))
    budget = st.slider(
        "Số khách hàng được xét ưu tiên (Top-K)",
        min_value=1,
        max_value=len(user_ids),
        value=default_budget,
        key="campaign_budget",
        help="Đây là số slot được xét, không phải số voucher chắc chắn được gửi.",
    )
    if st.button("🔄 Làm mới dữ liệu", width="stretch"):
        st.rerun()

try:
    client = LzdApiClient(api_url)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    with st.spinner("Đang đọc model, feature store và campaign..."):
        ready = client.ready()
        store_info = client.store_info()
        campaign = client.campaign_decide(user_ids, budget=budget)
except ApiError as exc:
    stop_with_api_error(exc)

results = campaign.get("results")
if not isinstance(results, list) or not results:
    st.error("API không trả về danh sách kết quả campaign hợp lệ.")
    st.json(campaign)
    st.stop()

status_cols = st.columns(4)
status_cols[0].metric("Hệ thống", "Sẵn sàng" if ready.get("status") == "ready" else "Chưa sẵn sàng")
status_cols[1].metric("Model", str(campaign.get("model_version", "Không rõ")))
status_cols[2].metric("Feature version", str(campaign.get("feature_version", "Không rõ")))
status_cols[3].metric("Khách có batch feature", int(store_info.get("batch_keys") or 0))

cache_misses = [str(row.get("user_id")) for row in results if not row.get("cache_hit")]
if cache_misses:
    st.warning(
        "Các user sau không có batch feature thật và có thể đang được điền default: "
        + ", ".join(cache_misses)
        + ". Không nên dùng các dòng này trong phần demo BA."
    )

now_epoch = time.time()
freshness_by_user = {
    str(row.get("user_id")): realtime_freshness(row, now_epoch=now_epoch)
    for row in results
}
fresh_count = sum(item.is_fresh for item in freshness_by_user.values())
realtime_key_count = int(store_info.get("realtime_keys") or 0)
if fresh_count == len(results):
    st.success(f"Tín hiệu gần nhất còn mới cho {fresh_count}/{len(results)} khách hàng.")
elif fresh_count == 0:
    st.error(
        "Tín hiệu realtime đã hết hạn hoặc chưa được tạo. Xếp hạng uplift vẫn có thể "
        "xem, nhưng không dùng các hành động gửi/chờ/chặn để thuyết trình."
    )
    st.code(REFRESH_COMMAND, language="bash")
else:
    st.warning(
        f"Chỉ {fresh_count}/{len(results)} khách hàng có tín hiệu realtime còn mới "
        f"(Redis hiện có {realtime_key_count} realtime key)."
    )

counts = Counter(str(row.get("campaign_action")) for row in results)
summary_cols = st.columns(5)
summary_cols[0].metric("Top-K slots", budget)
summary_cols[1].metric("🟢 Gửi ngay", counts.get("SEND_VOUCHER", 0))
summary_cols[2].metric("🟡 Chờ", counts.get("WAIT_FOR_INTENT", 0))
summary_cols[3].metric("🔴 Đã mua", counts.get("SUPPRESS_ALREADY_PURCHASED", 0))
summary_cols[4].metric("⚫ Uplift không dương", counts.get("SKIP_NON_POSITIVE_UPLIFT", 0))

sent = counts.get("SEND_VOUCHER", 0)
if sent < budget:
    st.info(
        f"Top-K có {budget} slot nhưng hiện chỉ gửi ngay {sent} voucher. "
        "Slot bị WAIT hoặc SUPPRESS không được tự động chuyển xuống hạng thấp hơn "
        "(policy không backfill)."
    )

st.subheader("Xếp hạng mức tăng chuyển đổi dự kiến")
chart_rows = [
    {
        "user_id": str(row.get("user_id")),
        "uplift": row.get("uplift_percentage_point"),
    }
    for row in results
    if row.get("uplift_percentage_point") is not None
]
render_uplift_chart(chart_rows)
st.caption(
    "Ví dụ +0,558 điểm % nghĩa là model ước lượng voucher làm xác suất chuyển đổi "
    "tăng khoảng 0,558 điểm phần trăm cho khách hàng đó."
)

st.subheader("Khuyến nghị campaign")
table_rows: list[list[str]] = []
row_backgrounds: list[str] = []
for row in results:
    user_id = str(row.get("user_id"))
    action = str(row.get("campaign_action"))
    freshness = freshness_by_user[user_id]
    table_rows.append([
        str(int(row.get("rank") or 0)),
        user_id,
        format_uplift_pp(row.get("uplift_percentage_point")),
        action_display(action),
        str(row.get("reason") or ""),
        format_age(freshness.age_seconds),
    ])
    row_backgrounds.append(action_view(action).background)

render_html_table(
    ["Hạng", "Khách hàng", "Uplift", "Khuyến nghị", "Lý do", "Tín hiệu"],
    table_rows,
    row_backgrounds,
)

st.subheader("Giải thích một khách hàng")
selected_user = st.selectbox(
    "Chọn khách hàng",
    options=[str(row.get("user_id")) for row in results],
    format_func=lambda uid: next(
        f"#{row.get('rank')} — {uid} — {format_uplift_pp(row.get('uplift_percentage_point'))}"
        for row in results
        if str(row.get("user_id")) == uid
    ),
)
selected = next(row for row in results if str(row.get("user_id")) == selected_user)
selected_action = str(selected.get("campaign_action"))
selected_freshness = freshness_by_user[selected_user]
detail_cols = st.columns(4)
detail_cols[0].metric("Xếp hạng", f"#{selected.get('rank')}/{len(results)}")
detail_cols[1].metric("Uplift dự kiến", format_uplift_pp(selected.get("uplift_percentage_point")))
detail_cols[2].metric("Khuyến nghị", action_view(selected_action).short_label)
detail_cols[3].metric("Tín hiệu", format_age(selected_freshness.age_seconds))
st.markdown(f"**Lý do nghiệp vụ:** {selected.get('reason', 'Không có lý do')}")

realtime = selected.get("realtime") if isinstance(selected.get("realtime"), dict) else {}
signal_cols = st.columns(3)
signal_cols[0].metric("Xem sản phẩm (1h)", float(realtime.get("rt_page_view_1h") or 0))
signal_cols[1].metric("Thêm vào giỏ (1h)", float(realtime.get("rt_add_to_cart_1h") or 0))
signal_cols[2].metric("Đơn hàng (1h)", float(realtime.get("rt_order_1h") or 0))

api_disclaimer = campaign.get("disclaimer")
if api_disclaimer:
    st.warning(f"Thông báo trực tiếp từ API: {api_disclaimer}")

with st.expander("Chi tiết kỹ thuật và bằng chứng artifact"):
    st.json({
        "ready": ready,
        "store_info": store_info,
        "model_contract": campaign.get("model_contract"),
        "policy": campaign.get("policy"),
    })
    render_artifact_evidence(str(campaign.get("model_version", "Không rõ")))
    if st.button("Gọi /decide?debug=true cho khách hàng đang chọn"):
        try:
            st.json(client.decide_debug(selected_user))
        except ApiError as exc:
            st.error(str(exc))
