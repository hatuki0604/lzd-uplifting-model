"""Pure business calculations for the BA campaign policy comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CampaignAssumptions:
    voucher_unit_cost: float
    redemption_rate: float
    contribution_margin_per_order: float
    maximum_budget: float
    fixed_campaign_cost: float = 0.0

    def __post_init__(self) -> None:
        monetary_values = (
            self.voucher_unit_cost,
            self.contribution_margin_per_order,
            self.maximum_budget,
            self.fixed_campaign_cost,
        )
        if any(value < 0 for value in monetary_values):
            raise ValueError("Các giá trị tiền tệ không được âm")
        if not 0 <= self.redemption_rate <= 1:
            raise ValueError("Tỷ lệ sử dụng voucher phải nằm trong 0–1")


@dataclass(frozen=True)
class ScenarioEconomics:
    target_percent: int
    selected_customers: int
    expected_redemptions: float
    variable_cost: float
    total_cost: float
    incremental_orders: float
    incremental_orders_low: float
    incremental_orders_high: float
    incremental_margin: float
    net_profit: float
    net_profit_low: float
    net_profit_high: float
    roi: float | None
    within_budget: bool


def evaluate_scenario(
    scenario: dict[str, Any],
    assumptions: CampaignAssumptions,
) -> ScenarioEconomics:
    selected = int(scenario["selected_customers"])
    expected_redemptions = selected * assumptions.redemption_rate
    variable_cost = expected_redemptions * assumptions.voucher_unit_cost
    total_cost = variable_cost + assumptions.fixed_campaign_cost

    point_orders = selected * float(scenario["measured_uplift_pp"]) / 100
    low_orders = selected * float(scenario["uplift_ci_95_low_pp"]) / 100
    high_orders = selected * float(scenario["uplift_ci_95_high_pp"]) / 100
    incremental_margin = point_orders * assumptions.contribution_margin_per_order
    net_profit = incremental_margin - total_cost
    net_profit_low = (
        low_orders * assumptions.contribution_margin_per_order - total_cost
    )
    net_profit_high = (
        high_orders * assumptions.contribution_margin_per_order - total_cost
    )
    roi = net_profit / total_cost if total_cost > 0 else None

    return ScenarioEconomics(
        target_percent=int(scenario["target_percent"]),
        selected_customers=selected,
        expected_redemptions=expected_redemptions,
        variable_cost=variable_cost,
        total_cost=total_cost,
        incremental_orders=point_orders,
        incremental_orders_low=low_orders,
        incremental_orders_high=high_orders,
        incremental_margin=incremental_margin,
        net_profit=net_profit,
        net_profit_low=net_profit_low,
        net_profit_high=net_profit_high,
        roi=roi,
        within_budget=total_cost <= assumptions.maximum_budget,
    )


def evaluate_scenarios(
    scenarios: Iterable[dict[str, Any]],
    assumptions: CampaignAssumptions,
) -> list[ScenarioEconomics]:
    return [evaluate_scenario(scenario, assumptions) for scenario in scenarios]


def recommend_scenario(
    evaluations: Iterable[ScenarioEconomics],
) -> ScenarioEconomics | None:
    """Return max expected profit within budget, or None for no campaign.

    A campaign with zero or negative expected profit is deliberately not picked:
    the comparison includes an implicit "do nothing" policy with profit zero.
    """
    feasible = [row for row in evaluations if row.within_budget]
    if not feasible:
        return None
    best = max(feasible, key=lambda row: (row.net_profit, -row.total_cost))
    return best if best.net_profit > 0 else None
