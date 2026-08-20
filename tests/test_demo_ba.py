from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from demo_ba.api_client import ApiError, LzdApiClient
from demo_ba.artifact_check import ARTIFACT_FILES, compare_artifacts
from demo_ba.campaign_economics import (
    CampaignAssumptions,
    evaluate_scenarios,
    recommend_scenario,
)
from demo_ba.policy_evidence import (
    DEFAULT_POLICY_PATH,
    PolicyEvidenceError,
    load_policy_evidence,
)
from demo_ba.presentation import (
    ACTION_VIEWS,
    action_display,
    format_decimal_vi,
    format_integer_vi,
    format_uplift_pp,
    realtime_freshness,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class PresentationTests(unittest.TestCase):
    def test_all_six_policy_actions_have_ba_labels(self):
        self.assertEqual(
            set(ACTION_VIEWS),
            {
                "SEND_VOUCHER",
                "WAIT_FOR_INTENT",
                "SUPPRESS_ALREADY_PURCHASED",
                "NOT_SELECTED_BUDGET",
                "SKIP_NON_POSITIVE_UPLIFT",
                "NO_MODEL_SCORE",
            },
        )
        self.assertIn("Gửi", action_display("SEND_VOUCHER"))

    def test_uplift_percentage_point_is_not_multiplied_again(self):
        self.assertEqual(format_uplift_pp(0.558134), "+0,558 điểm %")
        self.assertEqual(format_uplift_pp(-0.127888), "-0,128 điểm %")
        self.assertEqual(format_uplift_pp(None), "Không có điểm")

    def test_ba_numbers_use_vietnamese_separators(self):
        self.assertEqual(format_integer_vi(18167), "18.167")
        self.assertEqual(format_decimal_vi(69.7), "69,7")

    def test_realtime_freshness_uses_last_event_timestamp(self):
        fresh = realtime_freshness(
            {"realtime": {"rt_last_event_ts": 9_500}}, now_epoch=10_000
        )
        stale = realtime_freshness(
            {"realtime": {"rt_last_event_ts": 5_000}}, now_epoch=10_000
        )
        missing = realtime_freshness({"realtime": {}}, now_epoch=10_000)
        future = realtime_freshness(
            {"realtime": {"rt_last_event_ts": 11_000}}, now_epoch=10_000
        )
        self.assertTrue(fresh.is_fresh)
        self.assertEqual(stale.status, "stale")
        self.assertEqual(missing.status, "missing")
        self.assertEqual(future.status, "future")


class ApiClientTests(unittest.TestCase):
    @patch("demo_ba.api_client.urlopen")
    def test_campaign_payload_contains_explicit_cohort_and_budget(self, mocked):
        mocked.return_value = FakeResponse({"results": []})
        client = LzdApiClient("http://localhost:18000")

        client.campaign_decide(["U1", "U2"], budget=1)

        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:18000/campaign/decide")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"user_ids": ["U1", "U2"], "budget": 1, "context": {}},
        )

    @patch("demo_ba.api_client.urlopen")
    def test_422_detail_is_preserved_for_ui(self, mocked):
        mocked.side_effect = HTTPError(
            "http://localhost/campaign/decide",
            422,
            "Unprocessable Entity",
            {},
            io.BytesIO(json.dumps({"detail": "budget quá lớn"}).encode()),
        )
        client = LzdApiClient("http://localhost:18000")

        with self.assertRaises(ApiError) as raised:
            client.campaign_decide(["U1"], budget=2)

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "budget quá lớn")


class ArtifactTests(unittest.TestCase):
    def test_compare_artifacts_detects_equal_and_different_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_dir = root / "ds"
            de_dir = root / "de"
            ds_dir.mkdir()
            de_dir.mkdir()
            for name in ARTIFACT_FILES:
                (ds_dir / name).write_bytes(f"same-{name}".encode())
                (de_dir / name).write_bytes(f"same-{name}".encode())

            rows = compare_artifacts(ds_dir, de_dir)
            self.assertTrue(all(row["byte_identical"] for row in rows))

            (de_dir / ARTIFACT_FILES[0]).write_bytes(b"changed")
            rows = compare_artifacts(ds_dir, de_dir)
            self.assertFalse(rows[0]["byte_identical"])


class PolicyEvidenceTests(unittest.TestCase):
    def test_committed_rct_policy_evidence_is_consistent(self):
        evidence = load_policy_evidence()
        top20 = next(
            row for row in evidence["scenarios"] if row["target_percent"] == 20
        )
        self.assertEqual(evidence["source"]["population"], 90835)
        self.assertEqual(top20["selected_customers"], 18167)
        self.assertEqual(top20["incremental_orders"], 238)
        self.assertEqual(top20["voucher_saved_percent"], 80)
        self.assertLess(
            top20["uplift_ci_95_low_pp"], top20["measured_uplift_pp"]
        )
        self.assertGreater(
            top20["uplift_ci_95_high_pp"], top20["measured_uplift_pp"]
        )
        self.assertEqual(
            top20["n_treatment"] + top20["n_control"],
            top20["selected_customers"],
        )
        self.assertEqual(top20["feature_profile"][0]["feature"], "f28")

    def test_inconsistent_policy_evidence_is_rejected(self):
        payload = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
        payload["scenarios"][0]["selected_customers"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PolicyEvidenceError):
                load_policy_evidence(path)


class CampaignEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_policy_evidence()["scenarios"]

    def test_default_demo_assumptions_recommend_top_20_percent(self):
        assumptions = CampaignAssumptions(
            voucher_unit_cost=20_000,
            redemption_rate=0.10,
            contribution_margin_per_order=200_000,
            maximum_budget=100_000_000,
        )
        evaluations = evaluate_scenarios(self.scenarios, assumptions)
        recommendation = recommend_scenario(evaluations)

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.target_percent, 20)
        self.assertGreater(recommendation.net_profit, 0)
        self.assertLess(recommendation.net_profit_low, 0)
        self.assertGreater(recommendation.net_profit_high, 0)

    def test_no_campaign_wins_when_every_feasible_policy_loses_money(self):
        assumptions = CampaignAssumptions(
            voucher_unit_cost=1_000_000,
            redemption_rate=1.0,
            contribution_margin_per_order=1,
            maximum_budget=1_000_000_000_000,
        )
        evaluations = evaluate_scenarios(self.scenarios, assumptions)

        self.assertIsNone(recommend_scenario(evaluations))

    def test_budget_is_based_on_redemptions_plus_fixed_cost(self):
        assumptions = CampaignAssumptions(
            voucher_unit_cost=20_000,
            redemption_rate=0.10,
            contribution_margin_per_order=200_000,
            maximum_budget=40_000_000,
            fixed_campaign_cost=5_000_000,
        )
        top20 = next(
            row
            for row in evaluate_scenarios(self.scenarios, assumptions)
            if row.target_percent == 20
        )

        self.assertEqual(top20.variable_cost, 36_334_000)
        self.assertEqual(top20.total_cost, 41_334_000)
        self.assertFalse(top20.within_budget)


if __name__ == "__main__":
    unittest.main()
