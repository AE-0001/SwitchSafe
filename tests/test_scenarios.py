from switchsafe.llm_provider import evaluate_planner
from switchsafe.scenarios import build_scenarios


def test_scenario_matrix_has_120_labelled_cases():
    scenarios = build_scenarios()
    assert len(scenarios) == 120
    assert len({scenario["id"] for scenario in scenarios}) == 120
    assert all("expected_status" in scenario for scenario in scenarios)


class FakePlanner:
    model = "fake"

    def plan(self, query: str) -> dict:
        return {"steps": ["retrieve_guidance", "check_device_status"],
                "latency_ms": 1.0, "prompt_tokens": 10, "completion_tokens": 3}


def test_llm_planner_evaluation_is_reproducible_with_fake_provider():
    result = evaluate_planner(FakePlanner(), build_scenarios()[:1])
    assert result["schema_valid_rate"] == 1
    assert result["mean_required_step_recall"] == 1
    assert result["prompt_tokens"] == 10
