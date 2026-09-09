import json
import shutil

import httpx
import pytest

from scripts import compare_models as comparison
from scripts.compare_responses import run_chain
from stop_motion.storage import HarnessError, read_json

from .conftest import png_bytes
from .test_responses_experiment import client_for, response_body


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "OUTPUT", tmp_path)
    return comparison.prepare()


def test_prepare_is_repeatable_and_freezes_identical_prompts(plan):
    assert comparison.prepare() == plan
    newer, older = plan["conditions"].values()
    assert newer["baseline_requests"] == older["baseline_requests"]
    assert newer["driver_model"] == older["driver_model"] == "gpt-5.5"
    assert newer["image_tool"]["model"] == "gpt-image-2.5-sunburst"
    assert older["image_tool"]["model"] == "gpt-image-2"
    assert sum(arm["maximum_response_requests"] for arm in (newer, older)) == 14


def test_changed_plan_is_rejected(plan, monkeypatch):
    monkeypatch.setattr(comparison, "MODELS", {"sunburst": "different-model"})
    with pytest.raises(HarnessError, match="Saved plan differs"):
        comparison.prepare()


def test_two_independent_seven_call_chains_and_video_outputs(plan):
    requests = []
    png = png_bytes("coral")

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        body = response_body(len(requests), png)
        body["output"][0]["model"] = payload["tools"][0]["model"]
        body["output"][0]["action"] = payload["tools"][0]["action"]
        return httpx.Response(200, json=body)

    with client_for(respond) as client:
        for _ in range(2):  # Second invocation must be read-only.
            for name, arm in plan["conditions"].items():
                run_chain(client, arm, output=comparison.OUTPUT / name)
    assert len(requests) == 14
    for offset, (name, arm) in zip((0, 7), plan["conditions"].items(), strict=True):
        for index, request in enumerate(requests[offset : offset + 7]):
            assert request["input"] == arm["baseline_requests"][index]["prompt"]
            assert request["tools"][0]["model"] == arm["image_tool"]["model"]
            assert request["tools"][0]["quality"] == "medium"
            assert request["max_tool_calls"] == 1
            if index == 0:
                assert request["tools"][0]["action"] == "generate"
                assert "previous_response_id" not in request
            else:
                assert request["tools"][0]["action"] == "edit"
                assert request["previous_response_id"] == f"resp_{offset + index}"
        assert len(list((comparison.OUTPUT / name).glob("[0-9][0-9].png"))) == 7
    comparison.report(plan)
    metrics = read_json(comparison.OUTPUT / "metrics.json")
    assert metrics["conditions"]["sunburst"][0]["mean_absolute_rgb_difference"]["background"] == 0
    assert (comparison.OUTPUT / "comparison.png").exists()
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        comparison.render_review(plan)
        for name in comparison.MODELS:
            video = read_json(comparison.OUTPUT / name / "exports/export-0001.json")["output"]
            assert video["duration_seconds"] == 0.875
            assert video["frame_count"] == 21
        review = read_json(comparison.OUTPUT / "slow-review/exports/export-0001.json")["output"]
        assert review["duration_seconds"] == 7
        assert review["frame_count"] == 168
        comparison.render_review(plan)
        assert not (comparison.OUTPUT / "slow-review/exports/export-0002.mp4").exists()


def test_invalid_budget_stops_before_any_api_call(plan):
    arm = {**plan["conditions"]["sunburst"], "maximum_response_requests": 8}
    with pytest.raises(HarnessError, match="exactly match"):
        run_chain(None, arm, output=comparison.OUTPUT / "sunburst")
