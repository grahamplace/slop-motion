import json
import shutil

import httpx
import pytest

from scripts import compare_latest_image as experiment
from scripts import compare_models
from scripts.compare_responses import run_chain, sha256
from stop_motion.storage import HarnessError, atomic_write, read_json, write_json

from .conftest import png_bytes
from .test_responses_experiment import client_for, response_body


@pytest.fixture
def plan(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    monkeypatch.setattr(compare_models, "OUTPUT", baseline)
    compare_models.prepare()
    frames = baseline / "sunburst"
    attempts = []
    for index in range(7):
        path = frames / f"{index:02d}.png"
        atomic_write(path, png_bytes("coral"))
        attempts.append(
            {
                "step": index + 1,
                "status": "succeeded",
                "response_id": f"resp_source_{index}",
                "previous_response_id": f"resp_source_{index - 1}" if index else None,
                "output_sha256": sha256(path),
                "image_calls": [{"id": f"ig_source_{index}", "status": "completed"}],
            }
        )
    write_json(frames / "attempts.json", attempts, replace=True)
    monkeypatch.setattr(experiment, "SOURCE", baseline)
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path / "latest-only")
    return experiment.prepare()


def test_same_seed_prompts_instructions_and_settings(plan):
    assert experiment.prepare() == plan
    old = read_json(experiment.SOURCE / "plan.json")["conditions"]["sunburst"]
    assert plan["maximum_response_requests"] == 6
    assert plan["baseline_requests"] == old["baseline_requests"][1:]
    for key in ("driver_model", "instructions", "image_tool"):
        assert plan[key] == old[key]
    assert plan["seed_image_id"] == "ig_source_0"
    assert sha256(experiment.OUTPUT / "00.png") == plan["seed_sha256"]
    assert read_json(experiment.OUTPUT / "attempts.json") == []


def test_changed_plan_is_not_reused(plan):
    changed = {**plan, "seed_image_id": "ig_different"}
    write_json(experiment.OUTPUT / "plan.json", changed, replace=True)
    with pytest.raises(HarnessError, match="Saved plan differs"):
        experiment.prepare()


def test_six_edits_and_idempotent_reports_and_exports(plan):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        assert "previous_response_id" not in payload
        assert len(payload["input"]) == 2
        requests.append(payload)
        body = response_body(len(requests), png_bytes("blue"))
        body["output"][0]["action"] = "edit"
        return httpx.Response(200, json=body)

    with client_for(respond) as client:
        run_chain(client, plan, output=experiment.OUTPUT)
        run_chain(client, plan, output=experiment.OUTPUT)
    assert len(requests) == 6
    assert requests[0]["input"][1]["id"] == "ig_source_0"
    assert requests[-1]["input"][1]["id"] == "ig_5"
    experiment.report(plan)
    assert (experiment.OUTPUT / "comparison.png").exists()
    metrics = read_json(experiment.OUTPUT / "metrics.json")["conditions"]
    assert all(len(frames) == 7 for frames in metrics.values())
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        experiment.render(plan)
        experiment.render(plan)
        for relative, duration, count in (("exports", 0.875, 21), ("slow-review/exports", 7, 168)):
            directory = experiment.OUTPUT / relative
            result = read_json(directory / "export-0001.json")["output"]
            assert result["duration_seconds"] == duration
            assert result["frame_count"] == count
            assert not (directory / "export-0002.mp4").exists()
