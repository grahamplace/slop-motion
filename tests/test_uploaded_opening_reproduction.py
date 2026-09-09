import json
import shutil

import httpx
import pytest

from scripts import reproduce_uploaded_opening as experiment
from scripts.compare_responses import INSTRUCTIONS, build_request, run_chain, sha256
from stop_motion.storage import HarnessError, atomic_write, read_json, write_json

from .conftest import png_bytes
from .test_responses_experiment import client_for, response_body


@pytest.fixture
def plans(tmp_path, monkeypatch):
    butterfly = tmp_path / "original-butterfly"
    robot = tmp_path / "original-robot"
    butterfly.mkdir()
    (robot / "sunburst").mkdir(parents=True)
    monkeypatch.setattr(experiment, "BUTTERFLY", butterfly)
    monkeypatch.setattr(experiment, "ROBOT", robot)
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path / "reproduction")
    settings = {
        "driver_model": "gpt-5.5",
        "instructions": INSTRUCTIONS,
        "image_tool": {
            "type": "image_generation",
            "action": "edit",
            "model": "gpt-image-2.5-sunburst",
            "quality": "medium",
            "size": "1024x1024",
            "background": "opaque",
            "output_format": "png",
        },
    }
    for root, offset in ((butterfly, 1), (robot / "sunburst", 0)):
        attempts = []
        for i in range(7):
            path = root / f"{i:02d}.png"
            atomic_write(path, png_bytes("coral" if offset == 0 else "green"))
            if i >= offset:
                attempts.append(
                    {
                        "step": i + 1 - offset,
                        "status": "succeeded",
                        "response_id": f"resp_old_{i}",
                        "previous_response_id": f"resp_old_{i - 1}" if i > offset else None,
                        "output_sha256": sha256(path),
                    }
                )
        write_json(root / "attempts.json", attempts)
    write_json(
        butterfly / "plan.json",
        {
            **settings,
            "baseline_sha256": [sha256(butterfly / f"{i:02d}.png") for i in range(7)],
            "baseline_requests": [{"prompt": f"Butterfly edit {i}\n"} for i in range(6)],
        },
    )
    write_json(
        robot / "plan.json",
        {
            "conditions": {
                "sunburst": {
                    **settings,
                    "instructions": "Generate the opening, then edit.",
                    "generates_opening": True,
                    "baseline_requests": [
                        {"prompt": "Generate robot"},
                        *[{"prompt": f"Robot edit {i}\n"} for i in range(6)],
                    ],
                }
            }
        },
    )
    return {name: experiment.prepare(name) for name in ("butterfly", "robot")}


def test_exact_butterfly_requests_and_robot_bootstrap(plans):
    original = read_json(experiment.BUTTERFLY / "plan.json")
    dummy_attempts = [{"response_id": f"resp_new_{i}"} for i in range(6)]
    for i in range(6):
        assert build_request(
            plans["butterfly"], i, dummy_attempts, output=experiment.OUTPUT / "butterfly"
        ) == build_request(original, i, dummy_attempts, output=experiment.BUTTERFLY)
    robot = read_json(experiment.ROBOT / "plan.json")["conditions"]["sunburst"]
    assert plans["robot"]["baseline_requests"] == robot["baseline_requests"][1:]
    assert plans["robot"]["instructions"] == original["instructions"]
    assert plans["robot"]["image_tool"] == robot["image_tool"]
    for subject, plan in plans.items():
        assert experiment.prepare(subject) == plan
        assert plan["maximum_response_requests"] == 6
        request = build_request(plan, 0, [], output=experiment.OUTPUT / subject)
        assert "previous_response_id" not in request
        assert request["input"][0]["content"][0]["type"] == "input_image"
        assert request["tools"][0]["action"] == "edit"


def test_frozen_seed_and_plan(plans):
    seed = experiment.OUTPUT / "butterfly/00.png"
    atomic_write(seed, png_bytes("red"), replace=True)
    with pytest.raises(HarnessError, match="seed changed"):
        experiment.prepare("butterfly")


def test_control_review_gates_robot_and_two_chains_resume_safely(plans):
    with pytest.raises(HarnessError, match="visually review"):
        experiment.require_control_pass()
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        index = len(requests) % 6
        if index == 0:
            assert "previous_response_id" not in payload
            assert payload["input"][0]["content"][0]["type"] == "input_image"
        else:
            assert payload["previous_response_id"] == f"resp_{len(requests)}"
            assert isinstance(payload["input"], str)
        assert payload["tools"][0]["action"] == "edit"
        requests.append(payload)
        body = response_body(len(requests), png_bytes("blue"))
        body["output"][0]["action"] = "edit"
        return httpx.Response(200, json=body)

    with client_for(respond) as client:
        control = experiment.OUTPUT / "butterfly"
        run_chain(client, plans["butterfly"], output=control)
        with pytest.raises(HarnessError):
            experiment.require_control_pass()
        review = {
            "verdict": "fail",
            "frame_sha256": [sha256(control / f"{i:02d}.png") for i in range(7)],
        }
        write_json(control / "review.json", review)
        with pytest.raises(HarnessError, match="passing visual review"):
            experiment.require_control_pass()
        review["verdict"] = "pass"
        write_json(control / "review.json", review, replace=True)
        experiment.require_control_pass()
        run_chain(client, plans["robot"], output=experiment.OUTPUT / "robot")
        for subject, plan in plans.items():
            run_chain(client, plan, output=experiment.OUTPUT / subject)
    assert len(requests) == 12
    for subject, plan in plans.items():
        has_ffmpeg = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        experiment.report(plan, render=has_ffmpeg)
        experiment.report(plan, render=has_ffmpeg)
        directory = experiment.OUTPUT / subject
        assert (directory / "comparison.png").exists()
        if has_ffmpeg:
            assert read_json(directory / "exports/export-0001.json")["output"]["frame_count"] == 21
            assert (
                read_json(directory / "slow-review/exports/export-0001.json")["output"][
                    "duration_seconds"
                ]
                == 7
            )
            assert not (directory / "exports/export-0002.mp4").exists()
    review["frame_sha256"][0] = "different"
    write_json(control / "review.json", review, replace=True)
    with pytest.raises(HarnessError, match="passing visual review"):
        experiment.require_control_pass()
