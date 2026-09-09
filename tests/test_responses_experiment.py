"""Offline SDK checks for the paid, bounded Responses experiment."""

import base64
import json

import httpx
import pytest
from openai import OpenAI

from scripts import compare_responses as experiment
from stop_motion.storage import HarnessError, read_json, write_json

from .conftest import png_bytes


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    (tmp_path / "00.png").write_bytes(png_bytes())
    write_json(tmp_path / "attempts.json", [])
    return {
        "driver_model": "gpt-5.5",
        "instructions": experiment.INSTRUCTIONS,
        "baseline_requests": [{"prompt": f"Exact prompt {index}\n"} for index in range(6)],
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


def client_for(handler):
    return OpenAI(
        api_key="offline-test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def response_body(step, png):
    return {
        "id": f"resp_{step}",
        "model": "gpt-5.5",
        "status": "completed",
        "usage": None,
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "id": f"ig_{step}",
                "type": "image_generation_call",
                "status": "completed",
                "result": base64.b64encode(png).decode(),
                "revised_prompt": f"Rewritten {step}",
                "model": "gpt-image-2.5-sunburst",
                "quality": "medium",
                "size": "1024x1024",
                "output_format": "png",
                "background": "opaque",
            }
        ],
    }


def test_sdk_chain_uses_previous_response_id_and_stops_at_six(plan):
    requests = []
    png = png_bytes("blue")

    def respond(request):
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_body(len(requests), png),
            headers={"x-request-id": f"req_{len(requests)}"},
        )

    with client_for(respond) as client:
        experiment.run_chain(client, plan)
        experiment.run_chain(client, plan)  # Completed runs never spend again.
    assert len(requests) == 6
    content = requests[0]["input"][0]["content"]
    assert base64.b64decode(content[0]["image_url"].split(",", 1)[1]) == png_bytes()
    assert content[1]["text"] == plan["baseline_requests"][0]["prompt"]
    assert "previous_response_id" not in requests[0]
    for index, request in enumerate(requests):
        assert request["max_tool_calls"] == 1
        assert request["tools"] == [plan["image_tool"]]
        assert request["tool_choice"] == {"type": "image_generation"}
        assert request["store"] is True
        assert request["parallel_tool_calls"] is False
        if index:
            assert request["previous_response_id"] == f"resp_{index}"
            assert request["input"] == plan["baseline_requests"][index]["prompt"]
    attempts = read_json(experiment.OUTPUT / "attempts.json")
    assert all(attempt["status"] == "succeeded" for attempt in attempts)
    assert attempts[-1]["image_calls"][0]["revised_prompt"] == "Rewritten 6"
    assert "result" not in attempts[-1]["image_calls"][0]
    assert attempts[-1]["provider_request_id"] == "req_6"
    assert (experiment.OUTPUT / "06.png").read_bytes() == png


def test_latest_image_only_has_one_image_reference_and_no_response_history(plan):
    plan.update(
        context_mode="latest_image_id",
        seed_image_id="ig_seed",
        seed_sha256=experiment.sha256(experiment.OUTPUT / "00.png"),
    )
    requests = []

    def respond(request):
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        index = len(requests)
        assert "previous_response_id" not in payload
        assert "conversation" not in payload
        assert payload["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": plan["baseline_requests"][index]["prompt"]}
                ],
            },
            {"type": "image_generation_call", "id": f"ig_{index}" if index else "ig_seed"},
        ]
        assert payload["tools"] == [plan["image_tool"]]
        requests.append(payload)
        return httpx.Response(200, json=response_body(index + 1, png_bytes("blue")))

    with client_for(respond) as client:
        experiment.run_chain(client, plan)
        experiment.run_chain(client, plan)
    assert len(requests) == 6
    attempts = read_json(experiment.OUTPUT / "attempts.json")
    assert all(attempt["previous_response_id"] is None for attempt in attempts)
    assert attempts[0]["input_image_ids"] == ["ig_seed"]
    assert attempts[-1]["input_image_ids"] == ["ig_5"]
    attempts[-1]["input_image_ids"] = ["ig_seed"]
    write_json(experiment.OUTPUT / "attempts.json", attempts, replace=True)
    with pytest.raises(HarnessError, match="image reference"):
        experiment.run_chain(None, plan)


def test_latest_image_mode_requires_a_valid_seed_before_spending(plan):
    plan.update(context_mode="latest_image_id", seed_image_id="ig_seed", seed_sha256="changed")
    with pytest.raises(HarnessError, match="seed"):
        experiment.run_chain(None, plan)


@pytest.mark.parametrize("status", [400, 500])
def test_api_failure_stops_without_retry_or_replay(plan, status):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "Rejected",
                    "type": "api_error",
                    "code": "test_error",
                }
            },
        )

    with client_for(respond) as client:
        with pytest.raises(HarnessError, match="Rejected"):
            experiment.run_chain(client, plan)
        with pytest.raises(HarnessError, match="refusing to replay"):
            experiment.run_chain(client, plan)
    assert len(calls) == 1
    attempt = read_json(experiment.OUTPUT / "attempts.json")[0]
    assert attempt["status"] == ("unknown" if status >= 500 else "failed")


def test_connection_failure_is_unknown_and_never_replayed(plan):
    calls = []

    def respond(request):
        calls.append(request)
        raise httpx.ReadTimeout("Offline timeout", request=request)

    with client_for(respond) as client:
        with pytest.raises(HarnessError, match="remote outcome unknown"):
            experiment.run_chain(client, plan)
        with pytest.raises(HarnessError, match="refusing to replay"):
            experiment.run_chain(client, plan)
    assert len(calls) == 1


def test_invalid_output_preserves_response_id_and_stops(plan):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=response_body(1, b"not a PNG"))

    with client_for(respond) as client:
        with pytest.raises(HarnessError):
            experiment.run_chain(client, plan)
        with pytest.raises(HarnessError, match="refusing to replay"):
            experiment.run_chain(client, plan)
    assert len(calls) == 1
    attempt = read_json(experiment.OUTPUT / "attempts.json")[0]
    assert attempt["response_id"] == "resp_1"
    assert attempt["status"] == "unknown"
    assert not (experiment.OUTPUT / "01.png").exists()
