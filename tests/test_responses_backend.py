"""Default hand-driven CLI/project behavior and conservative provider failures."""

import json

import httpx
import pytest

from stop_motion.images import ImageRequestError, OpenAIResponses
from stop_motion.project import Project
from stop_motion.storage import HarnessError, read_json, write_json

from .conftest import png_bytes
from .test_responses_experiment import client_for, response_body


def test_new_projects_use_responses_and_branch_from_selected_parent(tmp_path, generator):
    project = Project.create(tmp_path / "shot", "A robot waves.")
    assert project.status()["settings"]["provider_options"]["backend"] == "responses"
    project.make_frame("Opening", generator=generator)
    project.make_frame("First edit", base_frame="f0001", generator=generator)
    Project(project.root).make_frame("Next edit", base_frame="f0002", generator=generator)
    project.make_frame("Alternative next edit", base_frame="f0002", generator=generator)
    project.make_frame("Alternative first edit", base_frame="f0001", generator=generator)
    requests = [call[0] for call in generator.calls]
    assert [r["previous_response_id"] for r in requests] == [
        None,
        None,
        "resp_test_1",
        "resp_test_1",
        None,
    ]
    assert [r["action"] for r in requests] == ["generate"] + ["edit"] * 4
    with pytest.raises(HarnessError, match="references are unsupported"):
        project.make_frame(
            "Extra ref", base_frame="f0002", references=["f0001"], generator=generator
        )
    assert len(generator.calls) == 5


def test_legacy_manifest_without_backend_still_uses_images(tmp_path, generator, monkeypatch):
    project = Project.create(tmp_path / "legacy", "Scene", backend="images")
    path = project.root / "project.json"
    state = read_json(path)
    options = state["settings"].pop("provider_options")
    state["settings"].pop("provider")
    state["settings"]["quality"] = options["quality"]
    write_json(path, state, replace=True)
    monkeypatch.setattr("stop_motion.project.OpenAIImages", lambda: generator)
    project.make_frame("Opening")
    project.make_frame("Edit", base_frame="f0001")
    assert "action" not in generator.calls[-1][0]
    assert generator.calls[-1][0]["n"] == 1


@pytest.mark.parametrize(
    "failure",
    ["timeout", 429, 500, "incomplete", "missing", "multiple", "wrong_action", "bad_image"],
)
def test_responses_failures_keep_metadata_without_retries(tmp_path, failure):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        if failure == "timeout":
            raise httpx.ReadTimeout("Timeout", request=request)
        if isinstance(failure, int):
            return httpx.Response(
                failure, json={"error": {"message": "No", "type": "api_error", "code": "no"}}
            )
        body = response_body(1, png_bytes())
        if failure == "incomplete":
            body["status"] = "incomplete"
        elif failure == "missing":
            body["output"] = []
        elif failure == "multiple":
            body["output"] *= 2
        elif failure == "wrong_action":
            body["output"][0]["action"] = "edit"
        else:
            body["output"][0]["result"] = "not base64"
        return httpx.Response(200, json=body, headers={"x-request-id": "req_test"})

    project = Project.create(tmp_path / "shot", "Scene", max_image_requests=1)
    with client_for(respond) as client, pytest.raises(ImageRequestError):
        project.make_frame("Opening", generator=OpenAIResponses(client))
    assert len(calls) == 1
    state = project.status()
    attempt = state["attempts"][0]
    assert attempt["status"] == ("failed" if failure == 429 else "unknown")
    assert state["frames"] == []
    assert state["request_count"] == 1
    if failure not in ("timeout", 429, 500):
        assert attempt["response_id"] == "resp_1"
        assert attempt["provider_request_id"] == "req_test"
        assert "result" not in (root := (project.root / "project.json").read_text())
        assert "base64," not in root
