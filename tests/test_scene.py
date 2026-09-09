"""Offline end-to-end compile tests, including real SDK serialization and ffmpeg."""

import base64
import copy
import json
import shutil
from pathlib import Path

import httpx
import pytest

from scripts import compare_responses as experiment
from stop_motion.cli import main
from stop_motion.images import EDIT_INSTRUCTIONS, ImageRequestError, OpenAIResponses
from stop_motion.project import Project
from stop_motion.scene import compile_scene, load_scene
from stop_motion.storage import HarnessError, project_lock, read_json, write_json

from .conftest import FakeImages, png_bytes
from .test_responses_experiment import client_for, response_body


@pytest.fixture
def scene(tmp_path):
    directory = tmp_path / "scene files"
    directory.mkdir()
    (directory / "opening.txt").write_text("Generate one clay robot.\n")
    path = directory / "scene.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "brief": "A robot waves.",
            "settings": {"max_image_requests": 7},
            "opening": {"prompt_file": "opening.txt", "hold": 3},
            "edits": [{"prompt": f"Edit: wave pose {i}.\n", "hold": 3} for i in range(6)],
        },
    )
    return path


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def media():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe needed for integration tests")


def test_compile_matches_tested_uploaded_opening_requests_and_resumes(scene, tmp_path, media):
    root = tmp_path / "build"
    requests = []
    images = []

    def respond(request):
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        requests.append(payload)
        png = png_bytes("blue", step=len(requests))
        images.append(png)
        body = response_body(len(requests), png)
        body["output"][0]["action"] = payload["tools"][0]["action"]
        return httpx.Response(200, json=body, headers={"x-request-id": f"req_{len(requests)}"})

    with client_for(respond) as client:
        first = compile_scene(scene, root, through=3, generator=OpenAIResponses(client))
    assert first["request_count"] == 3
    assert first["video"]["frame_count"] == 9
    original = {p.name: p.read_bytes() for p in (root / "frames").glob("*.png")}
    # A different SDK instance and no Python conversation state resumes from disk.
    with client_for(respond) as client:
        final = compile_scene(scene, root, generator=OpenAIResponses(client))
    assert final["requests_made"] == 4
    assert final["video"]["frame_count"] == 21
    assert final["video"]["duration_seconds"] == 0.875
    assert all((root / "frames" / name).read_bytes() == png for name, png in original.items())
    assert requests[0]["tools"][0]["action"] == "generate"
    assert "previous_response_id" not in requests[0]
    assert "previous_response_id" not in requests[1]  # NEVER link opening response.
    content = requests[1]["input"][0]["content"]
    assert base64.b64decode(content[0]["image_url"].split(",", 1)[1]) == images[0]
    assert content[0]["detail"] == "high"

    # Exact edit request parity with the successful experiment, including instructions.
    (tmp_path / "00.png").write_bytes(images[0])
    prompts = load_scene(scene)["steps"][1:]
    baseline = {
        "driver_model": "gpt-5.5",
        "instructions": experiment.INSTRUCTIONS,
        "baseline_requests": prompts,
        "image_tool": requests[1]["tools"][0],
    }
    parents = [{"response_id": f"resp_{i}"} for i in range(2, 8)]
    for index, request in enumerate(requests[1:]):
        assert request == experiment.build_request(baseline, index, parents, output=tmp_path)
    assert EDIT_INSTRUCTIONS == experiment.INSTRUCTIONS
    state = Project(root).status()
    assert state["attempts"][2]["response_id"] == "resp_3"
    assert state["attempts"][2]["provider_request_id"] == "req_3"
    assert state["attempts"][2]["image_calls"][0]["revised_prompt"] == "Rewritten 3"
    assert "base64" not in (root / "project.json").read_text()
    assert "result" not in state["attempts"][2]["image_calls"][0]
    again = compile_scene(scene, root)  # No key, requests, or duplicate export.
    assert again["requests_made"] == 0
    assert again["reused_video"] is True
    assert again["video"] == final["video"]
    assert len(requests) == 7


def test_import_opening_retime_and_append(scene, tmp_path, generator, media):
    raw = read_json(scene)
    seed = scene.parent / "seed.png"
    seed.write_bytes(png_bytes("pink"))
    raw["opening"] = {"image": "seed.png", "hold": 3}
    raw["edits"] = raw["edits"][:1]
    write_json(scene, raw, replace=True)
    root = tmp_path / "build"
    first = compile_scene(scene, root, generator=generator)
    assert first["requests_made"] == 1
    assert (root / "frames/f0001.png").read_bytes() == seed.read_bytes()
    assert generator.calls[0][0]["action"] == "edit"
    assert generator.calls[0][0]["previous_response_id"] is None
    assert generator.calls[0][1] == [root / "frames/f0001.png"]
    raw["opening"]["hold"] = 6
    raw["settings"]["fps"] = 12
    write_json(scene, raw, replace=True)
    second = compile_scene(scene, root)
    assert second["requests_made"] == 0
    assert second["video"]["duration_seconds"] == 0.75
    assert second["video"]["path"] != first["video"]["path"]
    assert Path(first["video"]["path"]).exists()
    raw["edits"].append({"prompt": "Edit: lower the waving arm."})
    write_json(scene, raw, replace=True)
    last = compile_scene(scene, root, generator=generator)
    assert last["requests_made"] == 1
    assert generator.calls[-1][0]["previous_response_id"] == "resp_test_0"


def test_unrendered_prompts_can_change_but_completed_prompts_cannot(
    scene, tmp_path, generator, media
):
    root = tmp_path / "build"
    compile_scene(scene, root, through=1, generator=generator)
    raw = read_json(scene)
    raw["edits"][0]["prompt"] = "A smaller movement."
    write_json(scene, raw, replace=True)
    assert compile_scene(scene, root, dry_run=True)["remaining_requests"] == 6
    compile_scene(scene, root, through=2, generator=generator)
    raw["edits"][0]["prompt"] = "A different completed movement."
    write_json(scene, raw, replace=True)
    before = (root / "project.json").read_bytes()
    with pytest.raises(HarnessError, match="Completed pose 2 changed"):
        compile_scene(scene, root, generator=generator)
    assert (root / "project.json").read_bytes() == before
    assert len(generator.calls) == 2


@pytest.mark.parametrize("unknown", [False, True])
def test_failed_compile_never_replays(scene, tmp_path, unknown, media):
    calls = []

    class Failing:
        def generate(self, request, inputs):
            calls.append(request)
            raise ImageRequestError("Test failure", unknown=unknown)

    root = tmp_path / "build"
    with pytest.raises(ImageRequestError):
        compile_scene(scene, root, generator=Failing())
    for dry_run in (True, False):
        with pytest.raises(HarnessError, match="refusing to replay"):
            compile_scene(scene, root, generator=Failing(), dry_run=dry_run)
    assert len(calls) == 1
    assert Project(root).status()["request_count"] == 1


def test_interrupted_compile_and_orphan_are_not_replayed(
    scene, tmp_path, generator, monkeypatch, media
):
    root = tmp_path / "build"
    save = Project._save

    def interrupted(self, data):
        if data["frames"]:
            raise KeyboardInterrupt
        save(self, data)

    with monkeypatch.context() as patch:
        patch.setattr(Project, "_save", interrupted)
        with pytest.raises(KeyboardInterrupt):
            compile_scene(scene, root, generator=generator)
    png = (root / "frames/f0001.png").read_bytes()
    assert Project(root).status()["attempts"][0]["status"] == "unknown"
    with pytest.raises(HarnessError, match="refusing to replay"):
        compile_scene(scene, root, generator=generator)
    assert len(generator.calls) == 1
    assert (root / "frames/f0001.png").read_bytes() == png


def test_changed_png_settings_and_chain_are_rejected(scene, tmp_path, generator, media):
    root = tmp_path / "build"
    compile_scene(scene, root, through=3, generator=generator)
    path = root / "project.json"
    initial = read_json(path)
    for mutation in ("chain", "model", "response", "steps", "attempts"):
        state = copy.deepcopy(initial)
        if mutation == "chain":
            state["attempts"][-1]["request"]["previous_response_id"] = "resp_wrong"
        elif mutation == "model":
            state["attempts"][-1]["request"]["model"] = "gpt-image-2.5-flare"
        elif mutation == "response":
            state["attempts"][-1]["response_id"] = None
        elif mutation == "steps":
            state["compile"]["steps"] = []
        else:
            state["frames"].pop()
        write_json(path, state, replace=True)
        with pytest.raises(HarnessError):
            compile_scene(scene, root, generator=generator)
    write_json(path, initial, replace=True)
    raw = read_json(scene)
    raw["settings"]["quality"] = "high"
    write_json(scene, raw, replace=True)
    with pytest.raises(HarnessError, match="Generation setting"):
        compile_scene(scene, root, generator=generator)
    raw["settings"]["quality"] = "medium"
    write_json(scene, raw, replace=True)
    (root / "frames/f0001.png").write_bytes(png_bytes("yellow"))
    with pytest.raises(HarnessError, match="Saved frame"):
        compile_scene(scene, root, generator=generator)
    assert len(generator.calls) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"schema_version": True},
        {"edits": "wrong"},
        {"settings": {"max_image_requests": 6}},
        {"settings": {"backend": "images"}},
        {"settings": {"model": []}},
        {"opening": {"prompt": "a", "prompt_file": "b"}},
        {"opening": {"image": "x", "prompt": "a"}},
        {"edits": [{"prompt": "x", "hold": True}]},
        {"brief": ""},
        {"edits": [{"prompt_file": "missing.txt"}]},
    ],
)
def test_invalid_scene_fails_before_writes_or_requests(scene, tmp_path, generator, change):
    raw = read_json(scene)
    raw.update(change)
    write_json(scene, raw, replace=True)
    root = tmp_path / "build"
    with pytest.raises((HarnessError, OSError)):
        compile_scene(scene, root, generator=generator)
    assert not root.exists()
    assert generator.calls == []


def test_cli_dry_run_and_progress_contract(scene, tmp_path, generator, monkeypatch, capsys, media):
    root = tmp_path / "build"
    argv = ["compile", "--scene", str(scene), "--project", str(root)]
    assert main([*argv, "--dry-run"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["remaining_requests"] == 7
    assert not root.exists()
    monkeypatch.setattr("stop_motion.project.OpenAIResponses", lambda: generator)
    assert main([*argv, "--through", "2"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["requests_made"] == 2
    events = [json.loads(line) for line in output.err.splitlines()]
    assert [e["event"] for e in events] == ["pose_started", "pose_saved"] * 2
    original = (root / "project.json").read_bytes()
    assert main([*argv, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["remaining_requests"] == 5
    assert (root / "project.json").read_bytes() == original
    with project_lock(root), pytest.raises(HarnessError, match="busy"):
        compile_scene(scene, root, generator=generator)
    with pytest.raises(HarnessError, match="compiled project"):
        Project(root).make_frame("Untracked manual edit", generator=generator)


def test_missing_ffmpeg_and_wrong_output_size_stop_before_more_calls(
    scene, tmp_path, monkeypatch, media
):
    root = tmp_path / "build"
    generator = FakeImages(size=(800, 1024))
    with monkeypatch.context() as patch:
        patch.setattr("stop_motion.scene.shutil.which", lambda _: None)
        with pytest.raises(HarnessError, match="ffmpeg"):
            compile_scene(scene, root, generator=generator)
    assert generator.calls == []
    assert not root.exists()
    with pytest.raises(HarnessError, match="dimensions"):
        compile_scene(scene, root, generator=generator)
    assert len(generator.calls) == 1
    assert (root / "frames/f0001.png").exists()
    with pytest.raises(HarnessError, match="dimensions"):
        compile_scene(scene, root, generator=generator)
    assert len(generator.calls) == 1
