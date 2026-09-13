"""Gemini HTTP contract and CLI compilation with real PNGs and video tools."""

import base64
import copy
import hashlib
import io
import json
import shutil
import subprocess
import sys

import httpx
import pytest
from PIL import Image

from stop_motion.cli import main
from stop_motion.images import ImageRequestError, prepare_request
from stop_motion.project import Project
from stop_motion.providers.gemini import DEFAULT_MODEL, EDIT_INSTRUCTIONS, GeminiImages
from stop_motion.scene import compile_scene, load_scene
from stop_motion.settings import resolve_settings
from stop_motion.storage import HarnessError, read_json, write_json

from .conftest import png_bytes


def image_body(number=1, *, mime="image/png", png=None):
    data = png or png_bytes("blue", step=number)
    if mime == "image/jpeg":
        with Image.open(io.BytesIO(data)) as image:
            output = io.BytesIO()
            image.save(output, format="JPEG")
            data = output.getvalue()
    return {
        "id": f"interaction_{number}",
        "status": "completed",
        "model": DEFAULT_MODEL,
        "steps": [
            {
                "type": "thought",
                "summary": [
                    {"type": "image", "mime_type": "image/png", "data": "ignore-this-thought-image"}
                ],
            },
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "The image follows."},
                    {"type": "image", "mime_type": mime, "data": base64.b64encode(data).decode()},
                ],
            },
        ],
        "usage": {"total_tokens": 123, "total_input_tokens": 23, "total_output_tokens": 100},
    }


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "brief": "A clay robot waves.",
            "settings": {"provider": "gemini", "max_image_requests": 7},
            "opening": {"prompt": "A clay robot", "hold": 3},
            "edits": [
                {"prompt": f"Raise the waving arm to pose {i}.", "hold": 3} for i in range(6)
            ],
        },
    )
    return path


@pytest.fixture
def media():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")


def generator(handler):
    return GeminiImages(api_key="offline-gemini-key", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
def test_http_auth_final_image_extraction_and_png_normalization(mime):
    calls = []
    body = image_body(mime=mime)

    def respond(request):
        calls.append(request)
        assert request.url == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == "offline-gemini-key"
        return httpx.Response(200, json=body, headers={"x-goog-request-id": "request-1"})

    result = generator(respond).generate(
        prepare_request(resolve_settings({"provider": "gemini"}), "Robot")
    )
    assert len(calls) == 1
    assert result.request_id == "request-1"
    assert result.continuation == {"interaction_id": "interaction_1"}
    assert result.usage == body["usage"]
    assert result.metadata["output_image_count"] == 1
    source = base64.b64decode(body["steps"][1]["content"][1]["data"])
    assert result.metadata["source_sha256"] == hashlib.sha256(source).hexdigest()
    with Image.open(io.BytesIO(result.png)) as output, Image.open(io.BytesIO(source)) as original:
        assert output.format == "PNG"
        assert output.size == original.size
        assert output.convert("RGB").tobytes() == original.convert("RGB").tobytes()


def test_staged_compile_reopens_chain_and_reuses_video(scene, tmp_path, media):
    root = tmp_path / "build"
    requests = []
    images = []

    def respond(request):
        requests.append(json.loads(request.content))
        png = png_bytes("blue", step=len(requests))
        images.append(png)
        return httpx.Response(200, json=image_body(len(requests), png=png))

    first = compile_scene(scene, root, through=3, generator=generator(respond))
    original = {p.name: p.read_bytes() for p in (root / "frames").glob("*.png")}
    assert first["requests_made"] == 3
    final = compile_scene(scene, root, generator=generator(respond))
    assert final["requests_made"] == 4
    assert final["video"]["frame_count"] == 21
    assert len(requests) == 7
    for index, request in enumerate(requests):
        assert request["model"] == DEFAULT_MODEL
        assert request["store"] is True
        assert request["stream"] is False
        assert request["background"] is False
        assert request["response_format"] == {
            "type": "image",
            "mime_type": "image/jpeg",
            "delivery": "inline",
            "aspect_ratio": "1:1",
            "image_size": "1K",
        }
        if index:
            assert request["system_instruction"] == EDIT_INSTRUCTIONS
        if index < 2:
            assert "previous_interaction_id" not in request
        else:
            assert request["previous_interaction_id"] == f"interaction_{index}"
            assert isinstance(request["input"], str)
    assert base64.b64decode(requests[1]["input"][0]["data"]) == images[0]
    assert all(
        (root / "frames" / name).read_bytes() == content for name, content in original.items()
    )
    again = compile_scene(scene, root)
    assert again["requests_made"] == 0 and again["reused_video"]
    assert again["video"] == final["video"]
    # An actual separate process can validate and reuse a completed build with no keys.
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "stop_motion",
            "compile",
            "--scene",
            str(scene),
            "--project",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(run.stdout)["requests_made"] == 0
    manifest = (root / "project.json").read_text()
    assert "offline-gemini-key" not in manifest and "ignore-this-thought-image" not in manifest
    assert base64.b64encode(images[0]).decode() not in manifest


def test_imported_opening_retime_append_and_manual_branch(scene, tmp_path, media):
    raw = read_json(scene)
    seed = tmp_path / "seed.png"
    seed.write_bytes(png_bytes("pink"))
    raw["opening"] = {"image": "seed.png"}
    raw["edits"] = raw["edits"][:1]
    write_json(scene, raw, replace=True)
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=image_body(len(calls)))

    root = tmp_path / "build"
    assert compile_scene(scene, root, generator=generator(respond))["requests_made"] == 1
    assert (root / "frames/f0001.png").read_bytes() == seed.read_bytes()
    assert "previous_interaction_id" not in calls[0]
    assert base64.b64decode(calls[0]["input"][0]["data"]) == seed.read_bytes()
    raw["settings"]["fps"] = 12
    write_json(scene, raw, replace=True)
    assert compile_scene(scene, root)["requests_made"] == 0
    raw["edits"].append({"prompt": "Wave again"})
    write_json(scene, raw, replace=True)
    assert compile_scene(scene, root, generator=generator(respond))["requests_made"] == 1
    assert calls[-1]["previous_interaction_id"] == "interaction_1"
    project = Project.create(tmp_path / "manual", "Robot", provider="gemini")
    project.make_frame("Opening", generator=generator(respond))
    project.make_frame("First edit", base_frame="f0001", generator=generator(respond))
    project.make_frame("Rejected edit", base_frame="f0002", generator=generator(respond))
    project.make_frame("Chosen edit", base_frame="f0002", generator=generator(respond))
    assert (
        calls[-2]["previous_interaction_id"]
        == calls[-1]["previous_interaction_id"]
        == "interaction_4"
    )
    project.make_frame("Fresh first edit", base_frame="f0001", generator=generator(respond))
    assert "previous_interaction_id" not in calls[-1]
    before = project.status()["request_count"]
    with pytest.raises(HarnessError, match="references are unsupported"):
        project.make_frame("Extra reference", base_frame="f0002", references=["f0001"])
    assert project.status()["request_count"] == before


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        400,
        401,
        403,
        404,
        408,
        429,
        500,
        302,
        "not-json",
        "empty",
        "incomplete",
        "missing-id",
        "missing-image",
        "multiple-images",
        "bad-data",
        "mime-mismatch",
        "wrong-model",
        "wrong-parent",
        "malformed-steps",
        "wrong-size",
        "transparent",
    ],
)
def test_failures_stop_without_retry_or_replay(scene, tmp_path, media, failure):
    calls = []
    body = image_body()
    image = body["steps"][1]["content"][1]
    if failure == "empty":
        body = []
    elif failure == "incomplete":
        body["status"] = "incomplete"
    elif failure == "missing-id":
        body.pop("id")
    elif failure == "missing-image":
        body["steps"] = []
    elif failure == "multiple-images":
        body["steps"][1]["content"].append(copy.deepcopy(image))
    elif failure == "bad-data":
        image["data"] = "bad-base64"
    elif failure == "mime-mismatch":
        image["mime_type"] = "image/jpeg"
    elif failure == "wrong-model":
        body["model"] = "another-model"
    elif failure == "wrong-parent":
        body["previous_interaction_id"] = "unrelated"
    elif failure == "malformed-steps":
        body["steps"] = [None]
    elif failure == "wrong-size":
        image["data"] = base64.b64encode(png_bytes(size=(800, 1024))).decode()
    elif failure == "transparent":
        output = io.BytesIO()
        Image.new("RGBA", (1024, 1024)).save(output, format="PNG")
        image["data"] = base64.b64encode(output.getvalue()).decode()

    def respond(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("Timeout with sensitive request details", request=request)
        if isinstance(failure, int):
            return httpx.Response(
                failure,
                json={"error": {"message": "sensitive"}},
                headers={"location": "https://example.com"},
            )
        if failure == "not-json":
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json=body)

    root = tmp_path / "build"
    for dry_run in (False, False, True):
        with pytest.raises(HarnessError):
            compile_scene(scene, root, generator=generator(respond), dry_run=dry_run)
    assert len(calls) == 1
    state = Project(root).status()
    assert state["request_count"] == 1
    assert "sensitive" not in (root / "project.json").read_text()
    if failure == "wrong-size":
        assert (root / "frames/f0001.png").exists()
    else:
        assert state["frames"] == []


@pytest.mark.parametrize(
    "settings",
    [
        {"model": "gpt-image-2.5-sunburst"},
        {"quality": "medium"},
        {"driver_model": "gpt-5.5"},
        {"backend": "responses"},
        {"provider_options": {"quality": "high"}},
        {"size": "1536x1024"},
        {"size": "512x512"},
    ],
)
def test_invalid_settings_fail_without_spending(scene, tmp_path, settings):
    raw = read_json(scene)
    raw["settings"].update(settings)
    write_json(scene, raw, replace=True)
    with pytest.raises(HarnessError):
        compile_scene(scene, tmp_path / "build")
    assert not (tmp_path / "build").exists()


def test_cli_defaults_size_mapping_missing_key_and_cap(scene, tmp_path, capsys):
    brief = tmp_path / "brief.txt"
    brief.write_text("Robot")
    root = tmp_path / "manual"
    assert (
        main(
            [
                "init",
                str(root),
                "--brief-file",
                str(brief),
                "--provider",
                "gemini",
                "--size",
                "2752x1536",
                "--max-image-requests",
                "1",
            ]
        )
        == 0
    )
    settings = json.loads(capsys.readouterr().out)["settings"]
    assert settings["model"] == DEFAULT_MODEL and settings["provider_options"] == {}
    project = Project(root)
    with pytest.raises(HarnessError, match="GEMINI_API_KEY"):
        project.make_frame("Robot")
    assert project.status()["request_count"] == 0
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(429)

    with pytest.raises(ImageRequestError):
        project.make_frame("Robot", generator=generator(respond))
    assert calls[0]["response_format"]["aspect_ratio"] == "16:9"
    assert calls[0]["response_format"]["image_size"] == "2K"
    with pytest.raises(HarnessError, match="budget exhausted"):
        project.make_frame("Robot", generator=generator(respond))
    assert len(calls) == 1
    assert load_scene(scene)["settings"]["provider"] == "gemini"


def test_cli_resumes_partial_build_in_fresh_process(scene, tmp_path, media):
    root = tmp_path / "build"
    counter = iter(range(1, 3))
    compile_scene(
        scene,
        root,
        through=2,
        generator=generator(lambda request: httpx.Response(200, json=image_body(next(counter)))),
    )
    script = """
import itertools, json, sys
import httpx
from stop_motion.cli import main
from stop_motion.providers import gemini
from tests.test_gemini import image_body
counter = itertools.count(3)
def respond(request):
    number = next(counter)
    payload = json.loads(request.content)
    assert payload['previous_interaction_id'] == f'interaction_{number - 1}'
    return httpx.Response(200, json=image_body(number))
gemini.create_generator = lambda settings: gemini.GeminiImages(
    api_key='offline-key', transport=httpx.MockTransport(respond))
raise SystemExit(main(sys.argv[1:]))
"""
    run = subprocess.run(
        [sys.executable, "-c", script, "compile", "--scene", str(scene), "--project", str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(run.stdout)
    assert result["requests_made"] == 5 and result["request_count"] == 7
    assert Project(root).status()["attempts"][-1]["continuation"] == {
        "interaction_id": "interaction_7"
    }


@pytest.mark.parametrize("mutation", ["missing-state", "parent", "provider", "model", "workflow"])
def test_inconsistent_saved_state_stops_before_more_requests(scene, tmp_path, media, mutation):
    root = tmp_path / "build"
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=image_body(len(calls)))

    compile_scene(scene, root, through=3, generator=generator(respond))
    path = root / "project.json"
    data = read_json(path)
    attempt = data["attempts"][-1]
    if mutation == "missing-state":
        attempt["continuation"] = None
    elif mutation == "parent":
        attempt["request"]["continuation"] = {"interaction_id": "wrong-parent"}
    elif mutation == "provider":
        attempt["provider"] = "openai"
    elif mutation == "model":
        attempt["request"]["settings"]["model"] = "wrong-model"
    else:
        attempt["request"]["workflow"] = "another-workflow"
    write_json(path, data, replace=True)
    before = path.read_bytes()
    with pytest.raises(HarnessError):
        compile_scene(scene, root, generator=generator(respond))
    assert len(calls) == 3 and path.read_bytes() == before
