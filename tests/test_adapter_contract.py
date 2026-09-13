"""Shared compiler behavior must not require a vendor conversation identifier."""

import json
import shutil
import subprocess
import sys
from types import ModuleType

import pytest

from stop_motion import providers
from stop_motion.images import GeneratedImage, ImageRequestError
from stop_motion.project import Project
from stop_motion.scene import compile_scene
from stop_motion.storage import HarnessError, read_json, write_json

from .conftest import png_bytes


@pytest.fixture
def stateless(monkeypatch):
    module = ModuleType("test_stateless_provider")
    module.DEFAULT_MODEL = "test-image"
    module.DEFAULT_OPTIONS = {}
    module.validate = lambda model, dimensions, options: None
    module.workflow = lambda settings: "test.stateless.v1"
    module.describe_workflow = lambda settings: "previous PNG only"
    module.supports_compile = lambda settings: True
    module.prepare = lambda request, parent: request
    module.saved_request = lambda attempt: attempt["request"]
    module.validate_result = lambda attempt: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(providers._PROVIDERS, "test", module.__name__)
    return module


class StatelessImages:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        assert request.continuation is None
        return GeneratedImage(
            png_bytes(step=len(self.calls)),
            metadata={"id": "vendor-id", "status": "vendor-status", "scene_step": 999},
        )


def test_stateless_compile_resume_and_metadata_isolation(stateless, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")
    scene = tmp_path / "scene.json"
    root = tmp_path / "build"
    write_json(
        scene,
        {
            "schema_version": 1,
            "brief": "A robot waves.",
            "settings": {"provider": "test"},
            "opening": {"prompt": "Robot"},
            "edits": [{"prompt": "Raise arm"}, {"prompt": "Wave"}],
        },
    )
    first = StatelessImages()
    assert compile_scene(scene, root, through=2, generator=first)["requests_made"] == 2
    resumed = StatelessImages()
    assert compile_scene(scene, root, generator=resumed)["requests_made"] == 1
    assert resumed.calls[0].base == root / "frames/f0002.png"
    again = compile_scene(scene, root)
    assert again["requests_made"] == 0
    assert again["reused_video"]
    attempts = read_json(root / "project.json")["attempts"]
    assert [a["scene_step"] for a in attempts] == [0, 1, 2]
    assert all(a["status"] == "succeeded" and a["continuation"] is None for a in attempts)
    assert attempts[0]["id"] == "a0001"
    assert attempts[0]["usage"] is None
    assert attempts[0]["provider_metadata"]["status"] == "vendor-status"
    stateless.workflow = lambda settings: "test.stateless.v2"
    with pytest.raises(HarnessError, match="workflow changed"):
        compile_scene(scene, root, generator=resumed)
    assert len(resumed.calls) == 1


def test_stateless_manual_branch_uses_selected_png(stateless, tmp_path):
    project = Project.create(tmp_path / "build", "A robot", provider="test")
    generator = StatelessImages()
    project.make_frame("Opening", generator=generator)
    project.make_frame("Rejected", base_frame="f0001", generator=generator)
    project.make_frame("Chosen", base_frame="f0001", generator=generator)
    assert generator.calls[-1].base == project.root / "frames/f0001.png"
    assert len(project.status()["frames"]) == 3


def test_error_metadata_cannot_overwrite_attempt_ledger(stateless, tmp_path):
    class Failing:
        def generate(self, request):
            raise ImageRequestError("Failure", metadata={"status": "succeeded", "id": "wrong"})

    project = Project.create(tmp_path / "build", "A robot", provider="test")
    with pytest.raises(ImageRequestError):
        project.make_frame("Opening", generator=Failing())
    attempt = project.status()["attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["id"] == "a0001"
    assert attempt["provider_metadata"]["status"] == "succeeded"


def test_offline_cli_does_not_import_vendor_sdks(tmp_path):
    brief = tmp_path / "brief.txt"
    brief.write_text("A robot")
    script = """
import sys
class NoVendorSDK:
    def find_spec(self, fullname, *args):
        if fullname == 'openai' or fullname.startswith('google'):
            raise AssertionError('Offline command imported a vendor SDK')
sys.meta_path.insert(0, NoVendorSDK())
from stop_motion.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "init", str(tmp_path / "build"), "--brief-file", str(brief)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["settings"]["provider"] == "openai"
    assert result.stderr == ""
