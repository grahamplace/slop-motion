import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from stop_motion.images import ImageRequestError
from stop_motion.project import Project
from stop_motion.storage import HarnessError, project_lock, read_json, write_json

from .conftest import FakeImages, png_bytes


def test_resume_thirteen_image_sequence_and_compare(project, generator):
    ids = []
    for index in range(7):
        result = project.make_frame(
            f"Pose {index}",
            base_frame=ids[-1] if ids else None,
            generator=generator,
        )
        ids.append(result["id"])
        project.set_timeline([{"frame_id": key, "hold": 3} for key in ids])
    original = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (project.root / "frames").glob("*.png")
    }

    resumed = Project(project.root)
    state = resumed.status()
    assert state["brief"] == "A clay butterfly lifts off."
    assert state["timeline"][-1]["frame_id"] == "f0007"
    assert state["request_count"] == 7
    for index in range(7, 13):
        result = resumed.make_frame(f"Pose {index}", base_frame=ids[-1], generator=generator)
        ids.append(result["id"])
    selected = [{"frame_id": key, "hold": 3} for key in ids]
    assert resumed.set_timeline(selected)["duration_seconds"] == 1.625
    for name, digest in original.items():
        assert hashlib.sha256((project.root / "frames" / name).read_bytes()).hexdigest() == digest
    assert len(generator.calls) == 13
    assert generator.calls[0][1] == []
    assert generator.calls[-1][1] == [project.root / "frames/f0012.png"]
    assert resumed.status()["attempts"][-1]["usage"] == {"total_tokens": 100}

    preview = resumed.view(ids, contact_sheet=True)
    assert len(preview["frames"]) == 13
    assert all(Path(frame["path"]).is_absolute() for frame in preview["frames"])
    with Image.open(preview["contact_sheet_path"]) as sheet:
        assert sheet.size == (1376, 1504)
    second_preview = resumed.view(ids[:2], contact_sheet=True)
    assert second_preview["contact_sheet_path"] != preview["contact_sheet_path"]


def test_retry_branches_from_selected_parent_and_preserves_candidates(project, generator):
    first = project.make_frame("Opening", generator=generator)
    rejected = project.make_frame("Too far", base_frame=first["id"], generator=generator)
    chosen = project.make_frame("Smaller move", base_frame=first["id"], generator=generator)
    project.set_timeline(
        [{"frame_id": first["id"], "hold": 3}, {"frame_id": chosen["id"], "hold": 3}]
    )
    state = project.status()
    assert len(state["frames"]) == 3
    assert Path(rejected["path"]).is_file()
    assert state["attempts"][-1]["base_frame_id"] == first["id"]
    assert generator.calls[-1][1] == [Path(first["path"])]


def test_base_and_references_are_passed_in_explicit_order(project, generator):
    project.make_frame("Opening", generator=generator)
    project.make_frame("Pose two", base_frame="f0001", generator=generator)
    project.make_frame("Pose three", base_frame="f0002", references=["f0001"], generator=generator)
    assert generator.calls[-1][1] == [
        project.root / "frames/f0002.png",
        project.root / "frames/f0001.png",
    ]
    project.make_frame("Alternative opening", references=["f0001"], generator=generator)
    assert generator.calls[-1][1] == [project.root / "frames/f0001.png"]


@pytest.mark.parametrize("hold", [True, False, 0, -1, 0.5, "3", None])
def test_invalid_holds_do_not_replace_saved_timeline(project, generator, hold):
    project.make_frame("Opening", generator=generator)
    original = [{"frame_id": "f0001", "hold": 3}]
    project.set_timeline(original)
    with pytest.raises(HarnessError, match="positive integers"):
        project.set_timeline([{"frame_id": "f0001", "hold": hold}])
    assert project.status()["timeline"] == original


def test_invalid_inputs_and_missing_key_do_not_consume_budget(project, generator, monkeypatch):
    with pytest.raises(HarnessError, match="Unknown frame"):
        project.make_frame("Next", base_frame="f9999", generator=generator)
    with pytest.raises(HarnessError, match="empty"):
        project.make_frame("   ", generator=generator)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(HarnessError, match="OPENAI_API_KEY"):
        project.make_frame("Opening")
    assert project.status()["request_count"] == 0
    assert generator.calls == []


@pytest.mark.parametrize("unknown", [False, True])
def test_failed_requests_consume_budget_without_retry(tmp_path, unknown):
    project = Project.create(tmp_path / "limited", "Scene", max_image_requests=1)

    class FailingImages:
        calls = 0

        def generate(self, request):
            self.calls += 1
            raise ImageRequestError("Provider failure", unknown=unknown, request_id="req_failed")

    generator = FailingImages()
    with pytest.raises(ImageRequestError):
        project.make_frame("Opening", generator=generator)
    state = project.status()
    assert state["request_count"] == 1
    assert state["attempts"][0]["status"] == ("unknown" if unknown else "failed")
    assert state["attempts"][0]["provider_request_id"] == "req_failed"
    assert state["attempts"][0]["usage"] is None
    with pytest.raises(HarnessError, match="budget exhausted"):
        project.make_frame("Explicit retry", generator=generator)
    assert generator.calls == 1


def test_interrupted_call_is_unknown_on_resume_and_does_not_reuse_id(project, generator):
    class InterruptedImages:
        def generate(self, request):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        project.make_frame("Opening", generator=InterruptedImages())
    restarted = Project(project.root)
    assert restarted.status()["incomplete_attempts"][0]["status"] == "unknown"
    assert len(restarted.warnings) == 1
    assert restarted.make_frame("Retry", generator=generator)["id"] == "f0002"
    assert len(generator.calls) == 1


def test_crash_between_image_and_manifest_preserves_orphan(project, generator, monkeypatch):
    save = project._save

    def fail_final_commit(data):
        if data["attempts"][-1]["status"] == "succeeded":
            raise OSError("Disk unavailable")
        save(data)

    monkeypatch.setattr(project, "_save", fail_final_commit)
    with pytest.raises(OSError, match="Disk unavailable"):
        project.make_frame("Opening", generator=generator)
    orphan = project.root / "frames/f0001.png"
    saved = orphan.read_bytes()
    restarted = Project(project.root)
    assert restarted.status()["frames"] == []
    assert any("Uncommitted image preserved" in warning for warning in restarted.warnings)
    assert restarted.make_frame("Retry", generator=generator)["id"] == "f0002"
    assert orphan.read_bytes() == saved


def test_lock_refuses_concurrent_commands_and_releases(project):
    with project_lock(project.root), pytest.raises(HarnessError, match="busy"):
        Project(project.root).status()
    assert project.status()["request_count"] == 0


def test_initialize_does_not_overwrite_existing_files(project):
    original = (project.root / "project.json").read_bytes()
    with pytest.raises(HarnessError, match="directory has files"):
        Project.create(project.root, "Replacement")
    assert (project.root / "project.json").read_bytes() == original


@pytest.mark.parametrize(
    "settings",
    [
        {"size": "512x512"},
        {"size": "1025x1024"},
        {"size": "0x0"},
        {"size": "auto"},
        {"model": "unknown"},
        {"fps": 0},
        {"max_image_requests": -1},
    ],
)
def test_invalid_settings_are_rejected_before_creating_project(tmp_path, settings):
    root = tmp_path / "invalid"
    with pytest.raises(HarnessError):
        Project.create(root, "Scene", **settings)
    assert not root.exists()


def test_wrong_size_is_saved_for_inspection_but_not_selected(project):
    result = project.make_frame("Opening", generator=FakeImages(size=(800, 1024)))
    assert Path(result["path"]).is_file()
    assert "expected 1024x1024" in project.warnings[0]
    with pytest.raises(HarnessError, match="dimensions"):
        project.set_timeline([{"frame_id": result["id"], "hold": 3}])
    assert project.view([result["id"]])["frames"][0]["id"] == result["id"]


def test_missing_input_fails_before_request(project, generator):
    result = project.make_frame("Opening", generator=generator)
    Path(result["path"]).unlink()
    with pytest.raises(HarnessError, match="decode PNG"):
        project.make_frame("Next", base_frame=result["id"], generator=generator)
    assert len(generator.calls) == 1
    assert project.status()["request_count"] == 1


def test_manifest_rejects_escaping_asset_path(project, generator):
    project.make_frame("Opening", generator=generator)
    path = project.root / "project.json"
    data = read_json(path)
    data["frames"][0]["path"] = "../../outside.png"
    write_json(path, data, replace=True)
    with pytest.raises(HarnessError, match="inside"):
        project.view(["f0001"])


@pytest.mark.parametrize("manifest", [[], {"schema_version": 99}, {"schema_version": 1}])
def test_invalid_manifest_is_actionable(project, manifest):
    (project.root / "project.json").write_text(json.dumps(manifest))
    with pytest.raises(HarnessError, match="manifest|schema"):
        project.status()


def test_unknown_orphan_id_is_not_overwritten(project, generator):
    orphan = project.root / "frames/f0009.png"
    original = png_bytes("blue")
    orphan.write_bytes(original)
    assert project.make_frame("Opening", generator=generator)["id"] == "f0010"
    assert orphan.read_bytes() == original
