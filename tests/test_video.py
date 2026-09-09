import json
import shutil
import subprocess
from pathlib import Path

import pytest

from stop_motion.storage import HarnessError

HAS_VIDEO_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.mark.skipif(not HAS_VIDEO_TOOLS, reason="ffmpeg/ffprobe are not installed")
def test_actual_encoder_preserves_order_holds_and_exports(project, generator):
    for index in range(13):
        project.make_frame(
            f"Pose {index}",
            base_frame=f"f{index:04d}" if index else None,
            generator=generator,
        )
    timeline = [{"frame_id": f"f{index:04d}", "hold": 3} for index in range(1, 14)]
    project.set_timeline(timeline)
    first = project.render()
    assert first["frame_count"] == 39
    assert first["duration_seconds"] == 1.625
    saved_video = Path(first["path"]).read_bytes()
    snapshot = json.loads(Path(first["snapshot_path"]).read_text())
    assert snapshot["timeline"] == timeline

    # Reverse the first two images and repeat one, so filename order cannot pass this check.
    replacement = [
        {"frame_id": "f0002", "hold": 2},
        {"frame_id": "f0001", "hold": 4},
        {"frame_id": "f0002", "hold": 3},
    ]
    project.set_timeline(replacement)
    second = project.render()
    assert second["frame_count"] == 9
    assert second["duration_seconds"] == 0.375
    assert Path(first["path"]).read_bytes() == saved_video
    assert first["path"] != second["path"]
    assert project.status()["request_count"] == 13

    decoded = subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-i",
            second["path"],
            "-vf",
            "crop=2:2:512:512",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    pixels = [decoded[index : index + 3] for index in range(0, len(decoded), 12)]
    assert len(pixels) == 9
    assert ["green" if green > red else "red" for red, green, blue in pixels] == (
        ["green"] * 2 + ["red"] * 4 + ["green"] * 3
    )
    assert not list((project.root / "exports").glob(".render-*"))


def test_missing_binaries_report_error_without_export(project, generator, monkeypatch):
    project.make_frame("Opening", generator=generator)
    project.set_timeline([{"frame_id": "f0001", "hold": 3}])
    monkeypatch.setattr("stop_motion.video.shutil.which", lambda _: None)
    with pytest.raises(HarnessError, match="Install ffmpeg"):
        project.render()
    assert list((project.root / "exports").iterdir()) == []


def test_empty_timeline_cannot_render(project):
    with pytest.raises(HarnessError, match="nonempty"):
        project.render()
