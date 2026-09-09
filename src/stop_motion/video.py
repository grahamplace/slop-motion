"""Deterministic image holds, ffmpeg encoding, and ffprobe verification."""

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from .storage import HarnessError, write_json


def next_artifact(directory: Path, prefix: str) -> str:
    numbers = []
    for path in directory.iterdir():
        match = re.fullmatch(re.escape(prefix) + r"(\d+)\.[^.]+", path.name)
        if match:
            numbers.append(int(match[1]))
    return f"{prefix}{max(numbers, default=0) + 1:04d}"


def run_media(command: list[str]):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        raise HarnessError(f"{Path(command[0]).name} failed: {exc.stderr[-3000:].strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"{Path(command[0]).name} exceeded the 300-second time limit.") from exc


def export_video(root: Path, snapshot: dict, sources: dict[str, Path]) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise HarnessError("Install ffmpeg and ffprobe and make both available on PATH.")
    settings = snapshot["settings"]
    fps = settings["fps"]
    width, height = map(int, settings["size"].split("x"))
    count = sum(entry["hold"] for entry in snapshot["timeline"])
    duration = count / fps
    exports = root / "exports"
    name = next_artifact(exports, "export-")

    with tempfile.TemporaryDirectory(dir=exports, prefix=".render-") as staging:
        stage = Path(staging)
        index = 0
        for entry in snapshot["timeline"]:
            for _ in range(entry["hold"]):
                # Same filesystem, no image recompression, and almost no extra disk space.
                os.link(sources[entry["frame_id"]], stage / f"{index:08d}.png")
                index += 1
        video = stage / "output.mp4"
        run_media(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-framerate",
                str(fps),
                "-start_number",
                "0",
                "-i",
                str(stage / "%08d.png"),
                "-frames:v",
                str(count),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ]
        )
        probe = run_media(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration",
                "-of",
                "json",
                str(video),
            ]
        )
        try:
            metadata = json.loads(probe.stdout)
            stream = metadata["streams"][0]
            actual_duration = float(metadata["format"]["duration"])
            valid = (
                stream["codec_name"] == "h264"
                and stream["pix_fmt"] == "yuv420p"
                and (stream["width"], stream["height"]) == (width, height)
                and Fraction(stream["avg_frame_rate"]) == fps
                and int(stream["nb_read_frames"]) == count
                and math.isclose(actual_duration, duration, abs_tol=0.001, rel_tol=0)
            )
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise HarnessError("Could not verify the encoded video's metadata.") from exc
        if not valid:
            raise HarnessError(
                "Encoded video does not match the timeline; export was not published."
            )

        result = {
            "path": str(exports / f"{name}.mp4"),
            "snapshot_path": str(exports / f"{name}.json"),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": count,
            "duration_seconds": duration,
        }
        write_json(exports / f"{name}.json", {**snapshot, "output": result})
        os.link(video, exports / f"{name}.mp4")
        return result
