"""Compile agent-authored scene files; provider state stays in the build project."""

import hashlib
import io
import json
import shutil
from pathlib import Path

from .images import DEFAULT_DRIVER, ImageGenerator
from .project import DEFAULT_MODEL, Project, inspect_png, positive_integer, validate_settings
from .storage import HarnessError, read_json
from .video import export_video

DEFAULTS = {
    "model": DEFAULT_MODEL,
    "driver_model": DEFAULT_DRIVER,
    "backend": "responses",
    "size": "1024x1024",
    "quality": "medium",
    "fps": 24,
    "max_image_requests": 20,
}


def _object(value, allowed: set[str], label: str):
    if not isinstance(value, dict) or set(value) - allowed:
        raise HarnessError(f"{label} must be an object with only: {', '.join(sorted(allowed))}.")


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{label} must be a nonempty string.")
    return value


def _path(directory: Path, value) -> Path:
    return (directory / _text(value, "File path")).resolve()


def _prompt(value: dict, directory: Path, label: str) -> str:
    if ("prompt" in value) == ("prompt_file" in value):
        raise HarnessError(f"{label} needs exactly one of prompt or prompt_file.")
    if "prompt" in value:
        return _text(value["prompt"], label)
    return _text(_path(directory, value["prompt_file"]).read_text(encoding="utf-8"), label)


def load_scene(path: Path) -> dict:
    """Read and validate every local input before any network request or build write."""
    raw = read_json(path)
    _object(raw, {"schema_version", "brief", "brief_file", "settings", "opening", "edits"}, "Scene")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise HarnessError("Scene schema_version must be 1.")
    if ("brief" in raw) == ("brief_file" in raw):
        raise HarnessError("Scene needs exactly one of brief or brief_file.")
    directory = path.resolve().parent
    brief = (
        _text(raw["brief"], "Brief")
        if "brief" in raw
        else _text(_path(directory, raw["brief_file"]).read_text(encoding="utf-8"), "Brief")
    )
    supplied = raw.get("settings", {})
    _object(supplied, set(DEFAULTS), "Settings")
    settings = {**DEFAULTS, **supplied}
    dimensions = validate_settings(settings)
    if settings["backend"] != "responses":
        raise HarnessError("compile requires the uploaded-opening Responses workflow.")
    opening = raw.get("opening")
    _object(opening, {"prompt", "prompt_file", "image", "hold"}, "Opening")
    png = None
    if "image" in opening:
        if "prompt" in opening or "prompt_file" in opening:
            raise HarnessError("Opening needs an image OR a prompt, not both.")
        png = _path(directory, opening["image"]).read_bytes()
        inspect_png(io.BytesIO(png), dimensions)
        first = {"image_sha256": hashlib.sha256(png).hexdigest()}
    else:
        first = {"prompt": _prompt(opening, directory, "Opening")}
    steps = [first]
    holds = [opening.get("hold", 3)]
    edits = raw.get("edits", [])
    if not isinstance(edits, list):
        raise HarnessError("edits must be an array.")
    for index, edit in enumerate(edits, 1):
        _object(edit, {"prompt", "prompt_file", "hold"}, f"Edit {index}")
        steps.append({"prompt": _prompt(edit, directory, f"Edit {index}")})
        holds.append(edit.get("hold", 3))
    if not all(positive_integer(hold) for hold in holds):
        raise HarnessError("Every hold must be a positive integer in output video frames.")
    requests = len(steps) - (png is not None)
    if requests > settings["max_image_requests"]:
        raise HarnessError(
            f"Scene requires {requests} requests, exceeding max_image_requests "
            f"({settings['max_image_requests']}). No request sent."
        )
    return {
        "brief": brief,
        "settings": settings,
        "steps": steps,
        "holds": holds,
        "opening_png": png,
        "scene_path": str(path.resolve()),
    }


def _check_resume(project: Project, data: dict, scene: dict) -> int:
    project._validate_manifest(data)
    saved = data.get("compile")
    if saved is None:
        if data["frames"] or data["attempts"]:
            raise HarnessError("compile needs a new/empty project or its own previous build.")
    elif (
        not isinstance(saved, dict)
        or not isinstance(saved.get("steps"), list)
        or len(saved["steps"]) < len(data["frames"])
    ):
        raise HarnessError("Invalid saved compile state; inspect the project manifest.")
    for key in DEFAULTS.keys() - {"fps", "max_image_requests"}:
        if data["settings"].get(key) != scene["settings"][key]:
            raise HarnessError(f"Generation setting {key} changed; use a new project directory.")
    for attempt in data["attempts"]:
        if attempt["status"] != "succeeded":
            raise HarnessError(
                f"Attempt {attempt['id']} is {attempt['status']}; refusing to replay it. "
                "Inspect status and saved files before explicitly starting a new build."
            )
    completed = len(data["frames"])
    if {a["id"] for a in data["attempts"]} != {
        f.get("attempt_id") for f in data["frames"] if f.get("attempt_id") is not None
    }:
        raise HarnessError("Saved attempts and committed frames disagree; refusing to replay.")
    if completed > len(scene["steps"]):
        raise HarnessError("Scene removes completed poses; use a new project directory.")
    for index, frame in enumerate(data["frames"]):
        if frame.get("scene_step") != index or saved["steps"][index] != scene["steps"][index]:
            raise HarnessError(f"Completed pose {index + 1} changed; use a new project directory.")
        path = project._asset(frame)
        inspect_png(path, validate_settings(scene["settings"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame.get("sha256"):
            raise HarnessError(f"Saved frame {frame['id']} changed; cannot resume its edit chain.")
        if index or "prompt" in scene["steps"][0]:
            attempt = next((a for a in data["attempts"] if a["id"] == frame["attempt_id"]), None)
            if not attempt or not attempt.get("response_id"):
                raise HarnessError("Missing response metadata; cannot resume this build.")
            parent = data["frames"][index - 1]["id"] if index else None
            previous = None
            if index > 1:
                previous = next(
                    a["response_id"]
                    for a in data["attempts"]
                    if a["id"] == data["frames"][index - 1]["attempt_id"]
                )
            if (
                attempt["base_frame_id"] != parent
                or attempt["request"].get("previous_response_id") != previous
                or attempt["prompt"] != scene["steps"][index]["prompt"]
                or attempt.get("scene_step") != index
                or attempt["request"].get("action") != ("edit" if index else "generate")
                or any(
                    attempt["request"].get(key) != scene["settings"][key]
                    for key in ("model", "driver_model", "quality", "size")
                )
            ):
                raise HarnessError("Saved response chain is inconsistent.")
    committed = {project._asset(frame) for frame in data["frames"]}
    if any(p.resolve() not in committed for p in (project.root / "frames").glob("*.png")):
        raise HarnessError("Uncommitted image preserved; inspect it before starting a new build.")
    return completed


def _plan(scene: dict, root: Path, completed: int, through: int | None) -> dict:
    total = len(scene["steps"])
    if through is not None and (not positive_integer(through) or through > total):
        raise HarnessError(
            f"--through must be a pose count from 1 to {total}, including the opening."
        )
    target = total if through is None else through
    requests = max(0, target - completed)
    if not completed and scene["opening_png"] is not None:
        requests -= 1
    return {
        "project_path": str(root),
        "scene_path": scene["scene_path"],
        "settings": scene["settings"],
        "total_poses": total,
        "target_poses": target,
        "completed_poses": completed,
        "remaining_requests": requests,
        "duration_seconds": sum(scene["holds"][:target]) / scene["settings"]["fps"],
        "workflow": "separate-opening -> uploaded-PNG-first-edit -> previous_response_id",
    }


def compile_scene(
    scene_path: Path,
    root: Path,
    *,
    dry_run: bool = False,
    through: int | None = None,
    generator: ImageGenerator | None = None,
    progress=None,
) -> dict:
    """Validate, resume a serial shot, and render it. Dry runs do not write or need a key."""
    scene = load_scene(scene_path)
    project = Project(root)
    manifest = project.root / "project.json"
    if manifest.exists():
        completed = _check_resume(project, read_json(manifest), scene)
    else:
        completed = 0
        if project.root.exists() and any(p.name != ".lock" for p in project.root.iterdir()):
            raise HarnessError("Build directory is not empty; use a new project directory.")
    plan = _plan(scene, project.root, completed, through)
    if dry_run:
        return {**plan, "dry_run": True, "warnings": []}
    # Discover missing video tools before making paid requests.
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise HarnessError("Install ffmpeg and ffprobe before compiling.")
    if not manifest.exists():
        Project.create(project.root, scene["brief"], **scene["settings"])
    with project._session() as data:
        completed = _check_resume(project, data, scene)
        plan = _plan(scene, project.root, completed, through)
        if (
            len(data["attempts"]) + plan["remaining_requests"]
            > scene["settings"]["max_image_requests"]
        ):
            raise HarnessError("Remaining compile requests exceed the budget; no request sent.")
        data["settings"] = scene["settings"]
        data["brief"] = scene["brief"]
        state = data.setdefault("compile", {})
        state.update(steps=scene["steps"], scene_path=scene["scene_path"])
        project._save(data)
        for index in range(completed, plan["target_poses"]):
            if progress:
                progress(
                    {"event": "pose_started", "pose": index + 1, "total": plan["target_poses"]}
                )
            if index == 0 and scene["opening_png"] is not None:
                project._import_opening(data, scene["opening_png"])
            else:
                project._make_frame(
                    data,
                    scene["steps"][index]["prompt"],
                    base_frame=data["frames"][index - 1]["id"] if index else None,
                    generator=generator,
                    scene_step=index,
                )
            # Wrong-sized outputs remain inspectable, but must stop the paid sequence.
            inspect_png(project._asset(data["frames"][-1]), validate_settings(scene["settings"]))
            if progress:
                progress({"event": "pose_saved", "frame": data["frames"][-1]})
        frames = data["frames"][: plan["target_poses"]]
        data["timeline"] = [
            {"frame_id": frame["id"], "hold": hold}
            for frame, hold in zip(frames, scene["holds"], strict=False)
        ]
        project._save(data)
        snapshot = {
            "schema_version": 1,
            "settings": data["settings"],
            "timeline": data["timeline"],
            "frames": frames,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "timeline": data["timeline"],
                    "frames": frames,
                    "fps": scene["settings"]["fps"],
                    "size": scene["settings"]["size"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = state.get("export", {})
        output = cached.get("output")
        reusable = False
        if output and cached.get("fingerprint") == fingerprint:
            video = Path(output["path"])
            reusable = (
                video.resolve().parent == (project.root / "exports").resolve()
                and video.is_file()
                and Path(output["snapshot_path"]).is_file()
                and hashlib.sha256(video.read_bytes()).hexdigest() == cached.get("sha256")
            )
        if not reusable:
            output = export_video(
                project.root, snapshot, {frame["id"]: project._asset(frame) for frame in frames}
            )
            state["export"] = {
                "fingerprint": fingerprint,
                "output": output,
                "sha256": hashlib.sha256(Path(output["path"]).read_bytes()).hexdigest(),
            }
            project._save(data)
        return {
            **plan,
            "dry_run": False,
            "completed_poses": len(data["frames"]),
            "requests_made": plan["remaining_requests"],
            "request_count": len(data["attempts"]),
            "remaining_requests": 0,
            "frames": [{**frame, "path": str(project._asset(frame))} for frame in frames],
            "video": output,
            "reused_video": reusable,
            "warnings": project.warnings,
        }
