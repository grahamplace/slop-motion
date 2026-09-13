"""Project operations shared by the CLI and callers embedding the harness."""

import hashlib
import io
import math
import re
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .images import (
    ImageGenerator,
    ImageRequestError,
    create_generator,
    prepare_request,
    validate_saved,
)
from .settings import positive_integer, resolve_settings, validate_settings
from .storage import HarnessError, atomic_write, project_lock, read_json, write_json
from .video import export_video, next_artifact


def inspect_png(source, expected_size: tuple[int, int] | None = None) -> tuple[int, int]:
    try:
        with Image.open(source) as image:
            image.load()
            if image.format != "PNG":
                raise HarnessError("Expected a PNG image.")
            if expected_size is not None and image.size != expected_size:
                raise HarnessError(f"Image dimensions {image.size} do not match {expected_size}.")
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise HarnessError("V1 requires opaque images; transparency is not supported.")
            return image.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HarnessError(f"Could not decode PNG: {exc}") from exc


class Project:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.warnings: list[str] = []

    @classmethod
    def create(
        cls,
        root: Path | str,
        brief: str,
        *,
        model: str | None = None,
        size: str = "1024x1024",
        quality: str | None = None,
        fps: int = 24,
        max_image_requests: int = 20,
        backend: str | None = None,
        driver_model: str | None = None,
        provider: str = "openai",
        provider_options: dict | None = None,
    ):
        if not brief.strip():
            raise HarnessError("The scene brief cannot be empty.")
        supplied = {
            "provider": provider,
            "model": model,
            "size": size,
            "quality": quality,
            "fps": fps,
            "max_image_requests": max_image_requests,
            "backend": backend,
            "driver_model": driver_model,
            "provider_options": provider_options,
        }
        settings = resolve_settings(
            {key: value for key, value in supplied.items() if value is not None}
        )
        project = cls(root)
        project.root.mkdir(parents=True, exist_ok=True)
        with project_lock(project.root):
            if any(path.name != ".lock" for path in project.root.iterdir()):
                raise HarnessError("Initialize a new or empty directory; this directory has files.")
            for directory in ("frames", "previews", "exports"):
                (project.root / directory).mkdir()
            write_json(
                project.root / "project.json",
                {
                    "schema_version": 1,
                    "brief": brief,
                    "settings": settings,
                    "attempts": [],
                    "frames": [],
                    "timeline": [],
                },
            )
        return project

    def _save(self, data: dict):
        write_json(self.root / "project.json", data, replace=True)

    def _asset(self, frame: dict) -> Path:
        path = Path(frame["path"])
        resolved = (self.root / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(self.root / "frames"):
            raise HarnessError(
                f"Frame path must stay inside the project's frames directory: {path}"
            )
        return resolved

    @contextmanager
    def _session(self):
        self.warnings = []
        with project_lock(self.root):
            data = read_json(self.root / "project.json")
            self._validate_manifest(data)
            changed = False
            for attempt in data["attempts"]:
                if attempt["status"] == "started":
                    attempt["status"] = "unknown"
                    attempt["error"] = "Command stopped before its outcome was committed."
                    changed = True
                if attempt["status"] == "unknown":
                    self.warnings.append(
                        f"Attempt {attempt['id']} has an unknown outcome; not retried."
                    )
            if changed:
                self._save(data)
            known_paths = {self._asset(frame) for frame in data["frames"]}
            for path in (self.root / "frames").glob("*.png"):
                if path.resolve() not in known_paths:
                    self.warnings.append(f"Uncommitted image preserved: {path}")
            for path in known_paths:
                if not path.is_file():
                    self.warnings.append(f"Saved image is missing: {path}")
            yield data

    def _validate_manifest(self, data):
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise HarnessError("Invalid project manifest.")
        if data["schema_version"] != 1:
            raise HarnessError(f"Unsupported project schema version: {data['schema_version']}")
        try:
            if not isinstance(data["brief"], str) or not isinstance(data["settings"], dict):
                raise ValueError("brief or settings has the wrong type")
            validate_settings(data["settings"])
            for field in ("attempts", "frames", "timeline"):
                if not isinstance(data[field], list):
                    raise ValueError(f"{field} must be an array")
            for field, pattern in (("attempts", r"a\d+"), ("frames", r"f\d+")):
                ids = [item["id"] for item in data[field]]
                if any(
                    not isinstance(item, str) or not re.fullmatch(pattern, item) for item in ids
                ):
                    raise ValueError(f"invalid {field} IDs")
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate {field} IDs")
            for attempt in data["attempts"]:
                if attempt["status"] not in {"started", "succeeded", "failed", "unknown"}:
                    raise ValueError("invalid attempt status")
                if not re.fullmatch(r"f\d+", attempt["frame_id"]):
                    raise ValueError("invalid reserved frame ID")
            for frame in data["frames"]:
                self._asset(frame)
            self._validate_timeline(data["timeline"], data)
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessError(f"Invalid project manifest: {exc}") from exc

    @staticmethod
    def _frame(data: dict, frame_id: str) -> dict:
        for frame in data["frames"]:
            if frame["id"] == frame_id:
                return frame
        raise HarnessError(f"Unknown frame: {frame_id}")

    @staticmethod
    def _validate_timeline(entries, data: dict):
        if not isinstance(entries, list):
            raise HarnessError("Timeline must be an array of {frame_id, hold} entries.")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"frame_id", "hold"}:
                raise HarnessError("Each timeline entry must contain exactly frame_id and hold.")
            Project._frame(data, entry["frame_id"])
            if not positive_integer(entry["hold"]):
                raise HarnessError("Timeline holds must be positive integers, in output frames.")

    def status(self) -> dict:
        with self._session() as data:
            return {
                **data,
                "settings": resolve_settings(data["settings"], legacy_manifest=True),
                "project_path": str(self.root),
                "frames": [{**frame, "path": str(self._asset(frame))} for frame in data["frames"]],
                "request_count": len(data["attempts"]),
                "requests_remaining": max(
                    0, data["settings"]["max_image_requests"] - len(data["attempts"])
                ),
                "incomplete_attempts": [a for a in data["attempts"] if a["status"] == "unknown"],
                "duration_seconds": sum(e["hold"] for e in data["timeline"])
                / data["settings"]["fps"],
            }

    def make_frame(
        self,
        prompt: str,
        *,
        base_frame: str | None = None,
        references: list[str] | None = None,
        generator: ImageGenerator | None = None,
    ) -> dict:
        with self._session() as data:
            if "compile" in data:
                raise HarnessError("This is a compiled project. Edit its scene and run compile.")
            return self._make_frame(
                data, prompt, base_frame=base_frame, references=references, generator=generator
            )

    def _make_frame(
        self,
        data: dict,
        prompt: str,
        *,
        base_frame: str | None = None,
        references: list[str] | None = None,
        generator: ImageGenerator | None = None,
        scene_step: int | None = None,
    ) -> dict:
        if not prompt.strip():
            raise HarnessError("The frame prompt cannot be empty.")
        references = references or []
        settings = resolve_settings(data["settings"], legacy_manifest=True)
        dimensions = validate_settings(settings)
        ids = ([base_frame] if base_frame else []) + references
        if len(ids) != len(set(ids)):
            raise HarnessError("Each input frame should appear only once; the base comes first.")
        paths = [self._asset(self._frame(data, frame_id)) for frame_id in ids]
        for path in paths:
            inspect_png(path, dimensions)
        for frame_id, path in zip(ids, paths, strict=True):
            digest = self._frame(data, frame_id).get("sha256")
            if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise HarnessError(
                    "A saved base image changed; refusing to use stale response context."
                )
        if len(data["attempts"]) >= settings["max_image_requests"]:
            raise HarnessError("Image request budget exhausted; no request was sent.")
        parent = self._frame(data, base_frame) if base_frame else None
        source = next(
            (a for a in data["attempts"] if parent and a["id"] == parent.get("attempt_id")), None
        )
        if parent and parent.get("attempt_id") is not None and source is None:
            raise HarnessError("The selected base has no saved generation attempt.")
        request = prepare_request(
            settings,
            prompt,
            base=paths[0] if base_frame else None,
            references=paths[1:] if base_frame else paths,
            parent=source,
        )
        if generator is None:
            generator = create_generator(settings)
        number = max((int(a["id"][1:]) for a in data["attempts"]), default=0) + 1
        reserved = [a["frame_id"] for a in data["attempts"]]
        reserved += [frame["id"] for frame in data["frames"]]
        reserved += [p.stem for p in (self.root / "frames").glob("f*.png")]
        frame_number = (
            max(
                (int(item[1:]) for item in reserved if re.fullmatch(r"f\d+", item)),
                default=0,
            )
            + 1
        )
        frame_id = f"f{frame_number:04d}"
        attempt = {
            "id": f"a{number:04d}",
            "status": "started",
            "frame_id": frame_id,
            "base_frame_id": base_frame,
            "reference_frame_ids": references,
            "prompt": prompt,
            "provider": settings["provider"],
            "request": request.record(),
            "usage": None,
            "provider_request_id": None,
            "error": None,
            "scene_step": scene_step,
        }
        data["attempts"].append(attempt)
        self._save(data)
        try:
            generated = generator.generate(request)
            attempt["usage"] = generated.usage
            attempt["provider_request_id"] = generated.request_id
            attempt["provider_metadata"] = generated.metadata
            attempt["continuation"] = generated.continuation
            validate_saved(attempt, request)
            width, height = inspect_png(io.BytesIO(generated.png))
            path = self.root / "frames" / f"{frame_id}.png"
            atomic_write(path, generated.png)
        except ImageRequestError as exc:
            attempt.update(
                status="unknown" if exc.unknown else "failed",
                error=str(exc),
                provider_request_id=exc.request_id,
            )
            attempt["provider_metadata"] = exc.metadata
            self._save(data)
            raise
        except Exception as exc:
            # An unexpected/local error after sending a request must not invite a blind retry.
            attempt.update(status="unknown", error=str(exc))
            self._save(data)
            raise HarnessError(
                f"Attempt {attempt['id']} could not be committed: {exc}. "
                "Check status before retrying."
            ) from exc
        frame = {
            "id": frame_id,
            "path": str(path.relative_to(self.root)),
            "attempt_id": attempt["id"],
            "width": width,
            "height": height,
            "sha256": hashlib.sha256(generated.png).hexdigest(),
            "scene_step": scene_step,
        }
        data["frames"].append(frame)
        attempt["status"] = "succeeded"
        self._save(data)
        if (width, height) != dimensions:
            self.warnings.append(
                f"Saved image is {width}x{height}, expected {settings['size']}; "
                "inspect it before retrying."
            )
        return {**frame, "path": str(path), "request_count": len(data["attempts"])}

    def _import_opening(self, data: dict, png: bytes) -> dict:
        if data["frames"] or data["attempts"]:
            raise HarnessError("Importing an opening requires an empty project.")
        width, height = inspect_png(io.BytesIO(png), validate_settings(data["settings"]))
        path = self.root / "frames/f0001.png"
        atomic_write(path, png)
        frame = {
            "id": "f0001",
            "path": "frames/f0001.png",
            "attempt_id": None,
            "width": width,
            "height": height,
            "sha256": hashlib.sha256(png).hexdigest(),
            "scene_step": 0,
        }
        data["frames"].append(frame)
        self._save(data)
        return frame

    def view(self, frame_ids: list[str], *, contact_sheet: bool = False) -> dict:
        if not frame_ids:
            raise HarnessError("Select at least one frame to view.")
        with self._session() as data:
            paths = [self._asset(self._frame(data, frame_id)) for frame_id in frame_ids]
            for path in paths:
                inspect_png(path)
            result = {
                "frames": [
                    {"id": key, "path": str(path)}
                    for key, path in zip(frame_ids, paths, strict=True)
                ]
            }
            if contact_sheet:
                columns = min(4, len(paths))
                sheet = Image.new(
                    "RGB", (columns * 344, math.ceil(len(paths) / columns) * 376), "#eeeeee"
                )
                draw = ImageDraw.Draw(sheet)
                for index, (frame_id, path) in enumerate(zip(frame_ids, paths, strict=True)):
                    x, y = (index % columns) * 344, (index // columns) * 376
                    with Image.open(path) as source:
                        thumbnail = ImageOps.contain(source.convert("RGB"), (320, 320))
                        sheet.paste(thumbnail, (x + (344 - thumbnail.width) // 2, y + 12))
                    draw.text(
                        (x + 12, y + 344), f"{index + 1}. {frame_id}", fill="#111111", font_size=18
                    )
                preview = self.root / "previews"
                path = preview / f"{next_artifact(preview, 'contact-')}.png"
                content = io.BytesIO()
                sheet.save(content, format="PNG")
                atomic_write(path, content.getvalue())
                result["contact_sheet_path"] = str(path)
            return result

    def set_timeline(self, entries) -> dict:
        with self._session() as data:
            self._validate_timeline(entries, data)
            for frame_id in {entry["frame_id"] for entry in entries}:
                inspect_png(
                    self._asset(self._frame(data, frame_id)), validate_settings(data["settings"])
                )
            data["timeline"] = entries
            self._save(data)
            return {
                "timeline": entries,
                "fps": data["settings"]["fps"],
                "duration_seconds": sum(e["hold"] for e in entries) / data["settings"]["fps"],
            }

    def render(self) -> dict:
        with self._session() as data:
            if not data["timeline"]:
                raise HarnessError("Select a nonempty timeline before rendering.")
            dimensions = validate_settings(data["settings"])
            selected = list(dict.fromkeys(e["frame_id"] for e in data["timeline"]))
            frames = [self._frame(data, frame_id) for frame_id in selected]
            sources = {frame["id"]: self._asset(frame) for frame in frames}
            for path in sources.values():
                inspect_png(path, dimensions)
            snapshot = {
                "schema_version": 1,
                "settings": data["settings"],
                "timeline": data["timeline"],
                "frames": frames,
            }
            return export_video(self.root, snapshot, sources)
