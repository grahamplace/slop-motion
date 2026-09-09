"""Compare six identical edits with high input fidelity or high output quality.

Uses the existing first seven butterfly frames as the baseline. No core harness
defaults are changed. Run --prepare to validate and build the baseline report
without contacting OpenAI. A completed or failed request is never replayed.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

from stop_motion.project import inspect_png
from stop_motion.storage import HarnessError, atomic_write, project_lock, read_json, write_json
from stop_motion.video import export_video

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "projects/butterfly"
OUTPUT = REPO / "projects/butterfly-settings-test"
EDIT_COUNT = 6
VARIANTS = {
    "high-input-fidelity": {"input_fidelity": "high"},
    "high-output-quality": {"quality": "high"},
}
REGIONS = {
    "background": (16, 16, 256, 192),
    "tabletop": (624, 832, 1008, 1008),
}


def load_key():
    # Read one named value as data; never execute/source the .env contents.
    if not os.environ.get("OPENAI_API_KEY"):
        env_file = REPO / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    value = line.partition("=")[2].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    os.environ["OPENAI_API_KEY"] = value
    if not os.environ.get("OPENAI_API_KEY"):
        raise HarnessError("API key unavailable. Run /private/tmp/stop-motion-key-setup.sh first.")


def redact(message):
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        message = message.replace(key, "<REDACTED>")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "<REDACTED>", message)


def prepare():
    source = read_json(SOURCE / "project.json")
    selected = source["timeline"][: EDIT_COUNT + 1]
    if len(selected) != EDIT_COUNT + 1:
        raise HarnessError("The source shot needs an opening frame plus six selected edits.")
    by_id = {frame["id"]: frame for frame in source["frames"]}
    attempts = {attempt["id"]: attempt for attempt in source["attempts"]}
    files = [SOURCE / by_id[entry["frame_id"]]["path"] for entry in selected]
    for path in files:
        inspect_png(path, (1024, 1024))
    requests = []
    for previous, entry in zip(selected, selected[1:], strict=False):
        attempt = attempts[by_id[entry["frame_id"]]["attempt_id"]]
        if attempt["base_frame_id"] != previous["frame_id"] or attempt["reference_frame_ids"]:
            raise HarnessError("Baseline must be a chain of edits without extra references.")
        request = attempt["request"]
        if request["quality"] != "medium" or "input_fidelity" in request:
            raise HarnessError(
                "Expected medium quality and omitted input_fidelity in the baseline."
            )
        requests.append(request.copy())
    plan = {
        "source_project": str(SOURCE),
        "source_frame_ids": [entry["frame_id"] for entry in selected],
        "source_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in files],
        "baseline_requests": requests,
        "variants": VARIANTS,
        "maximum_new_requests": EDIT_COUNT * len(VARIANTS),
        "note": "One chain per condition; this comparison is exploratory, not a repeated trial.",
    }
    plan_path = OUTPUT / "plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise HarnessError(
                "Saved comparison plan differs from the current source; refusing reuse."
            )
    else:
        write_json(plan_path, plan)
    for name in ("baseline", *VARIANTS):
        directory = OUTPUT / name
        directory.mkdir(exist_ok=True)
        (directory / "exports").mkdir(exist_ok=True)
        copies = files if name == "baseline" else files[:1]
        for index, source_file in enumerate(copies):
            target = directory / f"{index:02d}.png"
            if not target.exists():
                atomic_write(target, source_file.read_bytes())
        if name in VARIANTS and not (directory / "attempts.json").exists():
            write_json(directory / "attempts.json", [])
    return plan


def run_variant(client, name, changes, requests):
    directory = OUTPUT / name
    record_path = directory / "attempts.json"
    attempts = read_json(record_path)
    if attempts and attempts[-1]["status"] != "succeeded":
        print(
            f"{name}: stopped at a prior incomplete/failed request; not replaying it.", flush=True
        )
        return
    for index in range(len(attempts), EDIT_COUNT):
        request = {**requests[index], **changes}
        target = directory / f"{index + 1:02d}.png"
        parent = directory / f"{index:02d}.png"
        if target.exists():
            raise HarnessError(f"Uncommitted output already exists: {target}")
        attempt = {
            "step": index + 1,
            "status": "started",
            "request": request,
            "input": parent.name,
            "output": target.name,
            "provider_request_id": None,
            "usage": None,
            "response_metadata": {},
            "error": None,
        }
        attempts.append(attempt)
        write_json(record_path, attempts, replace=True)
        print(f"{name}: edit {index + 1}/{EDIT_COUNT}...", flush=True)
        try:
            with parent.open("rb") as image_file:
                response = client.images.edit(**request, image=[image_file])
            attempt["provider_request_id"] = getattr(response, "_request_id", None)
            attempt["usage"] = response.usage.model_dump(mode="json") if response.usage else None
            attempt["response_metadata"] = {
                key: getattr(response, key)
                for key in ("model", "quality", "size", "background", "output_format")
                if getattr(response, key, None) is not None
            }
            png = base64.b64decode(response.data[0].b64_json, validate=True)
            inspect_png(io.BytesIO(png), (1024, 1024))
            atomic_write(target, png)
            attempt["status"] = "succeeded"
        except APIStatusError as exc:
            attempt.update(
                status="failed",
                http_status=exc.status_code,
                provider_request_id=exc.request_id,
                error=redact(str(exc)),
            )
            write_json(record_path, attempts, replace=True)
            print(f"{name}: {attempt['error']}", flush=True)
            if exc.status_code in (401, 403, 429):
                raise HarnessError(
                    "Authentication/access/rate error; stopped further requests."
                ) from exc
            return
        except APIConnectionError as exc:
            attempt.update(status="unknown", error="Connection failed; remote outcome unknown.")
            write_json(record_path, attempts, replace=True)
            raise HarnessError(attempt["error"]) from exc
        except Exception as exc:
            attempt.update(status="unknown", error=redact(str(exc)))
            write_json(record_path, attempts, replace=True)
            raise
        write_json(record_path, attempts, replace=True)
        print(f"{name}: saved {target.name}", flush=True)


def texture(gray):
    width, height = gray.size
    horizontal = ImageChops.difference(
        gray.crop((1, 0, width, height)), gray.crop((0, 0, width - 1, height))
    )
    vertical = ImageChops.difference(
        gray.crop((0, 1, width, height)), gray.crop((0, 0, width, height - 1))
    )
    return (ImageStat.Stat(horizontal).mean[0] + ImageStat.Stat(vertical).mean[0]) / 2


def report():
    baseline_start = Image.open(OUTPUT / "baseline/00.png").convert("RGB")
    metrics = {}
    for name in ("baseline", *VARIANTS):
        frames = sorted((OUTPUT / name).glob("[0-9][0-9].png"))
        steps = []
        for path in frames:
            with Image.open(path) as source:
                image = source.convert("RGB")
                regions = {}
                for region, bounds in REGIONS.items():
                    original = baseline_start.crop(bounds)
                    current = image.crop(bounds)
                    difference = ImageStat.Stat(ImageChops.difference(original, current)).mean
                    regions[region] = {
                        "mean_absolute_rgb_difference": round(sum(difference) / 3, 4),
                        "texture": round(texture(current.convert("L")), 4),
                    }
                steps.append({"step": int(path.stem), "regions": regions})
        metrics[name] = steps
    write_json(
        OUTPUT / "metrics.json",
        {
            "regions": REGIONS,
            "note": "Static-region differences and texture describe drift, not overall quality.",
            "conditions": metrics,
        },
        replace=True,
    )

    sheet = Image.new("RGB", (4 * 280, 3 * 310), "#ededed")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(("baseline", *VARIANTS)):
        for column, step in enumerate((0, 2, 4, 6)):
            path = OUTPUT / name / f"{step:02d}.png"
            x, y = column * 280, row * 310
            if path.is_file():
                with Image.open(path) as source:
                    sheet.paste(
                        ImageOps.contain(source.convert("RGB"), (256, 256)), (x + 12, y + 8)
                    )
            label = f"{name}\nEdit {step}" if path.is_file() else f"{name}\nNot generated"
            draw.multiline_text((x + 12, y + 269), label, fill="#111111", font_size=13)
    content = io.BytesIO()
    sheet.save(content, format="PNG")
    atomic_write(OUTPUT / "comparison.png", content.getvalue(), replace=True)
    return metrics


def render_conditions(plan):
    for name in ("baseline", *VARIANTS):
        directory = OUTPUT / name
        if not (directory / f"{EDIT_COUNT:02d}.png").is_file():
            continue
        snapshot = {
            "schema_version": 1,
            "settings": {"size": "1024x1024", "fps": 24},
            "timeline": [{"frame_id": f"{step:02d}", "hold": 3} for step in range(EDIT_COUNT + 1)],
            "changes_from_baseline": VARIANTS.get(name, {}),
            "source_project": plan["source_project"],
        }
        sources = {f"{step:02d}": directory / f"{step:02d}.png" for step in range(EDIT_COUNT + 1)}
        result = export_video(directory, snapshot, sources)
        print(f"{name} video: {result['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare", action="store_true", help="Prepare and report without API calls"
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with project_lock(OUTPUT):
        plan = prepare()
        metrics = report()
        if args.prepare:
            print(
                json.dumps(
                    {
                        "output": str(OUTPUT),
                        "maximum_new_requests": plan["maximum_new_requests"],
                        "variants": VARIANTS,
                        "baseline": metrics["baseline"],
                    },
                    indent=2,
                )
            )
            return
        load_key()
        with OpenAI(max_retries=0, timeout=180.0) as client:
            for name, changes in VARIANTS.items():
                run_variant(client, name, changes, plan["baseline_requests"])
                report()
        render_conditions(plan)
        print(f"Comparison: {OUTPUT / 'comparison.png'}", flush=True)
        print(f"Metrics: {OUTPUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(
            "Stopped; any incomplete request remains recorded and will not be replayed."
        ) from None
    except Exception as exc:
        raise SystemExit(redact(str(exc))) from None
