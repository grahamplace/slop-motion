"""Bounded, resumable six-edit Responses API experiment.

Run with: .venv/bin/python -m scripts.compare_responses [--prepare | --report]
No retries or model fallbacks. --prepare and --report make no API calls.
"""

import argparse
import base64
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

from scripts.compare_edit_settings import REGIONS, load_key, redact, texture
from stop_motion.project import inspect_png
from stop_motion.storage import HarnessError, atomic_write, project_lock, read_json, write_json
from stop_motion.video import export_video

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "projects/butterfly-settings-test"
OUTPUT = REPO / "projects/butterfly-responses-test"
EDIT_COUNT = 6
DRIVER_MODEL = "gpt-5.5"
INSTRUCTIONS = (
    "For each user request, call the image_generation tool exactly once to edit the latest "
    "image in the conversation. Return the image without commentary."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    original = read_json(BASELINE / "plan.json")
    requests = original["baseline_requests"]
    expected_settings = {
        "model": "gpt-image-2.5-sunburst",
        "size": "1024x1024",
        "quality": "medium",
        "background": "opaque",
        "output_format": "png",
        "n": 1,
    }
    if len(requests) != EDIT_COUNT or len(original["source_sha256"]) != EDIT_COUNT + 1:
        raise HarnessError("Expected the original six-edit comparison baseline.")
    for request in requests:
        if {key: value for key, value in request.items() if key != "prompt"} != expected_settings:
            raise HarnessError("Baseline settings differ; refusing a mismatched comparison.")
    for index, digest in enumerate(original["source_sha256"]):
        path = BASELINE / "baseline" / f"{index:02d}.png"
        if sha256(path) != digest:
            raise HarnessError(f"Baseline frame changed: {path}")
    plan = {
        "baseline": str(BASELINE),
        "baseline_sha256": original["source_sha256"],
        "baseline_requests": requests,
        "driver_model": DRIVER_MODEL,
        "instructions": INSTRUCTIONS,
        "maximum_response_requests": EDIT_COUNT,
        "max_tool_calls_per_response": 1,
        "image_tool": {
            "type": "image_generation",
            "action": "edit",
            **{key: value for key, value in expected_settings.items() if key != "n"},
        },
        "note": (
            "One exploratory chain. Same original and user prompts; Responses introduces "
            "a text-model orchestrator, prompt rewriting, and persistent conversation context."
        ),
    }
    path = OUTPUT / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise HarnessError("Saved plan differs; refusing to reuse this experiment.")
    else:
        write_json(path, plan)
    seed = OUTPUT / "00.png"
    if not seed.exists():
        atomic_write(seed, (BASELINE / "baseline/00.png").read_bytes())
    if sha256(seed) != plan["baseline_sha256"][0]:
        raise HarnessError("Experiment seed differs from the original.")
    if not (OUTPUT / "attempts.json").exists():
        write_json(OUTPUT / "attempts.json", [])
    return plan


def latest_image_id(plan, index, attempts):
    if index == 0:
        image_id = plan.get("seed_image_id")
    else:
        calls = attempts[index - 1].get("image_calls", [])
        if len(calls) != 1 or calls[0].get("status") != "completed":
            raise HarnessError("Expected exactly one completed previous image reference.")
        image_id = calls[0].get("id")
    if not isinstance(image_id, str) or not image_id.startswith("ig_"):
        raise HarnessError("Missing or invalid latest image reference.")
    return image_id


def build_request(plan, index, attempts, *, output=None):
    output = OUTPUT if output is None else output
    prompt = plan["baseline_requests"][index]["prompt"]
    request = {
        "model": plan["driver_model"],
        "instructions": plan["instructions"],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 4096,
        "tools": [plan["image_tool"]],
        "tool_choice": {"type": "image_generation"},
        "max_tool_calls": 1,
        "parallel_tool_calls": False,
        "store": True,
    }
    if plan.get("context_mode") == "latest_image_id":
        request["input"] = [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            {"type": "image_generation_call", "id": latest_image_id(plan, index, attempts)},
        ]
    elif index == 0 and plan.get("generates_opening", False):
        request["tools"] = [{**plan["image_tool"], "action": "generate"}]
        request["input"] = prompt
    elif index == 0:
        encoded = base64.b64encode((output / "00.png").read_bytes()).decode("ascii")
        request["input"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded}",
                        "detail": "high",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ]
    else:
        request["previous_response_id"] = attempts[index - 1]["response_id"]
        request["input"] = prompt
    return request


def validate_attempts(
    attempts, *, output=None, request_count=EDIT_COUNT, frame_offset=1, plan=None
):
    output = OUTPUT if output is None else output
    if len(attempts) > request_count:
        raise HarnessError("Saved attempt count exceeds the budget.")
    for index, attempt in enumerate(attempts):
        if attempt["step"] != index + 1 or attempt["status"] != "succeeded":
            raise HarnessError("A prior request is incomplete/failed; refusing to replay it.")
        latest_only = plan is not None and plan.get("context_mode") == "latest_image_id"
        expected_parent = attempts[index - 1]["response_id"] if index and not latest_only else None
        if attempt["previous_response_id"] != expected_parent:
            raise HarnessError("Stored response chain is inconsistent.")
        if latest_only and attempt.get("input_image_ids") != [
            latest_image_id(plan, index, attempts)
        ]:
            raise HarnessError("Stored latest image reference is inconsistent.")
        if not attempt.get("response_id"):
            raise HarnessError("Stored request is missing its response ID.")
        if sha256(output / f"{index + frame_offset:02d}.png") != attempt["output_sha256"]:
            raise HarnessError("A saved frame changed; refusing to continue.")


def run_chain(client, plan, *, output=None):
    output = OUTPUT if output is None else output
    context_mode = plan.get("context_mode", "response_chain")
    if context_mode not in ("response_chain", "latest_image_id"):
        raise HarnessError("Unknown context mode; refusing to use a different input strategy.")
    if context_mode == "latest_image_id":
        if plan.get("generates_opening") or plan["image_tool"]["action"] != "edit":
            raise HarnessError(
                "Latest-image mode requires an existing seed and edit-only requests."
            )
        latest_image_id(plan, 0, [])
        if sha256(output / "00.png") != plan.get("seed_sha256"):
            raise HarnessError("Latest-image seed differs from the saved plan.")
    request_count = plan.get("maximum_response_requests", EDIT_COUNT)
    if (
        type(request_count) is not int
        or request_count < 1
        or len(plan["baseline_requests"]) != request_count
    ):
        raise HarnessError("Request plan must exactly match its explicit positive budget.")
    frame_offset = 0 if plan.get("generates_opening", False) else 1
    record = output / "attempts.json"
    attempts = read_json(record)
    validate_attempts(
        attempts, output=output, request_count=request_count, frame_offset=frame_offset, plan=plan
    )
    for index in range(len(attempts), request_count):
        target = output / f"{index + frame_offset:02d}.png"
        if target.exists():
            raise HarnessError(f"Uncommitted output exists: {target}")
        request = build_request(plan, index, attempts, output=output)
        attempt = {
            "step": index + 1,
            "status": "started",
            "started_at": datetime.now(UTC).isoformat(),
            "previous_response_id": request.get("previous_response_id"),
            "prompt": plan["baseline_requests"][index]["prompt"],
            "image_tool": request["tools"][0],
            "response_id": None,
            "provider_request_id": None,
            "error": None,
        }
        if context_mode == "latest_image_id":
            attempt["context_mode"] = context_mode
            attempt["input_image_ids"] = [request["input"][1]["id"]]
        attempts.append(attempt)
        write_json(record, attempts, replace=True)
        print(
            f"{output.name}: {request['tools'][0]['action']} {index + 1}/{request_count}...",
            flush=True,
        )
        try:
            response = client.responses.create(**request)
            attempt.update(
                response_id=response.id,
                provider_request_id=getattr(response, "_request_id", None),
                response_status=response.status,
                response_model=response.model,
                response_usage=response.usage.model_dump(mode="json") if response.usage else None,
                response_error=(response.error.model_dump(mode="json") if response.error else None),
                incomplete_details=(
                    response.incomplete_details.model_dump(mode="json")
                    if response.incomplete_details
                    else None
                ),
            )
            calls = [item for item in response.output if item.type == "image_generation_call"]
            attempt["image_calls"] = [
                call.model_dump(mode="json", exclude={"result"}) for call in calls
            ]
            # Persist remote IDs and metadata even if validation/publication fails.
            write_json(record, attempts, replace=True)
            if response.status != "completed" or len(calls) != 1:
                raise HarnessError("Response did not complete with exactly one image call.")
            call = calls[0]
            if call.status != "completed" or not call.result:
                raise HarnessError("Image tool did not return a completed image.")
            for key in ("model", "action", "quality", "size", "background", "output_format"):
                actual = getattr(call, key, None)
                if actual is not None and actual != request["tools"][0][key]:
                    raise HarnessError(f"Returned image {key} differs: {actual}")
            png = base64.b64decode(call.result, validate=True)
            inspect_png(io.BytesIO(png), (1024, 1024))
            atomic_write(target, png)
            attempt.update(status="succeeded", output_sha256=sha256(target))
        except APIStatusError as exc:
            attempt.update(
                status="unknown" if exc.status_code >= 500 else "failed",
                http_status=exc.status_code,
                provider_request_id=exc.request_id,
                error=redact(str(exc)),
            )
            raise HarnessError(attempt["error"]) from exc
        except APIConnectionError as exc:
            attempt.update(status="unknown", error="Connection failed; remote outcome unknown.")
            raise HarnessError(attempt["error"]) from exc
        except Exception as exc:
            attempt.update(status="unknown", error=redact(str(exc)))
            raise
        finally:
            attempt["finished_at"] = datetime.now(UTC).isoformat()
            write_json(record, attempts, replace=True)
        print(f"{output.name}: saved {target.name}", flush=True)


def report():
    directories = {"images-api-medium": BASELINE / "baseline", "responses-medium": OUTPUT}
    with Image.open(OUTPUT / "00.png") as seed:
        original = seed.convert("RGB")
    metrics = {}
    sheet = Image.new("RGB", (1120, 620), "#ededed")
    draw = ImageDraw.Draw(sheet)
    for row, (name, directory) in enumerate(directories.items()):
        steps = []
        for path in sorted(directory.glob("[0-9][0-9].png")):
            with Image.open(path) as source:
                current = source.convert("RGB")
            regions = {}
            for region, bounds in REGIONS.items():
                crop = current.crop(bounds)
                diff = ImageChops.difference(original.crop(bounds), crop)
                regions[region] = {
                    "mean_absolute_rgb_difference": round(sum(ImageStat.Stat(diff).mean) / 3, 4),
                    "texture": round(texture(crop.convert("L")), 4),
                }
            steps.append({"step": int(path.stem), "regions": regions})
        metrics[name] = steps
        for column, step in enumerate((0, 2, 4, 6)):
            path = directory / f"{step:02d}.png"
            x, y = column * 280, row * 310
            if path.exists():
                with Image.open(path) as source:
                    sheet.paste(
                        ImageOps.contain(source.convert("RGB"), (256, 256)), (x + 12, y + 8)
                    )
            label = f"{name}\nEdit {step}" if path.exists() else f"{name}\nNot generated"
            draw.multiline_text((x + 12, y + 269), label, fill="#111111", font_size=13)
    write_json(
        OUTPUT / "metrics.json",
        {
            "regions": REGIONS,
            "note": "Static-region difference is not a perceptual quality score.",
            "conditions": metrics,
        },
        replace=True,
    )
    content = io.BytesIO()
    sheet.save(content, format="PNG")
    atomic_write(OUTPUT / "comparison.png", content.getvalue(), replace=True)


def render(plan):
    if not (OUTPUT / "06.png").exists():
        return
    attempts = read_json(OUTPUT / "attempts.json")
    validate_attempts(attempts)
    if len(attempts) != EDIT_COUNT:
        return
    exports = OUTPUT / "exports"
    exports.mkdir(exist_ok=True)
    if (exports / "export-0001.mp4").exists():
        return
    snapshot = {
        "schema_version": 1,
        "settings": {"size": "1024x1024", "fps": 24},
        "timeline": [{"frame_id": f"{step:02d}", "hold": 3} for step in range(EDIT_COUNT + 1)],
        "experiment": plan,
    }
    result = export_video(
        OUTPUT,
        snapshot,
        {f"{step:02d}": OUTPUT / f"{step:02d}.png" for step in range(EDIT_COUNT + 1)},
    )
    print(f"Video: {result['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with project_lock(OUTPUT):
        plan = prepare()
        if args.prepare or args.report:
            report()
            if args.report:
                render(plan)
            print(
                json.dumps(
                    {
                        "output": str(OUTPUT),
                        "maximum_new_requests": EDIT_COUNT,
                        "driver_model": DRIVER_MODEL,
                        "image_tool": plan["image_tool"],
                    }
                )
            )
            return
        load_key()
        try:
            with OpenAI(max_retries=0, timeout=240.0) as client:
                run_chain(client, plan)
        finally:
            report()
        render(plan)
        print(f"Comparison: {OUTPUT / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped; incomplete requests will not be replayed.") from None
    except Exception as exc:
        raise SystemExit(redact(str(exc))) from None
