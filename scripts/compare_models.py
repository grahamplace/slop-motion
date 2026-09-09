"""Compare two image models on the robot-wave Responses video pipeline.

Run: .venv/bin/python -m scripts.compare_models [--prepare | --report]
One opening plus six edits per model. Maximum 14 image calls; no retries.
"""

import argparse
import io
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

from scripts.compare_edit_settings import load_key, redact
from scripts.compare_responses import run_chain, validate_attempts
from stop_motion.storage import HarnessError, atomic_write, project_lock, read_json, write_json
from stop_motion.video import export_video

REPO = Path(__file__).resolve().parents[1]
BRIEF = REPO / "examples/robot-wave"
OUTPUT = REPO / "projects/robot-wave-model-comparison"
MODELS = {"sunburst": "gpt-image-2.5-sunburst", "image-2": "gpt-image-2"}
LABELS = {"sunburst": "GPT Image 2.5 Sunburst", "image-2": "GPT Image 2"}
REGIONS = {"background": (24, 24, 200, 200), "tabletop": (880, 880, 1008, 1008)}
INSTRUCTIONS = (
    "For each user request, call the image_generation tool exactly once. Generate the "
    "opening image on the first turn; on subsequent turns edit the latest image in the "
    "conversation. Return the image without commentary."
)


def prepare():
    steps = read_json(BRIEF / "steps.json")
    if len(steps) != 6:
        raise HarnessError("This comparison authorizes exactly six edits per model.")
    opening = (BRIEF / "opening.txt").read_text(encoding="utf-8")
    template = (BRIEF / "step-template.txt").read_text(encoding="utf-8")
    prompts = [opening, *(template.format(**step) for step in steps)]
    shared = {
        "driver_model": "gpt-5.5",
        "instructions": INSTRUCTIONS,
        "maximum_response_requests": 7,
        "generates_opening": True,
        "baseline_requests": [{"prompt": prompt} for prompt in prompts],
    }
    plan = {
        "brief": (BRIEF / "brief.txt").read_text(encoding="utf-8"),
        "steps": [{"pose": "opening", "arm_angle_degrees": 0}, *steps],
        "maximum_image_calls": 14,
        "static_regions": REGIONS,
        "conditions": {
            name: {
                **shared,
                "image_tool": {
                    "type": "image_generation",
                    "action": "edit",
                    "model": model,
                    "quality": "medium",
                    "size": "1024x1024",
                    "background": "opaque",
                    "output_format": "png",
                },
            }
            for name, model in MODELS.items()
        },
        "limitations": (
            "One complete pipeline run per model, each with its own generated opening. "
            "Same user prompts and text orchestrator, but rewritten prompts and starting "
            "images can differ. Equal quality labels are not equal compute or dollar cost. "
            "This is exploratory, not a general model benchmark."
        ),
    }
    # Normalize tuples to their persisted JSON representation before comparison.
    plan = json.loads(json.dumps(plan))
    path = OUTPUT / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise HarnessError("Saved plan differs; refusing to mix experiment conditions.")
    else:
        write_json(path, plan)
    for name in MODELS:
        directory = OUTPUT / name
        directory.mkdir(exist_ok=True)
        (directory / "exports").mkdir(exist_ok=True)
        record = directory / "attempts.json"
        if not record.exists():
            write_json(record, [])
    return plan


def save_png(path, picture):
    content = io.BytesIO()
    picture.save(content, format="PNG")
    atomic_write(path, content.getvalue(), replace=True)


def report(plan):
    metrics = {}
    sheet = Image.new("RGB", (1120, 620), "#ededed")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(MODELS):
        directory = OUTPUT / name
        steps = []
        if (directory / "00.png").exists():
            with Image.open(directory / "00.png") as source:
                original = source.convert("RGB")
            for path in sorted(directory.glob("[0-9][0-9].png")):
                with Image.open(path) as source:
                    current = source.convert("RGB")
                regions = {}
                for region, bounds in REGIONS.items():
                    diff = ImageChops.difference(original.crop(bounds), current.crop(bounds))
                    regions[region] = round(sum(ImageStat.Stat(diff).mean) / 3, 4)
                steps.append({"frame": int(path.stem), "mean_absolute_rgb_difference": regions})
        metrics[name] = steps
        for column, frame in enumerate((0, 2, 4, 6)):
            path = directory / f"{frame:02d}.png"
            x, y = column * 280, row * 310
            if path.exists():
                with Image.open(path) as source:
                    sheet.paste(
                        ImageOps.contain(source.convert("RGB"), (256, 256)), (x + 12, y + 8)
                    )
            angle = plan["steps"][frame]["arm_angle_degrees"]
            suffix = f"Frame {frame} | target {angle} degrees" if path.exists() else "Not generated"
            draw.multiline_text(
                (x + 12, y + 269), f"{LABELS[name]}\n{suffix}", fill="#111111", font_size=13
            )
    save_png(OUTPUT / "comparison.png", sheet)
    write_json(
        OUTPUT / "metrics.json",
        {
            "regions": REGIONS,
            "conditions": metrics,
            "note": "Difference from each model's own opening; not an overall quality score.",
        },
        replace=True,
    )


def render_arm(name, plan):
    directory = OUTPUT / name
    attempts = read_json(directory / "attempts.json")
    if len(attempts) != 7 or any(attempt["status"] != "succeeded" for attempt in attempts):
        return False
    validate_attempts(attempts, output=directory, request_count=7, frame_offset=0)
    if not (directory / "exports/export-0001.mp4").exists():
        result = export_video(
            directory,
            {
                "schema_version": 1,
                "settings": {"size": "1024x1024", "fps": 24},
                "timeline": [{"frame_id": f"{frame:02d}", "hold": 3} for frame in range(7)],
                "condition": plan["conditions"][name],
            },
            {f"{frame:02d}": directory / f"{frame:02d}.png" for frame in range(7)},
        )
        print(f"{LABELS[name]} video: {result['path']}", flush=True)
    return True


def render_review(plan):
    complete = [render_arm(name, plan) for name in MODELS]
    if not all(complete):
        return
    directory = OUTPUT / "slow-review"
    directory.mkdir(exist_ok=True)
    (directory / "exports").mkdir(exist_ok=True)
    if (directory / "exports/export-0001.mp4").exists():
        return
    for frame in range(7):
        picture = Image.new("RGB", (1024, 576), "#ededed")
        draw = ImageDraw.Draw(picture)
        angle = plan["steps"][frame]["arm_angle_degrees"]
        for column, name in enumerate(MODELS):
            with Image.open(OUTPUT / name / f"{frame:02d}.png") as source:
                picture.paste(
                    source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS),
                    (column * 512, 64),
                )
            draw.text((column * 512 + 14, 7), LABELS[name], fill="#111111", font_size=22)
            draw.text(
                (column * 512 + 14, 35),
                f"Frame {frame} | target {angle} deg | review: 1 pose/sec",
                fill="#333333",
                font_size=16,
            )
        save_png(directory / f"{frame:02d}.png", picture)
    result = export_video(
        directory,
        {
            "schema_version": 1,
            "settings": {"size": "1024x576", "fps": 24},
            "timeline": [{"frame_id": f"{frame:02d}", "hold": 24} for frame in range(7)],
            "note": "Slowed review: same seven poses, not seven seconds of new animation.",
        },
        {f"{frame:02d}": directory / f"{frame:02d}.png" for frame in range(7)},
    )
    print(f"Side-by-side slow review: {result['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true", help="Prepare without API calls")
    modes.add_argument("--report", action="store_true", help="Report/render without API calls")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with project_lock(OUTPUT):
        plan = prepare()
        if not args.prepare and not args.report:
            load_key()
            try:
                with OpenAI(max_retries=0, timeout=240.0) as client:
                    for name, condition in plan["conditions"].items():
                        run_chain(client, condition, output=OUTPUT / name)
                        report(plan)
                        render_arm(name, plan)
            finally:
                report(plan)
        report(plan)
        if not args.prepare:
            render_review(plan)
        print(
            json.dumps({"output": str(OUTPUT), "models": MODELS, "maximum_image_calls": 14}),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped; incomplete requests will not be replayed.") from None
    except Exception as exc:
        raise SystemExit(redact(str(exc))) from None
