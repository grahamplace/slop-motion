"""Repeat butterfly, then conditionally test robot with the same uploaded-PNG bootstrap.

Run: .venv/bin/python -m scripts.reproduce_uploaded_opening butterfly|robot
Each arm is capped at six new edits, with no retries. --prepare/--report are offline.
The robot live run requires a hash-bound passing visual review of the butterfly.
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageStat

from scripts.compare_edit_settings import REGIONS as BUTTERFLY_REGIONS
from scripts.compare_edit_settings import load_key, redact
from scripts.compare_models import REGIONS as ROBOT_REGIONS
from scripts.compare_models import save_png
from scripts.compare_responses import run_chain, sha256, validate_attempts
from stop_motion.storage import HarnessError, atomic_write, project_lock, read_json, write_json
from stop_motion.video import export_video

REPO = Path(__file__).resolve().parents[1]
BUTTERFLY = REPO / "projects/butterfly-responses-test"
ROBOT = REPO / "projects/robot-wave-model-comparison"
OUTPUT = REPO / "projects/uploaded-opening-reproduction"
EDIT_COUNT = 6


def prepare(subject):
    butterfly = read_json(BUTTERFLY / "plan.json")
    if subject == "butterfly":
        source, original, offset = BUTTERFLY, butterfly, 1
        requests = original["baseline_requests"]
        seed_hash = original["baseline_sha256"][0]
        regions = BUTTERFLY_REGIONS
    elif subject == "robot":
        source = ROBOT / "sunburst"
        original = read_json(ROBOT / "plan.json")["conditions"]["sunburst"]
        offset = 0
        requests = original["baseline_requests"][1:]
        seed_hash = read_json(source / "attempts.json")[0]["output_sha256"]
        regions = ROBOT_REGIONS
    else:
        raise HarnessError("Unknown subject.")
    prior = read_json(source / "attempts.json")
    if len(prior) != 7 - offset or len(requests) != EDIT_COUNT:
        raise HarnessError("Expected a completed six-edit baseline.")
    validate_attempts(prior, output=source, request_count=7 - offset, frame_offset=offset)
    if sha256(source / "00.png") != seed_hash:
        raise HarnessError("Baseline opening changed.")
    if original["image_tool"] != butterfly["image_tool"]:
        raise HarnessError("Baseline image settings differ from the successful butterfly.")
    if original["driver_model"] != butterfly["driver_model"]:
        raise HarnessError("Baseline driver differs from the successful butterfly.")
    plan = {
        "subject": subject,
        "driver_model": butterfly["driver_model"],
        "instructions": butterfly["instructions"],
        "image_tool": butterfly["image_tool"],
        "maximum_response_requests": EDIT_COUNT,
        "generates_opening": False,
        "context_mode": "response_chain",
        "baseline_requests": requests,
        "baseline": str(source),
        "baseline_sha256": [sha256(source / f"{i:02d}.png") for i in range(7)],
        "seed_sha256": seed_hash,
        "regions": regions,
        "note": (
            "Opening PNG uploaded on first edit, then new previous_response_id chain. "
            "Butterfly repeats the saved request settings and user prompts exactly. "
            "Robot keeps its saved opening and edit prompts, but adopts the butterfly's "
            "uploaded-opening/edit-only initialization. Prompt rewriting and sampling remain."
        ),
    }
    plan = json.loads(json.dumps(plan))
    directory = OUTPUT / subject
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise HarnessError("Saved plan differs; refusing to mix experiments.")
    else:
        write_json(path, plan)
    seed = directory / "00.png"
    if not seed.exists():
        atomic_write(seed, (source / "00.png").read_bytes())
    if sha256(seed) != seed_hash:
        raise HarnessError("Copied seed changed.")
    if not (directory / "attempts.json").exists():
        write_json(directory / "attempts.json", [])
    (directory / "exports").mkdir(exist_ok=True)
    return plan


def require_control_pass():
    directory = OUTPUT / "butterfly"
    attempts = read_json(directory / "attempts.json")
    if len(attempts) != EDIT_COUNT:
        raise HarnessError("Complete and visually review the butterfly before running robot.")
    validate_attempts(attempts, output=directory, request_count=EDIT_COUNT)
    review = read_json(directory / "review.json")
    hashes = [sha256(directory / f"{i:02d}.png") for i in range(7)]
    if review.get("verdict") != "pass" or review.get("frame_sha256") != hashes:
        raise HarnessError(
            "Robot requires a passing visual review of these exact butterfly frames."
        )


def report(plan, *, render=False):
    directory = OUTPUT / plan["subject"]
    variants = {"Original run": Path(plan["baseline"]), "Uploaded-opening rerun": directory}
    metrics = {}
    sheet = Image.new("RGB", (1792, 584), "#ededed")
    draw = ImageDraw.Draw(sheet)
    with Image.open(directory / "00.png") as source:
        seed = source.convert("RGB")
    for row, (label, root) in enumerate(variants.items()):
        metrics[label] = []
        for i in range(7):
            path = root / f"{i:02d}.png"
            if not path.exists():
                continue
            with Image.open(path) as source:
                current = source.convert("RGB")
            diff = {}
            for name, bounds in plan["regions"].items():
                delta = ImageChops.difference(seed.crop(bounds), current.crop(bounds))
                diff[name] = round(sum(ImageStat.Stat(delta).mean) / 3, 4)
            metrics[label].append({"frame": i, "mean_absolute_rgb_difference": diff})
            draw.text((i * 256 + 6, row * 292 + 5), f"{label} | {i}", fill="#111", font_size=14)
            sheet.paste(
                current.resize((256, 256), Image.Resampling.LANCZOS), (i * 256, row * 292 + 30)
            )
    save_png(directory / "comparison.png", sheet)
    write_json(
        directory / "metrics.json",
        {
            "regions": plan["regions"],
            "conditions": metrics,
            "note": "Static crop RGB differences, not an overall visual quality score.",
        },
        replace=True,
    )
    attempts = read_json(directory / "attempts.json")
    if (
        not render
        or len(attempts) != EDIT_COUNT
        or any(a["status"] != "succeeded" for a in attempts)
    ):
        return
    validate_attempts(attempts, output=directory, request_count=EDIT_COUNT)
    export(directory, "1024x1024", 3)
    review = directory / "slow-review"
    review.mkdir(exist_ok=True)
    (review / "exports").mkdir(exist_ok=True)
    for i in range(7):
        picture = Image.new("RGB", (1024, 576), "#ededed")
        draw = ImageDraw.Draw(picture)
        for col, (label, root) in enumerate(variants.items()):
            with Image.open(root / f"{i:02d}.png") as source:
                picture.paste(
                    source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS),
                    (col * 512, 64),
                )
            draw.text((col * 512 + 12, 7), label, fill="#111", font_size=22)
            draw.text(
                (col * 512 + 12, 35),
                f"{plan['subject']} | frame {i} | 1 pose/sec",
                fill="#333",
                font_size=16,
            )
        save_png(review / f"{i:02d}.png", picture)
    export(review, "1024x576", 24)


def export(directory, size, hold):
    if not (directory / "exports/export-0001.mp4").exists():
        export_video(
            directory,
            {
                "schema_version": 1,
                "settings": {"size": size, "fps": 24},
                "timeline": [{"frame_id": f"{i:02d}", "hold": hold} for i in range(7)],
                "note": "Seven poses; review export holds poses longer, without extra generation.",
            },
            {f"{i:02d}": directory / f"{i:02d}.png" for i in range(7)},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", choices=("butterfly", "robot"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with project_lock(OUTPUT):
        plan = prepare(args.subject)
        if not args.prepare and not args.report:
            if args.subject == "robot":
                require_control_pass()
            load_key()
            try:
                with OpenAI(max_retries=0, timeout=240.0) as client:
                    run_chain(client, plan, output=OUTPUT / args.subject)
            finally:
                report(plan)
        report(plan, render=not args.prepare)
        print(json.dumps({"output": str(OUTPUT / args.subject), "maximum_new_calls": EDIT_COUNT}))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped; incomplete requests will not be replayed.") from None
    except Exception as exc:
        raise SystemExit(redact(str(exc))) from None
