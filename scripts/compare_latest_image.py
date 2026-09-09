"""Robot-wave test with only the latest image ID, not a Responses history chain.

Run: .venv/bin/python -m scripts.compare_latest_image [--prepare | --report]
Reuses the original Sunburst opening. Maximum six new image calls; no retries.
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageStat

from scripts.compare_edit_settings import load_key, redact
from scripts.compare_models import REGIONS, save_png
from scripts.compare_responses import run_chain, sha256, validate_attempts
from stop_motion.storage import HarnessError, atomic_write, project_lock, read_json, write_json
from stop_motion.video import export_video

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "projects/robot-wave-model-comparison"
OUTPUT = REPO / "projects/robot-wave-latest-image-test"
EDIT_COUNT = 6


def prepare():
    source_plan = read_json(SOURCE / "plan.json")
    original = source_plan["conditions"]["sunburst"]
    source_frames = SOURCE / "sunburst"
    attempts = read_json(source_frames / "attempts.json")
    if len(attempts) != 7 or len(original["baseline_requests"]) != 7:
        raise HarnessError("Expected the completed seven-frame Sunburst robot baseline.")
    validate_attempts(attempts, output=source_frames, request_count=7, frame_offset=0)
    if original["image_tool"]["model"] != "gpt-image-2.5-sunburst":
        raise HarnessError("Baseline is not Sunburst; refusing a model substitution.")
    plan = {
        **original,
        "generates_opening": False,
        "context_mode": "latest_image_id",
        "maximum_response_requests": EDIT_COUNT,
        "baseline_requests": original["baseline_requests"][1:],
        "seed_image_id": attempts[0]["image_calls"][0]["id"],
        "seed_sha256": sha256(source_frames / "00.png"),
        "baseline_sha256": [sha256(source_frames / f"{i:02d}.png") for i in range(7)],
        "baseline": str(source_frames),
        "steps": source_plan["steps"],
        "limitations": (
            "One exploratory chain versus the existing saved chain, not a replicated trial. "
            "Same opening, user prompts, driver instructions, and model/settings. Each request "
            "references only the immediately preceding image-generation item; no conversation "
            "or previous_response_id. Prompt rewriting remains enabled and may differ. "
            "No claim about undocumented server-side image conditioning or exact pixel copying."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise HarnessError("Saved plan differs; refusing to mix experiments.")
    else:
        write_json(path, plan)
    seed = OUTPUT / "00.png"
    if not seed.exists():
        atomic_write(seed, (source_frames / "00.png").read_bytes())
    if sha256(seed) != plan["seed_sha256"]:
        raise HarnessError("Saved seed changed; refusing to continue.")
    if not (OUTPUT / "attempts.json").exists():
        write_json(OUTPUT / "attempts.json", [])
    (OUTPUT / "exports").mkdir(exist_ok=True)
    return plan


def report(plan):
    directories = {"History chain (original)": Path(plan["baseline"]), "Latest image only": OUTPUT}
    metrics = {}
    sheet = Image.new("RGB", (1024, 1632), "#ededed")
    sheet_draw = ImageDraw.Draw(sheet)
    with Image.open(OUTPUT / "00.png") as source:
        opening = source.convert("RGB")
    for column, (label, directory) in enumerate(directories.items()):
        metrics[label] = []
        for frame in range(7):
            path = directory / f"{frame:02d}.png"
            if not path.exists():
                continue
            with Image.open(path) as source:
                current = source.convert("RGB")
            differences = {}
            for region, bounds in REGIONS.items():
                diff = ImageChops.difference(opening.crop(bounds), current.crop(bounds))
                differences[region] = round(sum(ImageStat.Stat(diff).mean) / 3, 4)
            metrics[label].append({"frame": frame, "mean_absolute_rgb_difference": differences})
            if frame in (0, 1, 6):
                row = (0, 1, 6).index(frame)
                y = row * 544
                sheet_draw.text(
                    (column * 512 + 10, y + 4),
                    f"{label} | frame {frame}",
                    fill="#111111",
                    font_size=18,
                )
                sheet.paste(
                    current.resize((512, 512), Image.Resampling.LANCZOS), (column * 512, y + 32)
                )
    save_png(OUTPUT / "comparison.png", sheet)
    write_json(
        OUTPUT / "metrics.json",
        {
            "regions": REGIONS,
            "conditions": metrics,
            "note": "Static crop RGB difference from the shared opening; not overall quality.",
        },
        replace=True,
    )


def render(plan):
    attempts = read_json(OUTPUT / "attempts.json")
    if len(attempts) != EDIT_COUNT or any(a["status"] != "succeeded" for a in attempts):
        return
    validate_attempts(attempts, output=OUTPUT, request_count=EDIT_COUNT, plan=plan)
    if not (OUTPUT / "exports/export-0001.mp4").exists():
        export_video(
            OUTPUT,
            {
                "schema_version": 1,
                "settings": {"size": "1024x1024", "fps": 24},
                "timeline": [{"frame_id": f"{i:02d}", "hold": 3} for i in range(7)],
                "experiment": plan,
            },
            {f"{i:02d}": OUTPUT / f"{i:02d}.png" for i in range(7)},
        )
    review = OUTPUT / "slow-review"
    review.mkdir(exist_ok=True)
    (review / "exports").mkdir(exist_ok=True)
    if (review / "exports/export-0001.mp4").exists():
        return
    for frame in range(7):
        picture = Image.new("RGB", (1024, 576), "#ededed")
        draw = ImageDraw.Draw(picture)
        for column, (label, directory) in enumerate(
            (("History chain (original)", Path(plan["baseline"])), ("Latest image only", OUTPUT))
        ):
            with Image.open(directory / f"{frame:02d}.png") as source:
                picture.paste(
                    source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS),
                    (column * 512, 64),
                )
            draw.text((column * 512 + 12, 7), label, fill="#111111", font_size=22)
            angle = plan["steps"][frame]["arm_angle_degrees"]
            draw.text(
                (column * 512 + 12, 35),
                f"Frame {frame} | target {angle} deg | 1 pose/sec",
                fill="#333333",
                font_size=16,
            )
        save_png(review / f"{frame:02d}.png", picture)
    export_video(
        review,
        {
            "schema_version": 1,
            "settings": {"size": "1024x576", "fps": 24},
            "timeline": [{"frame_id": f"{i:02d}", "hold": 24} for i in range(7)],
            "note": "Slowed comparison of seven poses, not seven seconds of new motion.",
        },
        {f"{i:02d}": review / f"{i:02d}.png" for i in range(7)},
    )


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
                    run_chain(client, plan, output=OUTPUT)
            finally:
                report(plan)
        report(plan)
        if not args.prepare:
            render(plan)
        print(
            json.dumps({"output": str(OUTPUT), "maximum_new_image_calls": EDIT_COUNT}), flush=True
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped; incomplete requests will not be replayed.") from None
    except Exception as exc:
        raise SystemExit(redact(str(exc))) from None
