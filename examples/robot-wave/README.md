# Robot wave

For the current CLI, use [scene.json](scene.json) and its `edits/*.txt` prompts:

```sh
uv run stop-motion compile --scene examples/robot-wave/scene.json --project projects/robot-cli --dry-run
uv run --env-file .env stop-motion compile --scene examples/robot-wave/scene.json --project projects/robot-cli
```

This generates an opening separately, uploads its PNG into a fresh edit
conversation, then chains six edits: at most seven requests, 1.75 seconds of
video. Add `--through 2` to inspect an opening/first-edit checkpoint before
continuing. See the [main README](../../README.md) for authoring and resume rules.

## Historical model comparison (not the recommended pipeline)

The experiment below used a different initialization and produced poor robot
continuity. It is preserved for comparison, not used by `compile`.

A full-body clay robot raises the arm on the viewer's right and makes a small
wave. Its face, three chest buttons, badge text, other limbs, stacked blocks,
and scenery must remain fixed. The six target arm angles are absolute, measured
from vertically downward toward the viewer's right: 30, 60, 90, 120, 100, 120 degrees.

Files: `brief.txt`, `opening.txt`, `step-template.txt`, and `steps.json`.

From the repository root:

```sh
.venv/bin/python -m scripts.compare_models --prepare
.venv/bin/python -m scripts.compare_models
```

The live command reads the existing local API-key setup, then runs GPT Image 2.5
Sunburst and GPT Image 2 through the same GPT-5.5 Responses workflow. Each model
generates its own opening, then six edits linked with `previous_response_id`.
Settings are medium quality, 1024x1024, opaque PNG, one image call per response,
and no retries. The cap is 14 calls total. Completed runs do not generate again;
incomplete or failed requests are never automatically replayed.

Outputs go to `projects/robot-wave-model-comparison/`. Each normal video contains
seven poses at eight poses per second (three-frame holds at 24 fps), lasting
0.875 seconds. A separately labeled seven-second side-by-side review holds each
pose for one second; it does not contain additional generated motion.

Assess pose following, movement of supposedly fixed parts, character details,
texture changes, and opening-frame quality separately. This is one run per model
with different generated openings and possibly different rewritten prompts, not
a controlled same-image editing benchmark or a general percentage-quality score.
Both use the same quality label, not necessarily the same compute or cost.
