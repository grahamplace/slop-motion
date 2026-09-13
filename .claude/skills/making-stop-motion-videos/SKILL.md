---
name: making-stop-motion-videos
description: Use when asked to create, extend, retime, or debug a stop-motion video or animated image sequence with the `stop-motion` CLI (scene.json, compile, frame, render), or when a stop-motion compile fails, is interrupted, or refuses to resume.
---

# Making stop-motion videos

## Overview

You author a scene (JSON + prompt text files) and judge the images. The CLI
generates each pose with paid OpenAI requests, keeps provider state in a build
directory, and encodes an MP4. **The CLI does not judge continuity. You must
open the PNGs and look.**

Run from the repo root: `uv run stop-motion …`. Commands that generate images
need `OPENAI_API_KEY`; if it lives in `.env`, use
`uv run --env-file .env stop-motion …`. Dry runs, `status`/`view`/`timeline`/
`render`, and a compile with no poses left to build (e.g. retiming) need no key.
`--scene` and `--project` accept any path, relative or absolute.

## Workflow

1. **Author** in `scenes/<name>/` (copy `examples/robot-wave/` as a start;
   ignore its `steps.json` and `step-template.txt`).
2. **Dry-run** — free, no writes:
   `stop-motion compile --scene scenes/<name>/scene.json --project projects/<name> --dry-run`
   Check `total_poses`, `completed_poses`, `remaining_requests`, `duration_seconds`.
   Set `max_image_requests` a little above the pose count: failed requests
   count too.
3. **Checkpoint** — `compile … --through 2` builds the opening + first edit.
4. **Look** at every returned `frames[].path` with your image-viewing tool
   (or `view --project P --frame f0001 --frame f0002 --contact-sheet`, which
   writes a PNG under `previews/` and returns `contact_sheet_path`). Show the
   user and get approval before spending the rest.
5. **Revise** unbuilt prompts if needed, then compile again without
   `--through` (or a larger `--through`). Completed poses are reused, not re-bought.
6. **Review** the final frames and `video.path` before reporting success.

## Scene format

```json
{
  "schema_version": 1,
  "brief": "A clay cat stretches. Fixed camera, wooden tabletop.",
  "settings": { "fps": 24, "max_image_requests": 5 },
  "opening": { "prompt_file": "opening.txt", "hold": 5 },
  "edits": [
    { "prompt_file": "edits/01.txt", "hold": 5 },
    { "prompt": "Edit only the front paws a little further forward. Preserve everything else.", "hold": 6 }
  ]
}
```

- Paths resolve relative to `scene.json`. Unknown fields are errors.
- `brief` (or `brief_file`) is metadata only — it is **not** sent with prompts.
  Put every visual detail (character, set, lighting, camera) in the opening prompt.
- `opening`: exactly one of `prompt`, `prompt_file`, `image` (an existing PNG;
  costs no request). Each edit: exactly one of `prompt`, `prompt_file`.
- Each edit is applied to the previous pose. Write small deltas: "Edit only X.
  Preserve everything else."
- `hold` = output video frames for that pose (default 3).
  **Duration = sum(holds) / fps.** The example above: (5+5+6) / 24 ≈ 0.67 s.
- Settings defaults: `gpt-image-2.5-sunburst`, driver `gpt-5.5`, `medium`,
  `1024x1024`, 24 fps, `max_image_requests` 20. Requests = poses (minus 1 if the
  opening is an `image`). The whole scene must fit the cap, even with `--through`.

## Output contract

- Success: one JSON object on stdout; artifact paths are absolute.
- Progress: JSON Lines on stderr. Errors: `{"error": "..."}` on stderr, exit ≠ 0.
- `--through N` counts poses **including** the opening, not requests.

## What you can change after building

| Change | Same `--project`? | Costs requests? |
|---|---|---|
| Holds, fps (retiming) | Yes | No — new video only |
| Unbuilt edit prompts, appending edits | Yes | Only new poses |
| Raising `max_image_requests` | Yes | No |
| A **completed** prompt, the opening image, model/quality/size | **No** — new project dir | Rebuilds from scratch |

If edit prompts repeat a detail you're changing (e.g. the character's color),
update them too.

To carry approved work into a new project, import the **last good pose** as
the opening: `"opening": {"image": "<abs path to projects/old/frames/f0003.png>"}`
and drop the edits already built before it. Only the opening can be an image;
every edit after it is regenerated.

## Failures

A failed or interrupted request is **never** retried automatically and blocks
that project (`Attempt … is failed; refusing to replay it`,
`Uncommitted image preserved`). Every attempt counts against the cap.

1. `stop-motion status --project projects/<name>` — read `attempts`,
   `incomplete_attempts`, `frames`.
2. Tell the user what was spent; get approval before spending more.
3. Fix the cause (key, network, prompt) and compile into a **new** project
   directory, importing the last good pose as the opening (see above). An
   uncommitted PNG can be imported the same way if it looks right.
4. Never edit or delete `project.json`, frames, or attempts to force a retry.

## Common mistakes

- Reporting success because the command exited 0 — an MP4 existing is not
  proof the motion looks right. Look at the images.
- Expecting the brief to style the images; it isn't sent.
- Big changes per edit (identity drifts). Use more, smaller edits.
- Running the full compile without a dry run and checkpoint.
- `frame --base …` on a compiled project — rejected ("This is a compiled
  project"). Revise the scene and compile, or branch in a separate `init` project.
- Hand-editing anything under `projects/`. The CLI owns it.

## Lower-level commands

For manual per-frame work in a non-compiled project: `init PATH --brief-file`,
`frame --project P --prompt-file F [--base fNNNN]`, `view`, `timeline --file`
(`[{"frame_id": "f0001", "hold": 3}]`), `render`. Use returned frame IDs; don't
guess them. See `README.md` for details.
