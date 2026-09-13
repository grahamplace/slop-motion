# Stop-motion

A small Python CLI for an existing, vision-capable AI agent. The agent writes
scene/config files and judges the images; the CLI generates frames, saves
provider state, and compiles an MP4. There is no separate agent planner to run.

## Setup

Requires Python 3.11+, macOS or Linux, and `ffmpeg`/`ffprobe` with H.264 support
on `PATH`.

```sh
uv sync
uv run stop-motion --help
```

Set `OPENAI_API_KEY` (the default provider) or `GEMINI_API_KEY` (Gemini) in the
environment. If you keep it in the ignored repo `.env`, use `uv run --env-file .env stop-motion …` for commands that generate
images. The CLI does not automatically search for credential files and never
stores credentials in a scene or project. Offline commands need no API key.
Alternatively, `pip install -e .` in a virtual environment.

## Agent workflow: scene files → video

Start from [the robot scene](examples/robot-wave/scene.json). Copy its directory
to a new authoring directory, then edit `scene.json`, `opening.txt`, and
`edits/*.txt`. The existing `step-template.txt` and `steps.json` belong to the
older experiments; compile reads only files referenced in `scene.json`.

Validate first; this prints the resolved settings, duration, completed poses, and
remaining request count without creating files or contacting an image provider:

```sh
uv run stop-motion compile --scene examples/robot-wave/scene.json --project projects/robot-cli --dry-run
```

Compile the example (seven paid Responses requests: one opening plus six edits):

```sh
uv run --env-file .env stop-motion compile --scene examples/robot-wave/scene.json --project projects/robot-cli
```

The result is one JSON object on stdout with frame paths and `video.path`.
Progress is JSON Lines on stderr. Errors are JSON on stderr with a nonzero exit
code. All final artifact paths are absolute. This example is seven poses held
for six output frames each: **1.75 seconds at 24 FPS**, not seven seconds.

For a visual checkpoint, add `--through 2` to build just the opening and first
edit, then inspect the returned PNGs with the agent host's image-viewing tool.
Re-run without `--through` to finish. `--through N` counts total poses including
the opening, not new requests. It also renders the selected prefix for review.
You can edit unbuilt prompts between checkpoints.

An example instruction for the agent:

> Make a stop-motion scene of a clay robot waving. Copy the robot scene into a
> new scene directory, edit the opening and small per-pose changes, and set the
> timing and request cap. Dry-run the CLI, compile through the first two poses,
> inspect the actual images, then compile the rest and review the video. Keep
> provider state in the build directory; don't hand-edit project.json or frames.

The CLI does not judge continuity automatically. A successful API call or valid
MP4 is not proof that the movement looks good.

## Scene format

```json
{
  "schema_version": 1,
  "brief": "A clay robot waves. Fixed camera and tabletop.",
  "settings": {
    "model": "gpt-image-2.5-sunburst",
    "driver_model": "gpt-5.5",
    "quality": "medium",
    "size": "1024x1024",
    "fps": 24,
    "max_image_requests": 7
  },
  "opening": { "prompt_file": "opening.txt", "hold": 6 },
  "edits": [
    { "prompt_file": "edits/01.txt", "hold": 6 },
    { "prompt": "Edit only the waving arm a little higher. Preserve everything else.", "hold": 6 }
  ]
}
```

- `brief` or `brief_file`: exactly one. This is planning metadata, not an
  automatically appended generation prompt.
- `opening`: exactly one of `prompt`, `prompt_file`, or `image`. To reuse
  an approved PNG, use `{"image": "approved-opening.png", "hold": 6}`; it is
  copied byte-for-byte and costs no opening-generation request.
- `edits`: ordered small changes, each with exactly one of `prompt` or
  `prompt_file`. Each edit uses the preceding pose, never an unrelated opening.
  Prompt files are sent as written; the Responses driver may revise them.
- All file paths resolve relative to `scene.json`, not the shell's directory.
  No template expansion is performed.
- `hold`: positive integer output frames, default 3. Duration is
  `sum(hold) / fps`. More holds slow playback; more edits add distinct poses.
- With the default OpenAI provider, settings default to Sunburst, GPT-5.5, medium
  quality, 1024×1024, 24 FPS, and 20 requests. Its `backend` defaults to `responses`,
  which is required for OpenAI scene compilation.
  `gpt-image-2.5-flare` is also accepted but wasn't used in the successful tests.
- Unknown fields and invalid inputs are errors. The entire scene, including all
  referenced prompts and the optional opening PNG, is checked before spending.
  The full scene must fit the explicit request cap, even for a staged compile.

## Provider configuration

Image settings accept an explicit `provider`, defaulting to `openai`, and a
`provider_options` object. The available providers are `openai` and `gemini`. For example:

```json
"settings": {
  "provider": "openai",
  "model": "gpt-image-2.5-sunburst",
  "size": "1024x1024",
  "provider_options": {
    "backend": "responses",
    "quality": "medium",
    "driver_model": "gpt-5.5"
  },
  "fps": 24,
  "max_image_requests": 7
}
```

Existing top-level `backend`, `quality`, and `driver_model` settings remain
accepted as OpenAI aliases. Conflicting nested and top-level values are errors.
New project manifests and command results contain the resolved options under
`provider_options`. Defaults are resolved only after choosing a provider.

`init` accepts `--provider openai` and `--provider-options options.json` (a JSON
object of provider options), alongside the existing OpenAI flags. Credentials
remain environment-only; credential fields and unknown options are rejected.
Each new attempt records its provider, generation settings, workflow version, and
optional continuation state. Provider diagnostics live under `provider_metadata`
so they cannot overwrite the attempt ledger. Historical top-level response fields
remain readable and existing attempts are preserved.

Reading an old project does not migrate it: missing provider means OpenAI, and
missing backend retains legacy Images behavior. Equivalent legacy/nested scene
settings can resume the same build. Changing effective generation options needs
a new project; retiming and explicit request-cap changes remain supported.

## Gemini image generation

Select `provider: "gemini"` in a scene or pass `--provider gemini` to `init`.
The initial integration supports `gemini-3.1-flash-image`, which is also its
default model. Gemini has no additional `provider_options` yet. OpenAI-only
settings such as `quality`, `backend`, and `driver_model` are errors for Gemini;
remove them when adapting an OpenAI scene.

The [Gemini robot scene](examples/robot-wave/scene-gemini.json) reuses the existing
opening and six edit prompts. Validate it without a key or network access:

```sh
uv run stop-motion compile --scene examples/robot-wave/scene-gemini.json --project projects/robot-gemini --dry-run
```

With `GEMINI_API_KEY` set (or present in your explicitly loaded `.env`), compile
a checkpoint and inspect its PNGs before completing the sequence:

```sh
uv run --env-file .env stop-motion compile --scene examples/robot-wave/scene-gemini.json --project projects/robot-gemini --through 2
uv run --env-file .env stop-motion compile --scene examples/robot-wave/scene-gemini.json --project projects/robot-gemini
```

The first command above makes two paid generation requests; completing the
example makes five more. Imported openings cost no generation request. The
request cap covers attempted submissions, including failures, and does not
estimate dollars. Usage is saved when returned; missing usage remains unknown.
No Google SDK is required: the adapter uses the Interactions HTTP interface.

The opening is standalone. The first edit uploads the approved opening PNG in a
fresh interaction; later edits continue the selected parent's saved
`interaction_id`. Manual `frame --base` can branch from an earlier edit; selecting
an opening starts a fresh edit chain. Extra `--reference` inputs are unsupported
in this initial workflow. Image instructions and output settings are sent on
every turn. The CLI does not replay failed/unknown attempts or automatically
restart expired/rejected chains. Stored interactions must remain available in the
same API project; Google documents the retention limits in its
[Interactions guide](https://ai.google.dev/gemini-api/docs/interactions-overview#data_storage_and_retention).

Supported 1K canvases are `1024x1024`, `848x1264`, `1264x848`, `896x1200`,
`1200x896`, `928x1152`, `1152x928`, `768x1376`, `1376x768`, and `1584x672`.
Doubling or quadrupling both dimensions selects the corresponding 2K or 4K size.
These map explicitly to Google's documented aspect ratios and resolutions;
arbitrary OpenAI sizes are not translated approximately. The compiler rejects a
wrong-sized result before requesting another pose and keeps the PNG for inspection.
See Google's [size table](https://ai.google.dev/gemini-api/docs/image-generation#aspect_ratios_and_image_size).

The adapter requests JPEG output using the endpoint's default delivery mode
and converts decoded pixels to PNG without resizing. An explicit delivery selector
is omitted because the live endpoint rejects it. The adapter also accepts valid PNG
responses. Only final model-output images are counted; intermediate thought images
are excluded, and a missing or multiple final image result stops the build.
Provider diagnostics retain the original MIME type and image hash, interaction ID,
returned model, and usage. Credentials and inline image payloads are never saved
in the manifest.

An HTTP 429 response can indicate exhausted or unavailable quota. Check quota and
[billing](https://ai.google.dev/gemini-api/docs/billing) for the Google API project
that owns `GEMINI_API_KEY`; a zero request limit requires an account change before
generation can proceed. Failed requests are recorded and never retried automatically.

The integration is covered by mocked HTTP, real image decoding/video encoding,
and fresh-process resume tests. The example is a continuity evaluation scene;
a live seven-pose Gemini sequence has not yet been visually validated. Passing
these tests does not establish motion quality or character consistency.

## The OpenAI continuity workflow

OpenAI projects using Responses use the setup that reproduced consistent
six-edit butterfly and robot sequences:

1. Generate an opening in a standalone Responses call, or import a local PNG.
2. Start a **fresh** Responses conversation for the first edit: upload that PNG
   as `input_image` with `detail: "high"`, alongside the first edit prompt.
   **Do not link the opening-generation response.**
3. Use the tested edit-only instructions, `action: "edit"`, forced image tool
   selection, one tool call maximum, and no parallel tool calls.
4. Link later edits with the selected preceding edit's `previous_response_id`.
   Do not upload every earlier PNG or switch to latest-image-ID-only inputs.

The driver is GPT-5.5 with low reasoning and a 4096-token output limit. The
image tool explicitly requests `gpt-image-2.5-sunburst`, medium quality, opaque
PNG, and the configured size. Stored Responses are required for later chain
references. The chain uses server-side conversation history, not necessarily
only one prior image internally. Image IDs and response IDs must remain
available to the same API project; if a reference expires or is rejected, the
CLI stops without switching workflows.

This preserves the tested initialization, not a guarantee of pixel identity or
long-sequence stability. The PNG initialization and edit-only instructions were
tested together, not isolated as separate causes.
[OpenAI's image tool documentation](https://developers.openai.com/api/docs/guides/tools-image-generation)
describes image inputs, edit mode, response chaining, and prompt revision.

## Resume, revisions, and safety

Keep authoring files separate from generated state:

```text
scenes/my-robot/           # agent edits these
  scene.json
  opening.txt
  edits/01.txt
projects/my-robot/         # CLI owns these
  project.json            # resolved prompts, settings, attempts, response IDs
  frames/f0001.png         # immutable PNGs with SHA-256 hashes
  previews/
  exports/export-0001.mp4
  exports/export-0001.json
```

Re-running the same compile reuses completed frames and the unchanged video
without an API key or additional image requests. Cleanly completed poses resume
across CLI processes. The compiler holds one project lock for the entire run.

You may change **unbuilt prompts**, append edits, adjust holds/FPS, and explicitly
raise the request cap. Completed prompts, the opening image, and generation
settings are checked against the saved build. If those change, choose a **new
project directory**; old frames and exports stay intact. Retiming only makes a
new video. To extend a scene beyond its cap, add edits and raise
`max_image_requests` in `scene.json`.

There are no automatic SDK retries. Every attempted request, including failures
and interruptions, consumes the cap. A failed/unknown attempt or an uncommitted
PNG blocks automatic compilation; inspect `status` and the saved files before
explicitly starting a new build. Never delete attempts to make a retry look free.
The cap limits request count, not dollars: Responses also uses the text driver
and conversation context. Usage is recorded when returned; unknown usage is not
zero.

PNGs must be opaque and match the configured canvas. Wrong-sized outputs are
preserved for inspection and stop the compile before another image request.
ffmpeg repeats poses without interpolation; ffprobe verifies dimensions, FPS,
frame count, and duration. Exports never overwrite earlier videos.

## Lower-level commands and existing projects

For interactive, per-frame work, `init`, `frame`, `view`, `timeline`, and
`render` remain available:

```sh
uv run stop-motion init projects/butterfly-new --brief-file examples/brief.txt
uv run --env-file .env stop-motion frame --project projects/butterfly-new --prompt-file examples/opening.txt
uv run --env-file .env stop-motion frame --project projects/butterfly-new --base f0001 --prompt-file examples/step.txt
uv run stop-motion view --project projects/butterfly-new --frame f0001 --frame f0002 --contact-sheet
uv run stop-motion timeline --project projects/butterfly-new --file examples/timeline.json
uv run stop-motion render --project projects/butterfly-new
```

Use returned frame IDs rather than assuming them after a failed request.
For OpenAI Responses, `frame --base` continues that selected edit's response chain; selecting an opening
starts a new uploaded-PNG edit chain. Selecting an earlier edit creates a branch
without including later rejected candidates. No `--base` means a new standalone
opening. Extra `--reference` images are rejected for the Responses workflow.

`status --project PATH` returns saved state, usage, and incomplete attempts.
`view` returns image paths or a contact sheet; the host must actually open them.
`timeline` accepts an array of `{"frame_id": "f0001", "hold": 3}` entries.
`render` always creates a fresh export from the selected timeline.

Existing OpenAI manifests **without a backend field keep the legacy Images API**;
they are not silently migrated. Explicit `init --backend images` preserves
that experimental option, including `--reference` inputs. New `init` defaults
to `responses` and accepts `--driver-model` plus the image/timing/budget flags.

Do not mix manual `frame` calls with a compiled build; the CLI rejects this.
Inspect with `view`/`status`, but revise its source scene and compile again.
Use a separate project for manual branching.

## Earlier experiments

The scripts under `scripts/` remain bounded historical experiments, not the
recommended agent interface. In particular, `compare_models` generated an opening
inside the edit conversation and `compare_latest_image` used image-ID-only
context; both produced poor robot continuity. Don't use them as the default
pipeline or as a clean model-quality comparison.

`scripts.reproduce_uploaded_opening` contains the successful uploaded-opening
butterfly/robot reproduction, with six calls per arm and a visual-review gate.
Existing experiment outputs remain untouched. The robot brief's older
[experiment notes](examples/robot-wave/README.md) describe that history.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests use fake generators, mocked Gemini HTTP transport, and the real OpenAI SDK
with mocked transport.
Integration tests use real ffmpeg/ffprobe when installed. They verify request
parity with the successful experiment, resumability, imported openings, request
caps, failure recovery, and retiming without paid requests. No test calls the
live API. See [DESIGN.md](DESIGN.md) for the implementation contract.
