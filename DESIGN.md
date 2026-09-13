# Stop-motion harness — current design

## Purpose

A Python CLI lets an existing vision-capable agent author a fixed-camera scene,
generate and inspect poses, and compile them into a stop-motion MP4.
The agent owns creative planning and visual judgment. The harness owns request
construction, provider state, immutable image files, timing, and video validation.

There is no standalone planner, web UI, audio, interpolation, or parallel frame
generation. One build directory represents one shot.

## Interface

The primary interface is:

```sh
stop-motion compile --scene scenes/robot/scene.json --project projects/robot
```

`--dry-run` validates every source file and returns a no-write, no-network plan.
`--through N` generates/renders the first N poses, including the opening, for
agent visual inspection. Re-running continues the serial sequence.

The agent edits scene JSON, an opening prompt or PNG, and ordered edit prompts.
Prompts may be inline strings or UTF-8 files relative to the scene file.
Holds count output video frames. The brief is planning metadata, not hidden
prompt injection. No template expansion or automatic planning occurs.

Success emits one JSON object on stdout with absolute artifact paths.
Progress events and errors are JSON on stderr; errors have nonzero exit codes.

Lower-level `init/status/frame/view/timeline/render` commands remain available.
Compiled projects prohibit manual frame generation so the compiler's ordered
sequence stays unambiguous. Inspecting and rendering existing artifacts is local.

## Generation invariant

The default backend is Responses with GPT-5.5 driving the image tool and
GPT Image 2.5 Sunburst explicitly selected for image generation/editing.

1. Generate an opening independently, or copy an approved local PNG.
2. First edit starts a **new** Responses conversation. Upload the opening PNG
   as base64 input_image with detail high and send the first edit prompt.
   The opening-generation response ID is never used as this edit's parent.
3. Every edit uses the successful experiment's edit-only instructions and
   action edit. Force the image tool, cap it at one call, disable parallel calls,
   store the response, and use low reasoning/max_output_tokens 4096.
4. Later edits send only their user prompt and the preceding edit's
   previous_response_id. The service maintains conversation history.
5. Persist returned response IDs, request IDs, image-call metadata/revised prompts,
   usage, and output hashes. Never persist base64 payloads or API credentials.

The adapter requires exactly one completed image call and validates any returned
action/model/settings against the request. The tool may not return an image-model
ID, so a requested model is not mislabeled as an independently confirmed model.
No fallback changes the configured model or context strategy.

A generated opening has a response ID for provenance, but it is deliberately
excluded from the edit chain. An imported opening has no response ID or API cost.

Manual frame generation follows the same rule. A selected edit continues its
chain; selecting the opening uploads it into a fresh chain. Selecting an earlier
edit branches before rejected descendants. Extra references are unsupported in
this workflow.

Six-edit butterfly and robot experiments reproduced good continuity using this
initialization. They do not prove long-sequence stability, pixel preservation, or
that PNG upload alone caused the improvement: instructions changed with it.

## Modules and internal seams

- `cli.py`: argument parsing and JSON stdout/stderr contract.
- `scene.py`: the deep compile module. Its small interface loads/validates
  authoring files, reconciles completed poses, resumes requests, applies timing,
  and reuses or produces a verified video.
- `project.py`: frame inventory, immutable assets, attempts, project locking,
  and per-frame operations. Compiler internals share its locked session so a
  frame and its scene-step identity commit in the same manifest write.
- `settings.py`: resolves provider defaults, validates canvas/timing settings, and
  normalizes legacy OpenAI aliases without rewriting historical manifests.
- `providers/`: built-in provider configuration and provider-specific constraints.
- `images.py`: the ImageGenerator seam and two adapters: Responses (default)
  and legacy direct Images. Dependencies can be injected for offline tests.
- `video.py`: hold expansion, ffmpeg, and ffprobe verification.
- `storage.py`: exclusive atomic asset publication, atomic manifest replacement,
  and an operating-system project lock released on process exit.

Production code does not import experimental scripts. A regression test compares
all six serialized edit requests against the successful experiment's builder.

## Persistence and resume

Authoring files and the build directory are separate. project.json is generated
state, not the agent's configuration surface. Images carry SHA-256 hashes and
immutable frame IDs; attempts carry scene-step indexes for compiled builds.

A compile holds the project lock across generation and export. Before each
request it durably records a started attempt. It publishes the PNG exclusively,
then commits the frame, response metadata, and successful attempt together.
An interrupted command can leave an unknown request or uncommitted PNG; neither
is automatically retried or adopted. Failed and unknown attempts consume budget.

Resume checks completed prompt contents, opening PNG hash, generation settings,
saved response linkage, and output hashes. New prompts can be appended and
unbuilt prompts can change. Completed prompts or generation settings require
a new project directory. Holds, FPS, and explicit budget changes are allowed.

Every input is validated before spending; the full scene must fit the configured
request cap, including opening generation when applicable. This caps calls, not
dollars. The driver and its history also consume tokens. SDK retries are disabled.
Expired/rejected response references fail visibly without a silent rebootstrap.

Source files remain necessary for resume. Editing source during an active compile
does not affect its loaded snapshot; a later invocation checks it again.

## Export

PNGs must be opaque and match the configured canvas. Wrong-sized returned images
are kept for inspection but stop compilation. Missing ffmpeg/ffprobe is detected
before image generation.

A timeline is an ordered list of frame_id/hold entries. Duration is sum(hold)/fps.
ffmpeg repeats PNGs using hard links, with no interpolation or recompression of
source images, and produces H.264/yuv420p MP4. ffprobe checks dimensions, FPS,
frame count, and duration before publication.

Exports have unique names and adjacent snapshot JSON. Compile reuses its last
export when the timeline, image records, dimensions, and FPS match and the
cached video hash still matches. Retiming or missing/corrupt cached output
produces a new export without image calls. Manual render always exports anew.

## Compatibility and verification

Project schema remains version 1 with additive provider, hash, response, and
compile fields. New settings use provider/model/size plus provider_options; old
flat OpenAI options normalize to the same effective configuration on read. Old manifests without backend retain direct Images behavior.
New init defaults to Responses; explicit --backend images retains the old path.
Compile only accepts the working Responses workflow and will not adopt unrelated
manual projects or experiment outputs.

Tests cover the public compile interface with a real SDK plus mocked transport,
real ffmpeg/ffprobe, imported/generated openings, staged resume, exact request
parity, no duplicate spending, source/output changes, wrong-size images, request
failures, interrupted publication, project locking, and free retiming.
Legacy command behavior is retained in separate tests. Visual review remains an
agent/human responsibility, not a consequence of passing plumbing tests.

API reference:
[OpenAI image generation tool guide](https://developers.openai.com/api/docs/guides/tools-image-generation).
