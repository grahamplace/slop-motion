# Gemini robot validation

On 2026-09-12, `scene-gemini.json` completed a live run using
`gemini-3.1-flash-image` at 1024x1024, with adapter revision `1b804ba`.
The build was `projects/robot-gemini-funded-2026-09-12`.

## Execution

The first CLI process compiled through pose 2. After visual inspection, a second
process resumed the same build and generated the remaining five poses. A third
process, with both provider API keys unset, reused the completed video.

| Check | Result |
| --- | --- |
| Submissions in this build | Seven succeeded; no failures or retries |
| Opening and first edit | Standalone opening, then uploaded PNG in a fresh interaction |
| Remaining edits | Five requests continued the selected parent's saved interaction ID |
| Saved images | Seven opaque 1024x1024 PNGs, decoded from JPEG responses |
| Checkpoint preservation | Opening and first-edit hashes unchanged after resume |
| Manifest validation | Frame hashes and saved parent interaction links matched |
| Video | 42 frames at 24 fps, 1024x1024, 1.75 seconds |
| Completed-build rerun | Zero requests; existing video reused without API keys |

The provider reported 10,783 input tokens and 9,920 output tokens across the seven
requests. These are API usage counters, not a dollar charge or billing estimate.
Earlier failed validation builds remain separate from this successful build.

Local artifacts include `frames/f0001.png` through `frames/f0007.png`,
`previews/contact-0001.png`, and `exports/export-0002.mp4` under the build directory.
The MP4 contains six-frame holds of each generated pose, without interpolation.

## Visual inspection

All seven PNGs and their contact sheet were inspected. The robot remained fully
visible with its cream faceplate, two eyes, smile, three buttons, and readable
blue "07" badge. The stationary arm, feet, blocks, and camera framing remained
visually stable. Clay texture and small surface details changed across edits;
the supposedly fixed areas were not pixel-identical.

Pose following was uneven. The first edit raised the arm farther than its
30-degree target. The second edit pointed diagonally upward instead of following
the requested 60-degree target from vertically downward. The third edit returned
to horizontal, causing an unintended reversal. The final three poses remained
at similar upward angles rather than clearly following the requested
120-to-100-to-120-degree wave. Hand shape and orientation also changed.

This run verifies the live integration and resume workflow. It demonstrates
useful character preservation, but does not validate precise arm-angle control
or smooth motion. It is one scene and one run, not a controlled vendor comparison.
