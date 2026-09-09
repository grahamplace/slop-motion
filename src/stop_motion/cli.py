"""JSON on stdout, diagnostics on stderr, with one subcommand per operation."""

import argparse
import json
import sys
from pathlib import Path

from .images import DEFAULT_DRIVER
from .project import DEFAULT_MODEL, Project
from .scene import compile_scene
from .storage import HarnessError, read_json


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Local tools for agent-directed stop-motion videos.")
    commands = cli.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a project in a new or empty directory.")
    init.add_argument("path", type=Path)
    init.add_argument("--brief-file", type=Path, required=True)
    init.add_argument("--model", default=DEFAULT_MODEL)
    init.add_argument("--size", default="1024x1024")
    init.add_argument("--quality", default="medium")
    init.add_argument("--fps", type=int, default=24)
    init.add_argument("--max-image-requests", type=int, default=20)
    init.add_argument("--backend", choices=("responses", "images"), default="responses")
    init.add_argument("--driver-model", default=DEFAULT_DRIVER)

    compile_command = commands.add_parser("compile", help="Compile an editable scene to a video.")
    compile_command.add_argument("--scene", type=Path, required=True)
    compile_command.add_argument("--project", type=Path, required=True)
    compile_command.add_argument(
        "--dry-run", action="store_true", help="Validate without writes or API calls."
    )
    compile_command.add_argument(
        "--through", type=int, help="Build only this many poses, including opening."
    )

    for name, help_text in (
        ("status", "Read saved state, usage, and incomplete attempts."),
        ("frame", "Generate or edit one image and save it."),
        ("view", "Return images for visual inspection, optionally as a contact sheet."),
        ("timeline", "Replace the selected sequence and holds."),
        ("render", "Encode and verify a new MP4 from the timeline."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--project", type=Path, required=True)
        if name == "frame":
            command.add_argument("--prompt-file", type=Path, required=True)
            command.add_argument("--base")
            command.add_argument("--reference", action="append", default=[])
        elif name == "view":
            command.add_argument("--frame", action="append", required=True)
            command.add_argument("--contact-sheet", action="store_true")
        elif name == "timeline":
            command.add_argument("--file", type=Path, required=True)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project = None
    try:
        if args.command == "init":
            project = Project.create(
                args.path,
                args.brief_file.read_text(encoding="utf-8"),
                model=args.model,
                size=args.size,
                quality=args.quality,
                fps=args.fps,
                max_image_requests=args.max_image_requests,
                backend=args.backend,
                driver_model=args.driver_model,
            )
            result = project.status()
        elif args.command == "compile":
            result = compile_scene(
                args.scene,
                args.project,
                dry_run=args.dry_run,
                through=args.through,
                progress=lambda event: print(json.dumps(event), file=sys.stderr, flush=True),
            )
        else:
            project = Project(args.project)
            if args.command == "status":
                result = project.status()
            elif args.command == "frame":
                result = project.make_frame(
                    args.prompt_file.read_text(encoding="utf-8"),
                    base_frame=args.base,
                    references=args.reference,
                )
            elif args.command == "view":
                result = project.view(args.frame, contact_sheet=args.contact_sheet)
            elif args.command == "timeline":
                result = project.set_timeline(read_json(args.file))
            else:
                result = project.render()
        if project is not None:
            result["warnings"] = project.warnings
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (HarnessError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            json.dumps({"error": "Interrupted. Run status before retrying an image request."}),
            file=sys.stderr,
        )
        return 130
