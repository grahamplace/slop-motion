"""Resolve authoring defaults and legacy manifests into one generation configuration."""

import re

from .providers import get_provider
from .storage import HarnessError

COMMON_DEFAULTS = {"size": "1024x1024", "fps": 24, "max_image_requests": 20}
OPENAI_ALIASES = {"backend", "quality", "driver_model"}
SETTING_FIELDS = set(COMMON_DEFAULTS) | {"provider", "model", "provider_options"} | OPENAI_ALIASES


def positive_integer(value) -> bool:
    return type(value) is int and value > 0


def canvas_size(size: str) -> tuple[int, int]:
    if not isinstance(size, str) or not re.fullmatch(r"\d{1,4}x\d{1,4}", size):
        raise HarnessError("Size must be explicit WIDTHxHEIGHT, such as 1024x1024.")
    width, height = map(int, size.split("x"))
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise HarnessError("Canvas dimensions must be positive even integers for video encoding.")
    return width, height


def resolve_settings(supplied: dict, *, legacy_manifest: bool = False) -> dict:
    if not isinstance(supplied, dict) or set(supplied) - SETTING_FIELDS:
        raise HarnessError(
            f"Settings must be an object with only: {', '.join(sorted(SETTING_FIELDS))}."
        )
    name = supplied.get("provider", "openai")
    provider = get_provider(name)
    options = supplied.get("provider_options", {})
    if not isinstance(options, dict) or set(options) - provider.DEFAULT_OPTIONS.keys():
        raise HarnessError(f"Invalid provider_options for {name}; unknown option or non-object.")
    options = dict(options)
    for key in OPENAI_ALIASES & supplied.keys():
        if name != "openai":
            raise HarnessError(f"{key} is an OpenAI-only setting.")
        if key in options and options[key] != supplied[key]:
            raise HarnessError(f"Conflicting {key} and provider_options.{key} settings.")
        options[key] = supplied[key]
    if legacy_manifest and name == "openai" and "backend" not in options:
        options["backend"] = "images"
    settings = {
        **COMMON_DEFAULTS,
        **{key: supplied[key] for key in COMMON_DEFAULTS if key in supplied},
        "provider": name,
        "model": supplied.get("model", provider.DEFAULT_MODEL),
        "provider_options": {**provider.DEFAULT_OPTIONS, **options},
    }
    for key in ("fps", "max_image_requests"):
        if not positive_integer(settings[key]):
            raise HarnessError(f"{key} must be a positive integer.")
    provider.validate(
        settings["model"], canvas_size(settings["size"]), settings["provider_options"]
    )
    return settings


def validate_settings(settings: dict) -> tuple[int, int]:
    """Validate persisted settings without filling in missing common manifest fields."""
    if not isinstance(settings, dict) or not {"model", *COMMON_DEFAULTS} <= settings.keys():
        raise HarnessError("Invalid project settings: missing model, size, fps, or request cap.")
    resolved = resolve_settings(settings, legacy_manifest=True)
    return canvas_size(resolved["size"])


def generation_settings(settings: dict, *, legacy_manifest: bool = False) -> dict:
    return {
        key: value
        for key, value in resolve_settings(settings, legacy_manifest=legacy_manifest).items()
        if key not in {"fps", "max_image_requests"}
    }
