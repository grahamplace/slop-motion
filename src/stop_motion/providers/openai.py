"""OpenAI-specific configuration and generation constraints."""

from ..storage import HarnessError

DEFAULT_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_DRIVER = "gpt-5.5"
DEFAULT_OPTIONS = {"backend": "responses", "quality": "medium", "driver_model": DEFAULT_DRIVER}
SUPPORTED_MODELS = {DEFAULT_MODEL, "gpt-image-2.5-flare"}
QUALITIES = {"low", "medium", "high", "xhigh", "max", "auto"}


def validate(model: str, dimensions: tuple[int, int], options: dict):
    if options["backend"] not in ("images", "responses"):
        raise HarnessError("Backend must be responses or images.")
    if not isinstance(options["driver_model"], str) or not options["driver_model"].strip():
        raise HarnessError("Responses requires a driver_model.")
    if not isinstance(model, str) or model not in SUPPORTED_MODELS:
        raise HarnessError(f"Supported image models: {', '.join(sorted(SUPPORTED_MODELS))}")
    if not isinstance(options["quality"], str) or options["quality"] not in QUALITIES:
        raise HarnessError(f"Supported qualities: {', '.join(sorted(QUALITIES))}")
    width, height = dimensions
    if not (
        width % 16 == height % 16 == 0
        and max(width, height) <= 3840
        and max(width, height) <= 3 * min(width, height)
        and 655360 <= width * height <= 8294400
    ):
        raise HarnessError(
            "Size requires multiples of 16, edges <=3840, aspect ratio between 1:3 and 3:1, "
            "and 655360–8294400 total pixels."
        )
