"""OpenAI-specific configuration and generation constraints."""

from __future__ import annotations

import base64
import binascii
import os
from contextlib import ExitStack
from dataclasses import replace
from typing import TYPE_CHECKING

from ..images import GeneratedImage, ImageRequest, ImageRequestError
from ..settings import generation_settings
from ..storage import HarnessError

if TYPE_CHECKING:
    from openai import OpenAI

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


def workflow(settings: dict) -> str:
    return (
        "openai.responses.uploaded-opening.v1"
        if settings["provider_options"]["backend"] == "responses"
        else "openai.images.v1"
    )


def describe_workflow(settings: dict) -> str:
    if settings["provider_options"]["backend"] == "responses":
        return "separate-opening -> uploaded-PNG-first-edit -> previous_response_id"
    return "direct generation or edit with ordered input PNGs"


def supports_compile(settings: dict) -> bool:
    return settings["provider_options"]["backend"] == "responses"


def prepare(request: ImageRequest, parent: dict | None) -> ImageRequest:
    if request.settings["provider_options"]["backend"] == "images":
        return request
    if request.references:
        raise HarnessError(
            "Responses uses one --base and its edit chain; references are unsupported."
        )
    if parent is not None:
        validate_result(parent)
        # Generated openings never seed the edit conversation.
        if saved_request(parent)["action"] == "edit":
            return replace(request, continuation=result_state(parent))
    return request


def result_state(attempt: dict) -> dict | None:
    if "settings" in attempt["request"]:
        return attempt.get("continuation")
    return {"response_id": attempt.get("response_id")}


def validate_result(attempt: dict):
    if saved_request(attempt)["settings"]["provider_options"]["backend"] == "responses":
        state = result_state(attempt)
        if (
            not isinstance(state, dict)
            or set(state) != {"response_id"}
            or not isinstance(state["response_id"], str)
            or not state["response_id"].strip()
        ):
            raise HarnessError("Missing response metadata; cannot resume this build.")


def saved_request(attempt: dict) -> dict:
    raw = attempt["request"]
    if "settings" in raw:
        return raw
    # Historical OpenAI requests stay byte-for-byte intact in the manifest.
    responses = "action" in raw
    settings = generation_settings(
        {
            "model": raw["model"],
            "size": raw["size"],
            "quality": raw["quality"],
            "backend": "responses" if responses else "images",
            **({"driver_model": raw["driver_model"]} if responses else {}),
        }
    )
    previous = raw.get("previous_response_id")
    return {
        "settings": settings,
        "workflow": workflow(settings),
        "action": raw.get("action", "edit" if attempt["base_frame_id"] else "generate"),
        "prompt": raw["prompt"],
        "continuation": {"response_id": previous} if previous is not None else None,
    }


def image_options(request: ImageRequest) -> dict:
    settings = request.settings
    options = settings["provider_options"]
    result = {
        "model": settings["model"],
        "size": settings["size"],
        "quality": options["quality"],
        "background": "opaque",
        "output_format": "png",
        "prompt": request.prompt,
    }
    if options["backend"] == "images":
        result["n"] = 1
    else:
        result.update(
            action=request.action,
            driver_model=options["driver_model"],
            previous_response_id=(request.continuation or {}).get("response_id"),
        )
    return result


def create_generator(settings: dict):
    return OpenAIResponses() if supports_compile(settings) else OpenAIImages()


class OpenAIImages:
    def __init__(self, client: OpenAI | None = None):
        if client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise HarnessError("Set OPENAI_API_KEY before generating a frame.")
            client = OpenAI(max_retries=0, timeout=180.0)
        # Also disable retries on injected SDK clients, so the harness owns the request count.
        self.client = client.with_options(max_retries=0)

    def generate(self, request: ImageRequest) -> GeneratedImage:
        from openai import APIConnectionError, APIStatusError

        inputs = request.input_paths
        payload = image_options(request)
        try:
            with ExitStack() as stack:
                if inputs:
                    files = [stack.enter_context(path.open("rb")) for path in inputs]
                    response = self.client.images.edit(**payload, image=files)
                else:
                    response = self.client.images.generate(**payload)
        except APIConnectionError as exc:
            raise ImageRequestError(
                "Image request connection failed or timed out; its remote outcome is unknown. "
                "No automatic retry was made.",
                unknown=True,
            ) from exc
        except APIStatusError as exc:
            # Persist useful provider fields, without logging credentials or HTTP headers.
            code = exc.code if isinstance(exc.code, str) else "api_error"
            raise ImageRequestError(
                f"OpenAI returned HTTP {exc.status_code} ({code}). No automatic retry was made.",
                unknown=exc.status_code >= 500,
                request_id=exc.request_id,
            ) from exc

        request_id = getattr(response, "_request_id", None)
        try:
            encoded = response.data[0].b64_json
            if not encoded:
                raise ValueError("Missing image data")
            png = base64.b64decode(encoded, validate=True)
        except (IndexError, TypeError, ValueError, binascii.Error) as exc:
            raise ImageRequestError(
                "OpenAI returned no valid base64 image. The request may still be billable.",
                request_id=request_id,
            ) from exc
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        return GeneratedImage(png, usage, request_id)


EDIT_INSTRUCTIONS = (
    "For each user request, call the image_generation tool exactly once to edit "
    "the latest image in the conversation. Return the image without commentary."
)
OPENING_INSTRUCTIONS = (
    "Call the image_generation tool exactly once to generate the opening image. "
    "Return the image without commentary."
)


def response_request(frame: ImageRequest) -> dict:
    """Build the tested bootstrap/chain; never put PNG bytes in the saved manifest."""
    request = image_options(frame)
    inputs = frame.input_paths
    action = request["action"]
    parent = request.get("previous_response_id")
    if action not in {"generate", "edit"}:
        raise HarnessError("Responses requires an explicit generate or edit action.")
    if (action == "generate" and (inputs or parent)) or (
        action == "edit" and (len(inputs) != 1 or (parent is not None and not parent))
    ):
        raise HarnessError("An edit needs one base frame; an opening needs no image or history.")
    result = {
        "model": request["driver_model"],
        "instructions": EDIT_INSTRUCTIONS if action == "edit" else OPENING_INSTRUCTIONS,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 4096,
        "tools": [
            {
                "type": "image_generation",
                **{
                    key: request[key]
                    for key in ("model", "action", "quality", "size", "background", "output_format")
                },
            }
        ],
        "tool_choice": {"type": "image_generation"},
        "max_tool_calls": 1,
        "parallel_tool_calls": False,
        "store": True,
        "input": request["prompt"],
    }
    if parent:
        result["previous_response_id"] = parent
    elif action == "edit":
        encoded = base64.b64encode(inputs[0].read_bytes()).decode("ascii")
        result["input"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded}",
                        "detail": "high",
                    },
                    {"type": "input_text", "text": request["prompt"]},
                ],
            }
        ]
    return result


class OpenAIResponses:
    """One billed Responses request per frame; callers persist all chain state."""

    def __init__(self, client: OpenAI | None = None):
        if client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise HarnessError("Set OPENAI_API_KEY before generating a frame.")
            client = OpenAI(max_retries=0, timeout=240.0)
        self.client = client.with_options(max_retries=0)

    def generate(self, request: ImageRequest) -> GeneratedImage:
        from openai import APIConnectionError, APIStatusError

        payload = response_request(request)
        expected = image_options(request)
        try:
            response = self.client.responses.create(**payload)
        except APIConnectionError as exc:
            raise ImageRequestError(
                "Response connection failed or timed out; remote outcome unknown. No retry made.",
                unknown=True,
            ) from exc
        except APIStatusError as exc:
            code = exc.code if isinstance(exc.code, str) else "api_error"
            raise ImageRequestError(
                f"OpenAI returned HTTP {exc.status_code} ({code}). No retry made; "
                "no fallback to a different model or image workflow was attempted.",
                unknown=exc.status_code >= 500,
                request_id=exc.request_id,
            ) from exc
        request_id = getattr(response, "_request_id", None)
        calls = [item for item in response.output if item.type == "image_generation_call"]
        metadata = {
            "response_id": response.id,
            "response_status": response.status,
            "response_model": response.model,
            "response_usage": response.usage.model_dump(mode="json") if response.usage else None,
            "image_calls": [call.model_dump(mode="json", exclude={"result"}) for call in calls],
        }
        try:
            if response.status != "completed" or not response.id or len(calls) != 1:
                raise ValueError("Expected exactly one completed image call and a response ID")
            call = calls[0]
            if call.status != "completed" or not call.result:
                raise ValueError("Image tool did not complete")
            for key in ("model", "action", "quality", "size", "background", "output_format"):
                actual = getattr(call, key, None)
                if actual is not None and actual != expected[key]:
                    raise ValueError(f"Returned image {key} does not match the request")
            png = base64.b64decode(call.result, validate=True)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise ImageRequestError(
                f"Invalid image response: {exc}. It may be billable; no retry made.",
                unknown=True,
                request_id=request_id,
                metadata=metadata,
            ) from exc
        return GeneratedImage(
            png, metadata["response_usage"], request_id, metadata, {"response_id": response.id}
        )
