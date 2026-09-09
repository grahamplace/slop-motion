"""The only code that contacts OpenAI; no client or credentials at import time."""

import base64
import binascii
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai import APIConnectionError, APIStatusError, OpenAI

from .storage import HarnessError


@dataclass
class GeneratedImage:
    png: bytes
    usage: dict | None = None
    request_id: str | None = None
    metadata: dict | None = None


class ImageRequestError(HarnessError):
    def __init__(
        self,
        message: str,
        *,
        unknown: bool = False,
        request_id: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(message)
        self.unknown = unknown
        self.request_id = request_id
        self.metadata = metadata


class ImageGenerator(Protocol):
    def generate(self, request: dict, inputs: list[Path]) -> GeneratedImage: ...


class OpenAIImages:
    def __init__(self, client: OpenAI | None = None):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise HarnessError("Set OPENAI_API_KEY before generating a frame.")
            client = OpenAI(max_retries=0, timeout=180.0)
        # Also disable retries on injected SDK clients, so the harness owns the request count.
        self.client = client.with_options(max_retries=0)

    def generate(self, request: dict, inputs: list[Path]) -> GeneratedImage:
        try:
            with ExitStack() as stack:
                if inputs:
                    files = [stack.enter_context(path.open("rb")) for path in inputs]
                    response = self.client.images.edit(**request, image=files)
                else:
                    response = self.client.images.generate(**request)
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
DEFAULT_DRIVER = "gpt-5.5"


def response_request(request: dict, inputs: list[Path]) -> dict:
    """Build the tested bootstrap/chain; never put PNG bytes in the saved manifest."""
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
            if not os.environ.get("OPENAI_API_KEY"):
                raise HarnessError("Set OPENAI_API_KEY before generating a frame.")
            client = OpenAI(max_retries=0, timeout=240.0)
        self.client = client.with_options(max_retries=0)

    def generate(self, request: dict, inputs: list[Path]) -> GeneratedImage:
        payload = response_request(request, inputs)
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
                if actual is not None and actual != request[key]:
                    raise ValueError(f"Returned image {key} does not match the request")
            png = base64.b64decode(call.result, validate=True)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise ImageRequestError(
                f"Invalid image response: {exc}. It may be billable; no retry made.",
                unknown=True,
                request_id=request_id,
                metadata=metadata,
            ) from exc
        return GeneratedImage(png, metadata["response_usage"], request_id, metadata)
