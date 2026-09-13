"""Gemini image generation and editing through the Interactions HTTP interface."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
from dataclasses import replace
from typing import TYPE_CHECKING

from PIL import Image

from ..images import GeneratedImage, ImageRequest, ImageRequestError
from ..storage import HarnessError

if TYPE_CHECKING:
    import httpx

DEFAULT_MODEL = "gemini-3.1-flash-image"
DEFAULT_OPTIONS = {}
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
WORKFLOW = "gemini.interactions.uploaded-opening.v1"
# Explicit documented sizes: do not approximate a ratio or silently resize a frame.
# https://ai.google.dev/gemini-api/docs/image-generation#aspect_ratios_and_image_size
_ONE_K = {
    "1:1": (1024, 1024),
    "2:3": (848, 1264),
    "3:2": (1264, 848),
    "3:4": (896, 1200),
    "4:3": (1200, 896),
    "4:5": (928, 1152),
    "5:4": (1152, 928),
    "9:16": (768, 1376),
    "16:9": (1376, 768),
    "21:9": (1584, 672),
}
SIZE_CONFIG = {
    (width * scale, height * scale): (ratio, f"{scale}K")
    for ratio, (width, height) in _ONE_K.items()
    for scale in (1, 2, 4)
}
OPENING_INSTRUCTIONS = "Generate exactly one opening image. Return the image without commentary."
EDIT_INSTRUCTIONS = (
    "Edit the latest image to apply the user's requested change. Preserve the scene, "
    "camera, lighting, and all other details. Return exactly one image without commentary."
)


def validate(model: str, dimensions: tuple[int, int], options: dict):
    if model != DEFAULT_MODEL:
        raise HarnessError(f"Supported Gemini image model: {DEFAULT_MODEL}.")
    if dimensions not in SIZE_CONFIG:
        raise HarnessError(
            "Unsupported Gemini canvas size. Use a documented 1K, 2K, or 4K size, "
            "such as 1024x1024, 1376x768, or 2048x2048. No resizing is performed."
        )


def workflow(settings: dict) -> str:
    return WORKFLOW


def describe_workflow(settings: dict) -> str:
    return "separate-opening -> uploaded-PNG-first-edit -> previous_interaction_id"


def supports_compile(settings: dict) -> bool:
    return True


def prepare(request: ImageRequest, parent: dict | None) -> ImageRequest:
    if request.references:
        raise HarnessError("Gemini uses one --base and its edit chain; references are unsupported.")
    if parent is not None:
        validate_result(parent)
        if parent["request"]["action"] == "edit":
            return replace(request, continuation=parent["continuation"])
    return request


def saved_request(attempt: dict) -> dict:
    return attempt["request"]


def validate_result(attempt: dict):
    state = attempt.get("continuation")
    if (
        not isinstance(state, dict)
        or set(state) != {"interaction_id"}
        or not isinstance(state["interaction_id"], str)
        or not state["interaction_id"].strip()
    ):
        raise HarnessError("Missing Gemini interaction metadata; cannot resume this build.")


def interaction_request(request: ImageRequest) -> dict:
    dimensions = tuple(map(int, request.settings["size"].split("x")))
    ratio, resolution = SIZE_CONFIG[dimensions]
    payload = {
        "model": request.settings["model"],
        "input": request.prompt,
        "system_instruction": EDIT_INSTRUCTIONS
        if request.action == "edit"
        else OPENING_INSTRUCTIONS,
        "store": True,
        "stream": False,
        "background": False,
        "response_format": {
            "type": "image",
            "aspect_ratio": ratio,
            "image_size": resolution,
            # The OpenAPI spec accepts JPEG output. Normalize decoded pixels to PNG locally.
            # https://ai.google.dev/static/api/interactions.openapi.json
            "mime_type": "image/jpeg",
            # Leave delivery unspecified: the live endpoint rejects an explicit selector.
        },
    }
    if request.continuation:
        payload["previous_interaction_id"] = request.continuation["interaction_id"]
    elif request.base is not None:
        payload["input"] = [
            {
                "type": "image",
                "mime_type": "image/png",
                "data": base64.b64encode(request.base.read_bytes()).decode("ascii"),
            },
            {"type": "text", "text": request.prompt},
        ]
    return payload


def create_generator(settings: dict):
    return GeminiImages()


class GeminiImages:
    def __init__(self, *, api_key: str | None = None, transport: httpx.BaseTransport | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise HarnessError("Set GEMINI_API_KEY before generating a Gemini frame.")
        self.transport = transport

    def generate(self, request: ImageRequest) -> GeneratedImage:
        import httpx

        payload = interaction_request(request)
        try:
            # One POST per attempt; redirects and automatic transport retries are disabled.
            with httpx.Client(
                transport=self.transport or httpx.HTTPTransport(retries=0),
                timeout=httpx.Timeout(240.0, connect=10.0),
                follow_redirects=False,
            ) as client:
                response = client.post(
                    ENDPOINT, json=payload, headers={"x-goog-api-key": self.api_key}
                )
        except httpx.TransportError as exc:
            raise ImageRequestError(
                "Gemini connection failed or timed out; its remote outcome is unknown. "
                "No automatic retry was made.",
                unknown=True,
            ) from exc
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "x-goog-request-id"
        )
        if not response.is_success:
            hint = ""
            if response.status_code == 429:
                hint = " Check the API project's quota and billing."
            elif request.continuation and response.status_code in {400, 404}:
                hint = (
                    " If the saved interaction expired or was rejected, use a new project;"
                    " the edit chain was not restarted."
                )
            raise ImageRequestError(
                f"Gemini returned HTTP {response.status_code}. No automatic retry was made.{hint}",
                unknown=response.status_code >= 500 or response.status_code == 408,
                request_id=request_id,
                metadata={"http_status": response.status_code},
            )
        metadata = {}
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Expected an interaction object")
            metadata = {
                "interaction_id": body.get("id"),
                "interaction_status": body.get("status"),
                "response_model": body.get("model"),
                "usage": body.get("usage"),
            }
            if (
                body.get("status") != "completed"
                or not isinstance(body.get("id"), str)
                or not body["id"].strip()
            ):
                raise ValueError("Expected a completed interaction with an ID")
            if body.get("model") is not None and body["model"] != request.settings["model"]:
                raise ValueError("Returned model differs from the requested image model")
            if "previous_interaction_id" in body and body["previous_interaction_id"] != payload.get(
                "previous_interaction_id"
            ):
                raise ValueError("Returned parent interaction differs from the selected base")
            if body.get("usage") is not None and not isinstance(body["usage"], dict):
                raise ValueError("Invalid usage metadata")
            images = [
                block
                for step in body["steps"]
                if step["type"] == "model_output"
                for block in step["content"]
                if block["type"] == "image"
            ]
            metadata["output_image_count"] = len(images)
            if len(images) != 1:
                raise ValueError("Expected exactly one final image")
            block = images[0]
            mime = block.get("mime_type")
            if mime not in {"image/png", "image/jpeg"}:
                raise ValueError("Expected a PNG or JPEG image")
            source = base64.b64decode(block["data"], validate=True)
            with Image.open(io.BytesIO(source)) as image:
                image.load()
                if image.format != {"image/png": "PNG", "image/jpeg": "JPEG"}[mime] or getattr(
                    image, "is_animated", False
                ):
                    raise ValueError("Image bytes do not match their declared format")
                if mime == "image/png":
                    png = source
                else:
                    output = io.BytesIO()
                    image.convert("RGB").save(output, format="PNG")
                    png = output.getvalue()
            metadata.update(source_mime_type=mime, source_sha256=hashlib.sha256(source).hexdigest())
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            binascii.Error,
            Image.DecompressionBombError,
        ) as exc:
            raise ImageRequestError(
                "Gemini returned an invalid or incomplete image result. The request may be "
                "billable; no automatic retry was made.",
                unknown=True,
                request_id=request_id,
                metadata=metadata,
            ) from exc
        return GeneratedImage(
            png, body.get("usage"), request_id, metadata, {"interaction_id": body["id"]}
        )
