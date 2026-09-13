"""The frame-generation seam: local preparation, durable audit records, one remote submission."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .providers import get_provider
from .settings import generation_settings
from .storage import HarnessError


@dataclass(frozen=True)
class ImageRequest:
    settings: dict
    workflow: str
    prompt: str
    action: Literal["generate", "edit"]
    base: Path | None = None
    references: tuple[Path, ...] = ()
    continuation: dict | None = None

    @property
    def input_paths(self) -> list[Path]:
        return ([self.base] if self.base else []) + list(self.references)

    def record(self) -> dict:
        # File identities live in the attempt/frame ledger, never inline image bytes.
        return deepcopy(
            {
                "settings": self.settings,
                "workflow": self.workflow,
                "prompt": self.prompt,
                "action": self.action,
                "continuation": self.continuation,
            }
        )


@dataclass
class GeneratedImage:
    png: bytes
    usage: dict | None = None
    request_id: str | None = None
    metadata: dict | None = None
    continuation: dict | None = None


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
    def generate(self, request: ImageRequest) -> GeneratedImage: ...


def prepare_request(
    settings: dict,
    prompt: str,
    *,
    base: Path | None = None,
    references: list[Path] | None = None,
    parent: dict | None = None,
) -> ImageRequest:
    config = generation_settings(settings)
    provider = get_provider(config["provider"])
    request = ImageRequest(
        config,
        provider.workflow(config),
        prompt,
        "edit" if base else "generate",
        base,
        tuple(references or []),
    )
    if parent is not None:
        saved = _saved_request(provider, parent)
        if (
            parent["status"] != "succeeded"
            or saved["settings"] != config
            or saved["workflow"] != request.workflow
            or parent.get("provider", "openai") != config["provider"]
        ):
            raise HarnessError("Edit-chain settings changed or parent failed; start a new project.")
    return provider.prepare(request, parent)


def validate_saved(attempt: dict, request: ImageRequest):
    provider = get_provider(request.settings["provider"])
    if (
        _saved_request(provider, attempt) != request.record()
        or attempt.get("provider", "openai") != request.settings["provider"]
    ):
        raise HarnessError("Saved generation request or continuation is inconsistent.")
    provider.validate_result(attempt)


def _saved_request(provider, attempt: dict) -> dict:
    try:
        record = provider.saved_request(attempt)
        if (
            not isinstance(record, dict)
            or not {"settings", "workflow", "action", "prompt", "continuation"} <= record.keys()
            or record["action"] not in {"generate", "edit"}
        ):
            raise ValueError("Invalid saved request")
        return record
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessError(
            "Invalid saved generation request; inspect the project manifest."
        ) from exc


def create_generator(settings: dict) -> ImageGenerator:
    return get_provider(settings["provider"]).create_generator(settings)
