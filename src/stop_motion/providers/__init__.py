"""Built-in providers; importing their configuration never creates a network client."""

from ..storage import HarnessError


def get_provider(name: str):
    if name == "openai":
        from . import openai

        return openai
    raise HarnessError(f"Unknown image provider: {name!r}. Available providers: openai.")
