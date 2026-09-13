"""Built-in providers; importing their configuration never creates a network client."""

from importlib import import_module

from ..storage import HarnessError

_PROVIDERS = {
    "openai": "stop_motion.providers.openai",
    "gemini": "stop_motion.providers.gemini",
}


def get_provider(name: str):
    if not isinstance(name, str) or name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS))
        raise HarnessError(f"Unknown image provider: {name!r}. Available providers: {available}.")
    return import_module(_PROVIDERS[name])
