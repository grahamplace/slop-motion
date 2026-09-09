"""Atomic local writes and an operating-system lock released on process exit."""

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


class HarnessError(Exception):
    """An actionable error suitable for a CLI caller."""


@contextmanager
def project_lock(root: Path):
    if not root.is_dir():
        raise HarnessError(f"Project directory does not exist: {root}")
    # Keep this file in place: deleting it could give concurrent callers different locks.
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HarnessError("Project is busy; another command holds its lock.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write(path: Path, content: bytes, *, replace: bool = False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as out:
            temporary = Path(out.name)
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # An atomic, exclusive publication: an existing asset can never be overwritten.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value, *, replace: bool = False):
    atomic_write(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"),
        replace=replace,
    )


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise HarnessError(f"Could not read JSON from {path}: {exc}") from exc
