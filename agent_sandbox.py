"""Filesystem confinement for the Claude CLI subprocess, via bubblewrap.

Claude Code agents in sciMAS run with cwd = the repo root and the full built-in
toolset, Bash included. State isolation (see the note in ``claude_runner``)
closes the *automatic* leak — memory injection and transcripts left where the
next run reads them — but it does not confine the agent: a run that goes
looking can still read the dataset's ground-truth files and any previous run's
output under ``tests/results/`` or ``runs-dashboard/``. An audit of this
deployment found sessions that did exactly that, and one of them calibrated its
answer to ``golden_answer`` field by field.

So the CLI is launched inside a bubblewrap sandbox that mounts the whole
filesystem read-only and masks the answer-bearing paths: a *file* is replaced
by an empty file (reads return ``""``), a *directory* by an empty tmpfs.
Masking rather than removing, because the paths have to keep existing for
anything that stats them, and because a masked file is indistinguishable from
an empty one — where ``--ro-bind /dev/null`` would make reads fail with
``EACCES`` and advertise the boundary.

What has to keep working, and how:

  tools          MCP servers are spawned by the sandboxed CLI, so they see the
                 same masked filesystem. That is intended: a tool that reads
                 the answers on the model's behalf is the same leak one step
                 removed. Nothing in the sandbox machinery touches
                 ``--mcp-config``.
  writes         The repo becomes read-only, so the run's own output directory
                 is bound writable — that is where tools write figures,
                 trajectories and evaluations. An agent that tries to scribble
                 in the repo now fails at that call instead of silently
                 succeeding; the prompts already point agents at the run
                 directory.
  credentials    Under state isolation ``CLAUDE_CONFIG_DIR`` lives in ``/tmp``,
                 and the sandbox gives ``/tmp`` a private tmpfs. That path is
                 bound back in, or the CLI would start with no credentials at
                 all. This is the one bind that is not about hiding.
  network        No network namespace is unshared (``--unshare-net`` is
                 deliberately absent), so literature and web MCP servers keep
                 working.

bubblewrap is Linux-only, so on macOS ``build_prefix`` returns ``None`` and the
run proceeds unsandboxed. That is the right default for a laptop dev run and
the wrong one for a scoring deployment, which is what ``SCIMAS_SANDBOX=require``
is for: it turns a missing sandbox into an error instead of a warning.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

BWRAP_ENV = "SCIMAS_SANDBOX"

#: ``auto`` uses the sandbox when bubblewrap is present, ``off`` never does,
#: ``require`` errors when it is not.
_MODES = ("auto", "off", "require")

_warned_unavailable = False


def sandbox_mode() -> str:
    """The configured mode: ``auto`` (default), ``off`` or ``require``."""
    value = os.environ.get(BWRAP_ENV, "").strip().lower()
    if value not in _MODES:
        if value:
            print(
                f"sandbox: ignoring {BWRAP_ENV}={value!r} (expected one of "
                f"{', '.join(_MODES)})",
                file=sys.stderr,
            )
        return "auto"
    return value


def bubblewrap() -> Optional[str]:
    """Path to ``bwrap``, or ``None`` when this machine cannot sandbox."""
    return shutil.which("bwrap")


def build_prefix(
    hide: Iterable[str] = (),
    writable: Iterable[str] = (),
    cwd: Optional[str] = None,
    mask_source: Optional[str] = None,
) -> Optional[list[str]]:
    """Return the argv prefix that sandboxes one CLI invocation.

    ``hide`` and ``writable`` are absolute paths; a path in both stays writable,
    because the run's own output directory sits inside the very tree that holds
    the previous runs. ``mask_source`` is an empty file to stand in for masked
    files: reading it gives ``""`` and ``stat`` gives size 0, where binding
    ``/dev/null`` gives ``EACCES`` — which reads as "you are not allowed to see
    this" and tells the agent a boundary exists. It has to be visible at bind
    time, i.e. under a path that is bound writable but not masked; the state
    directory is the natural home for it.

    Returns ``None`` when the sandbox is off or unavailable; raises
    ``RuntimeError`` under ``SCIMAS_SANDBOX=require``.
    """
    global _warned_unavailable
    mode = sandbox_mode()
    if mode == "off":
        return None
    bwrap = bubblewrap()
    if bwrap is None:
        message = (
            "sandbox: bubblewrap is not available on this machine, so the "
            f"agent runs unconfined (set {BWRAP_ENV}=off to silence this, or "
            f"{BWRAP_ENV}=require to make it fatal)"
        )
        if mode == "require":
            raise RuntimeError(message)
        if not _warned_unavailable:
            print(message, file=sys.stderr)
            _warned_unavailable = True
        return None

    hidden = [p for p in _resolved(hide) if p != "/tmp"]
    writable_paths = _resolved(writable, create=True)
    masked = [p for p in hidden if p not in writable_paths]
    # A writable path inside a masked tree can only be re-exposed after the
    # mask is in place, and everything else has to be bound before the masks
    # so the mask source itself survives.
    inner = [p for p in writable_paths if _under_any(p, masked)]
    outer = [p for p in writable_paths if p not in inner]

    prefix = [
        bwrap,
        # Everything readable, nothing writable: a run can still import the
        # repo's tools and read its own inputs.
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        # Private scratch: the CLI makes /tmp/cc-socks here, and MCP servers
        # get a writable /tmp without seeing the host's — which also keeps one
        # run from reading another run's state directory.
        "--tmpfs", "/tmp",
        "--unshare-pid",
        "--die-with-parent",
        "--chdir", cwd or os.getcwd(),
    ]
    for path in outer:
        prefix += _bind(path)
    for path in masked:
        prefix += _mask(path, mask_source)
    for path in inner:
        prefix += _bind(path)
    return prefix


def _resolved(paths: Iterable[str], create: bool = False) -> list[str]:
    """Absolute, deduplicated paths — order preserved, existing unless created."""
    seen: list[str] = []
    for raw in paths:
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(str(raw)))
        if create:
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                continue
        if path not in seen and os.path.exists(path):
            seen.append(path)
    return seen


def _under_any(path: str, parents: Sequence[str]) -> bool:
    for parent in parents:
        if path == parent or path.startswith(parent.rstrip(os.sep) + os.sep):
            return True
    return False


def _bind(path: str) -> list[str]:
    return ["--bind", path, path]


def _mask(path: str, mask_source: Optional[str]) -> list[str]:
    """One masking rule: an empty file for a file, an empty dir for a dir."""
    if os.path.isdir(path):
        # A tmpfs keeps the path stat-able and listable-but-empty, which reads
        # as "this directory is empty" rather than "this path is broken".
        return ["--tmpfs", path]
    source = mask_source if mask_source and os.path.isfile(mask_source) else "/dev/null"
    return ["--ro-bind", source, path]


def describe(hide: Sequence[str], writable: Sequence[str]) -> str:
    """One-line summary for a run log."""
    return (
        f"sandbox: {len(list(hide))} masked path(s), "
        f"{len(list(writable))} writable"
    )
