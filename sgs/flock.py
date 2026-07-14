"""flock: single-instance lock for long-running solver processes.

Why
---
Two solver invocations on the same ``shareId`` race against the oracle's
server-side "already guessed" lock, plus the same word might be probed
twice, which the server silently turns into ``data: null`` (case-5 in the
case study). We want the second invocation to **fail fast**, not silently
corrupt the replay log.

This is the standard ``fcntl.flock(LOCK_EX | LOCK_NB)`` recipe — the same
one we use for skw-trade-agent workers and cron daemons (see the
``single-instance-flock`` skill).

Lock path
---------
``/tmp/sgs-<name>.lock`` by default. The file is intentionally **kept** on
release so a future ``lsof`` can still see who held it.

Scope
-----
Linux / WSL / macOS only. Windows would need ``msvcrt``; not supported.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

logger = logging.getLogger("sgs.flock")

LOCK_DIR = Path(os.environ.get("SGS_LOCK_DIR", "/tmp"))

__all__ = ["AlreadyRunningError", "SingleInstanceLock", "single_instance"]


class AlreadyRunningError(RuntimeError):
    """Raised when the lock is held by another process."""


class SingleInstanceLock:
    """Context manager that grabs ``/tmp/sgs-<name>.lock`` exclusively.

    Example
    -------
    >>> with SingleInstanceLock("playwright-probe") as lock:
    ...     # do work
    ...     ...
    """

    def __init__(self, name: str, lock_dir: Path | None = None) -> None:
        import sgs.flock as _flock_mod

        self.name = name
        # Read LOCK_DIR at construction time so tests can monkeypatch it —
        # using a module-level `from … import LOCK_DIR` here would pin the
        # value at import-time and ignore the patch.
        self.lock_path = (lock_dir or _flock_mod.LOCK_DIR) / f"sgs-{name}.lock"
        self._fd: object | None = None  # type: ignore[assignment]  # noqa: ERA001

    def __enter__(self) -> "SingleInstanceLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Open with O_RDWR so we can write a diagnostic banner even if the
        # file already exists from a previous run.
        self._fd = open(self.lock_path, "w+", encoding="utf-8")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[arg-type]
        except BlockingIOError:
            prev = self._peek_holder_pid()
            try:
                self._fd.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._fd = None
            raise AlreadyRunningError(
                f"[{self.name}] 另一个 solver 进程正在运行 "
                f"(lock={self.lock_path}, holder={prev})"
            )

        self._fd.seek(0)  # type: ignore[union-attr]
        self._fd.truncate()  # type: ignore[union-attr]
        self._fd.write(  # type: ignore[union-attr]
            f"pid={os.getpid()} ppid={os.getppid()} started_at={os.getpid()}\n"
        )
        self._fd.flush()  # type: ignore[union-attr]
        logger.info("🔒 获得单实例锁: %s (pid=%d)", self.lock_path, os.getpid())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
                self._fd.close()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("释放锁失败: %s", e)
        # Intentionally do NOT delete the file — keeping it lets `lsof`
        # show the previous holder's diagnostic banner.
        return False

    def _peek_holder_pid(self) -> str:
        """Try to surface the PID currently holding the lock.

        Reads the lockfile banner that :meth:`__enter__` writes (avoids
        spawning ``lsof`` which can take 200 ms+ on a cold WSL start).
        """
        try:
            with open(self.lock_path, "r", encoding="utf-8") as fp:
                banner = fp.read(200)
            for line in banner.splitlines():
                if line.startswith("pid="):
                    return line.strip()
        except OSError:
            pass
        return "(unknown)"


_P = ParamSpec("_P")
_R = TypeVar("_R")


def single_instance(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator flavour of :class:`SingleInstanceLock`."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with SingleInstanceLock(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator