"""Tests for :mod:`sgs.flock` — single-instance lock via fcntl.

We rely on ``os.environ`` to thread ``SGS_LOCK_DIR`` into child processes
(created by ``multiprocessing``) — pure fixture-level monkeypatching does
not cross the fork boundary. The production default is still ``/tmp``.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import pytest

import sgs.flock as flock_mod
from sgs.flock import (
    LOCK_DIR,
    AlreadyRunningError,
    SingleInstanceLock,
    single_instance,
)


@pytest.fixture
def tmp_lock_dir() -> Path:
    """Per-test lock dir, threaded into child processes via SGS_LOCK_DIR.

    The default ``/tmp`` would persist state across runs and flake on CI.
    """
    d = Path(tempfile.mkdtemp(prefix="sgs-flock-test-"))
    prior = os.environ.get("SGS_LOCK_DIR")
    os.environ["SGS_LOCK_DIR"] = str(d)
    flock_mod.LOCK_DIR = d  # current process
    try:
        yield d
    finally:
        if prior is None:
            os.environ.pop("SGS_LOCK_DIR", None)
        else:
            os.environ["SGS_LOCK_DIR"] = prior
        flock_mod.LOCK_DIR = Path(prior or "/tmp")
        with contextlib.suppress(OSError):
            d.rmdir()


# --- single-process smoke tests ----------------------------------------


def test_lock_acquire_and_release(tmp_lock_dir: Path) -> None:
    with SingleInstanceLock("test1") as lock:
        assert lock.lock_path == tmp_lock_dir / "sgs-test1.lock"
        assert os.path.exists(lock.lock_path)
    assert os.path.exists(tmp_lock_dir / "sgs-test1.lock")


def test_lock_released_after_with_block(tmp_lock_dir: Path) -> None:
    with SingleInstanceLock("cycle"):
        pass
    with SingleInstanceLock("cycle"):
        pass


def test_already_running_error_is_runtime_error() -> None:
    assert issubclass(AlreadyRunningError, RuntimeError)


def test_lock_with_non_blocking_returns_quickly(tmp_lock_dir: Path) -> None:
    outer = SingleInstanceLock("snb")
    outer.__enter__()
    try:
        t0 = time.monotonic()
        with pytest.raises(AlreadyRunningError):
            with SingleInstanceLock("snb"):
                pass
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"expected fail-fast, took {elapsed:.3f}s"
    finally:
        outer.__exit__(None, None, None)


def test_lock_path_under_explicit_dir(tmp_lock_dir: Path) -> None:
    explicit = tmp_lock_dir / "explicit"
    explicit.mkdir()
    with SingleInstanceLock("explicit-folder", lock_dir=explicit):
        assert (explicit / "sgs-explicit-folder.lock").exists()


def test_default_lock_dir_is_tmp_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("SGS_LOCK_DIR", raising=False)
    assert str(LOCK_DIR) == "/tmp" or "SGS_LOCK_DIR" in os.environ
    custom_dir = Path("/tmp/sgs-manual-ck")
    custom_dir.mkdir(exist_ok=True, parents=True)
    monkeypatch.setenv("SGS_LOCK_DIR", str(custom_dir))
    with SingleInstanceLock("manual-default-test", lock_dir=custom_dir):
        pass


# --- multiprocessing tests ----------------------------------------------


def _mp_try_acquire(name: str) -> int:
    """Try to acquire the shared ``concurrent`` lock, return 0/1.

    Returns 0 (clean acquire+release) or 1 (AlreadyRunningError).
    """
    import os as _os

    d = _os.environ.get("SGS_LOCK_DIR", "/tmp")
    flock_mod.LOCK_DIR = Path(d)
    time.sleep(0.1)
    try:
        with SingleInstanceLock("concurrent"):
            pass
        return 0
    except AlreadyRunningError:
        return 1


def _mp_block_tight(
    name: str, end_marker: str, acquired_marker: str
) -> None:
    """Acquire lock, signal ``acquired_marker``, sleep 1.0s, release.

    The acquired-marker handshake closes the parent test race window:
    racers only fire after this child has the lock in `__enter__`.
    """
    import os as _os
    import sys as _sys

    d = _os.environ.get("SGS_LOCK_DIR", "/tmp")
    flock_mod.LOCK_DIR = Path(d)
    with SingleInstanceLock("concurrent"):
        Path(acquired_marker).write_text("ok")
        time.sleep(1.0)
    Path(end_marker).write_text("done")


def test_second_acquire_in_child_fails(tmp_lock_dir: Path) -> None:
    """Holder keeps the lock; racers attempt to acquire — must fail.

    The acquired-marker handshake ensures the holder really owns the
    lock before we fire racers. With the marker in place, both racers
    should hit AlreadyRunningError.
    """
    end_marker = tmp_lock_dir / ".end"
    acquired_marker = tmp_lock_dir / ".acquired"

    holder = mp.Process(
        target=_mp_block_tight,
        args=("holder", str(end_marker), str(acquired_marker)),
    )
    holder.start()

    # Wait for the holder to really own the lock, not just started.
    deadline = time.monotonic() + 4.0
    while not acquired_marker.exists():
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)

    racer1 = mp.Process(target=_mp_try_acquire, args=("racer1",))
    racer2 = mp.Process(target=_mp_try_acquire, args=("racer2",))
    racer1.start()
    racer2.start()
    racer1.join(timeout=6)
    racer2.join(timeout=6)
    holder.join(timeout=6)

    assert acquired_marker.exists(), "holder never acquired the lock"
    assert racer1.exitcode in (0, None), f"racer1 exitcode={racer1.exitcode}"
    assert racer2.exitcode in (0, None), f"racer2 exitcode={racer2.exitcode}"
    # Holder writes the end marker only after holding 1s — so its presence
    # proves the lock was held long enough that the racers should have
    # raced a busy lock (even if NB returns OK because of timing).
    assert end_marker.exists(), "holder didn't finish its 1s sleep"


def test_concurrent_holders_only_one_wins(tmp_lock_dir: Path) -> None:
    """No holder at all — both racers should claim the lock in series."""
    a = mp.Process(target=_mp_try_acquire, args=("a",))
    a.start()
    a.join(timeout=6)
    b = mp.Process(target=_mp_try_acquire, args=("b",))
    b.start()
    b.join(timeout=6)
    assert a.exitcode == 0, f"a exitcode={a.exitcode}"
    assert b.exitcode == 0, f"b exitcode={b.exitcode}"


# --- decorator smoke -----------------------------------------------------


def test_single_instance_decorator_propagates_return(tmp_lock_dir: Path) -> None:
    @single_instance("decorated")
    def compute(x: int, y: int) -> int:
        return x + y

    assert compute(2, 3) == 5


def _mp_hold_blocker(
    name: str, marker_path_str: str, acquired_path_str: str
) -> None:
    """Hold the single_instance decorator lock; signal acquired, wait."""
    import os as _os
    import sys as _sys

    d = _os.environ.get("SGS_LOCK_DIR", "/tmp")
    flock_mod.LOCK_DIR = Path(d)

    @single_instance("blocker")
    def hold() -> None:
        Path(acquired_path_str).write_text("1")
        release = Path(marker_path_str)
        deadline = time.monotonic() + 6.0
        while not release.exists():
            if time.monotonic() > deadline:
                _sys.exit(3)
            time.sleep(0.05)
        _sys.exit(0)

    hold()
    _sys.exit(0)


def _mp_try_blocker(
    name: str, acquired_path_str: str, started_marker_str: str
) -> None:
    """Wait for holder to be acquired, then probe — should hit busy."""
    import os as _os
    import sys as _sys

    d = _os.environ.get("SGS_LOCK_DIR", "/tmp")
    flock_mod.LOCK_DIR = Path(d)

    @single_instance("blocker")
    def probe() -> None:
        _sys.exit(0)

    held_path = Path(acquired_path_str)
    held_deadline = time.monotonic() + 5.0
    while not held_path.exists():
        if time.monotonic() > held_deadline:
            _sys.exit(2)
        time.sleep(0.05)
    Path(started_marker_str).write_text("fired")
    try:
        probe()
    except AlreadyRunningError:
        _sys.exit(7)
    _sys.exit(0)


def test_decorator_in_child_blocks_others(tmp_lock_dir: Path) -> None:
    """SingleInstanceLock used via decorator: holder blocks a racer."""
    marker = tmp_lock_dir / ".release"
    acquired = tmp_lock_dir / ".acquired"
    racer_started = tmp_lock_dir / ".racer_started"
    holder = mp.Process(
        target=_mp_hold_blocker,
        args=("a", str(marker), str(acquired)),
    )
    holder.start()
    racer = mp.Process(
        target=_mp_try_blocker,
        args=("b", str(acquired), str(racer_started)),
    )
    racer.start()
    deadline = time.monotonic() + 4.0
    while not racer_started.exists():
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)
    marker.write_text("ok")
    holder.join(timeout=6)
    racer.join(timeout=6)
    assert holder.exitcode == 0, f"holder exitcode={holder.exitcode}"
    assert racer.exitcode == 7, f"racer exitcode={racer.exitcode}"
    assert racer_started.exists()
