"""Unit tests for the live integration skip-guard.

These tests do NOT need live credentials — they verify that the
opt-in infrastructure works correctly when the credentials are
absent (the common case in CI).

Located in `tests/integration/` to keep live-related code together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_skip_guard_module_importable():
    """`tests.integration.test_oracle_live` must be importable
    even without a Playwright install."""
    import tests.integration.test_oracle_live as live_mod

    # The module exposes a `LIVE_WORD_POOL` constant that is the
    # source of truth for "guaranteed-loser" words.
    assert hasattr(live_mod, "LIVE_WORD_POOL")
    assert len(live_mod.LIVE_WORD_POOL) >= 2
    # Every candidate is a str.
    assert all(isinstance(w, str) for w in live_mod.LIVE_WORD_POOL)


def test_skip_guard_helper_returns_false_when_no_persistent_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With no `.playwright-data` directory, `_has_live_credentials()`
    must return False — meaning every live test will SKIP, never FAIL.
    """
    from tests.integration.test_oracle_live import _has_live_credentials

    # Force CWD to an empty dir so the real `.playwright-data` is invisible.
    monkeypatch.chdir(tmp_path)
    assert _has_live_credentials() is False


def test_skip_guard_helper_returns_true_when_cookies_db_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With a fake `.playwright-data/Default/Cookies` file present,
    the guard must return True (we have *some* credentials, even if
    they turn out to be expired)."""
    from tests.integration.test_oracle_live import _has_live_credentials

    monkeypatch.chdir(tmp_path)
    cookies = tmp_path / ".playwright-data" / "Default" / "Cookies"
    cookies.parent.mkdir(parents=True)
    cookies.write_bytes(b"fake cookie jar bytes")
    assert _has_live_credentials() is True
