"""Tests for :mod:`sgs.wire.base` — response parsing + endpoint shape.

Reference cases: case-5 (shareId 376634286041, three phases), the cleanest
shareId in the case study.
"""

from __future__ import annotations

import pytest

from sgs.oracle import OracleResponse
from sgs.wire.base import (
    DEFAULT_HEADERS,
    GUESS_PATH,
    WireEndpoint,
    parse_response,
)


# ----- happy path (case-5 phase-1: 忍者, score=0.989, double=true) -----

def test_parse_response_happy_path_returns_score_and_correct() -> None:
    raw = {"code": 0, "msg": "ok",
           "data": {"score": 0.989, "doubleScore": True, "correct": True}}
    r = parse_response(raw, "忍者")
    assert r.word == "忍者"
    assert r.score == 0.989
    assert r.correct is True
    assert r.double_score is True
    assert r.rate_limited is False


def test_parse_response_happy_path_no_double() -> None:
    raw = {"code": 0, "msg": "ok",
           "data": {"score": 0.5, "correct": False, "doubleScore": False}}
    r = parse_response(raw, "剑客")
    assert r.score == 0.5
    assert r.correct is False
    assert r.double_score is False


# ----- server-side lock (data: null) ---------------------------------------

def test_parse_response_data_null_returns_none_score() -> None:
    raw = {"code": 0, "msg": "ok", "data": None}
    r = parse_response(raw, "萧山")
    assert r.score is None
    assert r.correct is False  # locked, NOT correct
    assert r.rate_limited is False
    # Caller is responsible for tracking these in the lock list.


# ----- business error / rate-limit -----------------------------------------

def test_parse_response_business_error_returns_none_score() -> None:
    raw = {"code": 500, "msg": "internal error", "data": None}
    r = parse_response(raw, "上海")
    assert r.score is None
    assert r.correct is False
    assert r.rate_limited is False  # not a rate-limit signal


def test_parse_response_code_one_treated_as_rate_limit() -> None:
    raw = {"code": 1, "msg": "rate limit exceeded", "data": None}
    r = parse_response(raw, "浙江")
    assert r.score is None
    assert r.rate_limited is True


def test_parse_response_msg_with_chinese_limit_keyword_is_rate_limit() -> None:
    raw = {"code": 999, "msg": "频率超限，请稍后重试", "data": None}
    r = parse_response(raw, "江苏")
    assert r.score is None
    assert r.rate_limited is True


# ----- caller-side bugs ----------------------------------------------------

def test_parse_response_missing_data_field_is_value_error() -> None:
    with pytest.raises(ValueError, match="required"):
        parse_response({}, "上海")


def test_parse_response_data_not_dict_when_present_is_value_error() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        parse_response({"code": 0, "data": "string"}, "上海")


def test_parse_response_score_out_of_range_raises() -> None:
    raw = {"code": 0, "msg": "ok",
           "data": {"score": 1.5, "correct": False}}
    with pytest.raises(ValueError, match="out of"):
        parse_response(raw, "上海")


def test_parse_response_nan_score_raises() -> None:
    raw = {"code": 0, "msg": "ok",
           "data": {"score": float("nan"), "correct": False}}
    with pytest.raises(ValueError, match="out of"):
        parse_response(raw, "上海")


# ----- endpoint / header sanity -------------------------------------------

def test_endpoint_guess_url_contains_share_id_and_flag() -> None:
    e = WireEndpoint("376634286041")
    url = e.guess_url("忍者")
    assert "shareId=376634286041" in url
    assert "skipBusinessErrorToast=true" in url
    assert "word=%E5%BF%8D%E8%80%85" in url  # 忍者 URL-encoded
    assert "date=" not in url  # v0.8.0: no date when only shareId


def test_endpoint_guess_path_matches_canonical() -> None:
    assert GUESS_PATH == "/api/v0/quiz/daily/GuessWord/guessV1"


def test_default_headers_include_fun_device() -> None:
    assert "fun-device" in DEFAULT_HEADERS
    assert DEFAULT_HEADERS["fun-device"] == "web"


# ----- v0.8.0 daily-mode (date-based, visitor-accessible) ----------------


def test_endpoint_date_only_includes_date_param() -> None:
    """date=YYYYMMDD lets unauthenticated visitors probe today's daily."""
    e = WireEndpoint(date="20260715")
    url = e.guess_url("南宁")
    assert "date=20260715" in url
    assert "word=%E5%8D%97%E5%AE%81" in url  # 南宁 URL-encoded
    assert "shareId=" not in url


def test_endpoint_with_both_shareid_and_date_includes_both() -> None:
    """shareId wins as the primary key when both are set (login user
    playing their own share + date context for fallback resolution)."""
    e = WireEndpoint(share_id="123", date="20260715")
    url = e.guess_url("南宁")
    assert "shareId=123" in url
    assert "date=20260715" in url


def test_endpoint_requires_at_least_one_of_shareid_date() -> None:
    with pytest.raises(ValueError, match="exactly one of share_id / date"):
        WireEndpoint()


def test_endpoint_accepts_positional_shareid() -> None:
    """Backwards compat: positional share_id still works."""
    e = WireEndpoint("12345")
    assert e.share_id == "12345"
    assert e.date is None
    assert "shareId=12345" in e.guess_url("学校")