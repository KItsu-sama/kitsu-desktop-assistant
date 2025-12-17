import sys
from pathlib import Path
import pytest

# Ensure workspace package imports work during tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.meta.action_parser import parse_action_from_text, Action


def test_parse_search_success():
    txt = """I'll answer.

<action:search query=\"best ramen near me\">"""
    res = parse_action_from_text(txt)
    assert res["ok"] is True
    action = res["action"]
    assert isinstance(action, Action)
    assert action.kind == "search"
    assert action.payload["query"] == "best ramen near me"


def test_parse_get_time_success():
    txt = "<action:get_time>\nKitsu: It's lunchtime."
    res = parse_action_from_text(txt)
    assert res["ok"] is True
    action = res["action"]
    assert action.kind == "get_time"


def test_parse_continue_with_reason():
    txt = "<action:continue reason=\"more details\">"
    res = parse_action_from_text(txt)
    assert res["ok"] is True
    action = res["action"]
    assert action.kind == "continue"
    assert action.payload["reason"] == "more details"


def test_multiple_actions_rejected():
    txt = """<action:search query=\"foo\">\n<action:get_time>"""
    res = parse_action_from_text(txt)
    assert res["ok"] is False
    assert "multiple" in res["error"]


def test_malformed_token_missing_quote():
    txt = "<action:search query=\"missing>"
    res = parse_action_from_text(txt)
    assert res["ok"] is False
    assert "search requires query" in res["error"] or "unsupported" in res["error"]


def test_unsupported_action_rejected():
    txt = "<action:fly>"
    res = parse_action_from_text(txt)
    assert res["ok"] is False
    assert "unsupported" in res["error"]


def test_inline_token_ignored():
    txt = "Please check <action:search query=\"foo\"> inline"
    res = parse_action_from_text(txt)
    assert res["ok"] is True
    assert res["action"] is None


def test_query_too_long_rejected():
    long_q = "a" * 300
    txt = f"<action:search query=\"{long_q}\">"
    res = parse_action_from_text(txt)
    assert res["ok"] is False
    assert "too long" in res["error"]
