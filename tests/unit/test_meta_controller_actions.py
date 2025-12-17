import sys
from pathlib import Path
import datetime
import types

import pytest

# Ensure workspace package imports work during tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.meta.meta_controller import MetaController, ContinuationState
from core.meta.action_parser import Action
from core.meta import action_executor


@pytest.fixture
def controller():
    return MetaController(max_continuations=2)


def test_deny_search_without_permission(controller):
    action = Action(kind="search", payload={"query": "weather"}, raw="<action:search ...>")
    dec = controller.decide_and_handle_action(action, user_permissions={"allow_search": False})
    assert dec["approved"] is False
    assert "permission" in dec["reason"]


def test_require_confirmation_when_topic_differs(controller):
    action = Action(kind="search", payload={"query": "weather in hanoi"}, raw="<action:search ...>")
    dec = controller.decide_and_handle_action(action, user_permissions={"allow_search": True}, last_topic="sports")
    assert dec["approved"] is False
    assert dec["requires_confirmation"] is True


def test_deny_continue_when_limit_reached(controller):
    action = Action(kind="continue", payload={"reason": "more"}, raw="<action:continue ...>")
    state = ContinuationState(count=2)
    dec = controller.decide_and_handle_action(action, user_permissions={"allow_continue": True}, continuation_state=state)
    assert dec["approved"] is False
    assert "limit" in dec["reason"]


def test_deny_continue_when_emotion_high(controller):
    action = Action(kind="continue", payload={"reason": "more"}, raw="<action:continue ...>")
    state = ContinuationState(count=0)
    emotion_state = {"intensity": 0.9}
    dec = controller.decide_and_handle_action(action, user_permissions={"allow_continue": True}, continuation_state=state, emotion_state=emotion_state)
    assert dec["approved"] is False
    assert "emotion" in dec["reason"]


def test_execute_search_stub():
    res = action_executor.execute_search("test query", max_results=3)
    assert res["query"] == "test query"
    assert len(res["results"]) == 3
    assert res["results"][0]["title"].startswith("Result 1")


def test_execute_get_time_with_injected_now():
    now = datetime.datetime(2025, 12, 16, 12, 0, 0)
    s = action_executor.execute_get_time(None, now=now)
    assert s == "2025-12-16T12:00:00Z"


def test_handle_continue_respects_max_and_increments():
    out = action_executor.handle_continue("reason", continuation_count=0, max_allowed=2)
    assert out["allowed"] is True
    assert out["new_count"] == 1

    out2 = action_executor.handle_continue("reason", continuation_count=2, max_allowed=2)
    assert out2["allowed"] is False
    assert out2["new_count"] == 2
