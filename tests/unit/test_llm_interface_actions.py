import sys
from pathlib import Path
import asyncio
import types

import pytest

# Ensure project packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.llm.llm_interface import LLMInterface
from core.meta.action_parser import parse_action_from_text
from core.meta.action_executor import execute_get_time


class DummyAdapter:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, **options):
        self.calls.append(prompt)
        # If this is a continuation with internal tool result, return a reply that uses the tool result
        if "[INTERNAL] Tool result" in prompt:
            return "Kitsu: Got the tool result, the time is 2025-12-16T12:00:00Z"
        # Default: initial generation including an action token
        return "<action:get_time>\nKitsu: It's lunchtime.\n[thought]internal reasoning[/thought]"

    def stream(self, prompt: str, **options):
        # Yield two chunks combining into a response with an action token
        yield "<action:get_time>"
        yield "\nKitsu: It's noon."


def test_non_streaming_action_executes_and_sanitizes(monkeypatch):
    llm = LLMInterface(streaming=False)

    # Replace adapter with dummy
    dummy = DummyAdapter()
    llm.adapter = dummy

    # Ensure execute_get_time returns deterministic value by monkeypatching it
    monkeypatch.setattr("core.meta.action_executor.execute_get_time", lambda timezone=None, now=None: "2025-12-16T12:00:00Z")

    # Call generate_response; user_permissions allow all by default
    resp = asyncio.run(llm.generate_response("Hello", stream=False))

    # Response should not contain action tokens, thought blocks, or INTERNAL lines
    assert "<action:" not in resp
    assert "[thought]" not in resp
    assert "[INTERNAL]" not in resp

    # last_action_decision should indicate approval for get_time
    assert llm.last_action_decision is not None
    assert llm.last_action_decision.get("approved") is True


def test_requires_confirmation_search(monkeypatch):
    # Simulate a search where last_topic differs -> requires_confirmation True
    llm = LLMInterface(streaming=False)
    dummy = DummyAdapter()

    # Modify initial generation to include a search action
    def gen_search(prompt: str, **options):
        return "<action:search query=\"hockey scores\">\nKitsu: Checking..."

    dummy.generate = gen_search
    llm.adapter = dummy

    resp = asyncio.run(llm.generate_response("Hi", user_permissions={"allow_search": True}, last_topic="weather"))

    # Since topic differs, MetaController should set requires_confirmation and not execute
    assert llm.last_action_decision is not None
    assert llm.last_action_decision.get("requires_confirmation") is True
    # Response should be sanitized and still not contain action tokens
    assert "<action:" not in resp


def test_streaming_action_flow(monkeypatch):
    llm = LLMInterface(streaming=True)
    dummy = DummyAdapter()
    llm.adapter = dummy

    # Monkeypatch execute_get_time for deterministic result
    monkeypatch.setattr("core.meta.action_executor.execute_get_time", lambda timezone=None, now=None: "2025-12-16T12:00:00Z")

    resp = asyncio.run(llm.generate_response("Hey", stream=True))

    assert "<action:" not in resp
    assert "[thought]" not in resp
    assert "2025-12-16T12:00:00Z" in resp
    assert llm.last_action_decision is not None and llm.last_action_decision.get("approved")


def test_multiple_actions_are_rejected(monkeypatch):
    llm = LLMInterface(streaming=False)
    dummy = DummyAdapter()

    def gen_multi(prompt: str, **options):
        return "<action:search query=\"a\">\n<action:get_time>\nKitsu: messy"

    dummy.generate = gen_multi
    llm.adapter = dummy

    resp = asyncio.run(llm.generate_response("Hi"))
    # Multiple actions should result in parse error and no execution
    assert llm.last_action_decision is None or llm.last_action_decision.get("approved") is False
    assert "<action:" not in resp
