"""
Shared test setup: the suite runs fully offline.

- A dummy API key is set before config.py loads project/.env (python-dotenv
  never overrides variables that already exist), so the real key is unused.
- Every real OpenAI call is blocked with an exception the agents' broad
  `except Exception` handlers cannot swallow, so an accidental LLM call fails
  the test loudly instead of silently taking an error fallback path.
- Tests that need LLM output stub it (fake client or patched respond).
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ["OPENAI_API_KEY"] = "sk-offline-test-key"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RealLLMCallError(BaseException):
    """Raised when a test reaches the real OpenAI API (BaseException on purpose)."""


@pytest.fixture(autouse=True)
def block_real_llm_calls(monkeypatch):
    """Fail any test that tries to reach the real OpenAI chat completions API."""
    def blocked(*args, **kwargs):
        raise RealLLMCallError("Offline test attempted a real OpenAI call")
    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", blocked)


class FakeOpenAIClient:
    """
    Minimal stand-in for openai.OpenAI used to feed canned JSON responses.

    `reply` is either a dict (returned for every call) or a callable that
    receives the create() kwargs and returns a dict. Every call's kwargs are
    recorded in `calls`.
    """

    def __init__(self, reply):
        """Store the canned reply and set up the chat.completions namespace."""
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        """Record the call and return a response shaped like the OpenAI SDK's."""
        self.calls.append(kwargs)
        payload = self.reply(kwargs) if callable(self.reply) else self.reply
        message = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_client():
    """Factory fixture: fake_client(reply) -> FakeOpenAIClient."""
    return FakeOpenAIClient


@pytest.fixture
def valid_message():
    """A well-formed pipeline message (dict format used by every pattern)."""
    return {
        "message_id": "MSG-1",
        "message_type": "MT103",
        "reference": "REF123456",
        "amount": "2500.00 USD",
        "sender_bic": "CHASUS33XXX",
        "receiver_bic": "BARCGB22XXX",
        "value_date": "251121",
        "remittance_info": "Invoice payment",
    }
