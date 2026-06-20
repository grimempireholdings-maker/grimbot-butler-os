from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def deterministic_conversation_provider(monkeypatch):
    """Keep the test suite independent of developer API-key configuration."""
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "mock")
