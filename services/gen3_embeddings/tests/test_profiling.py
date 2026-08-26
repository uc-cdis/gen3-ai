"""Tests that common.config reaches the continuous profiling agent."""

from collections.abc import Iterator
from typing import Any

import pytest
from cdispyutils.observability import continuous_profiling

from common import config
from common.profiling import configure_profiling

SERVICE_NAME = "gen3_embeddings"


class FakeAgent:
    """Stands in for the Pyroscope SDK, so no agent starts and nothing is pushed."""

    def __init__(self) -> None:
        """Record no calls yet."""
        self.configure_calls: list[dict[str, Any]] = []

    def configure(self, **kwargs: Any) -> None:
        """
        Record the keyword arguments the agent would have been started with.

        Args:
            **kwargs: Whatever `configure_profiling` passes through.
        """
        self.configure_calls.append(kwargs)

    def shutdown(self) -> None:
        """Accept the request to stop."""


@pytest.fixture(autouse=True)
def agent(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAgent]:
    """
    Return the fake agent standing in for the SDK, and reset the module afterwards.

    Yields:
        FakeAgent: The recorder installed in place of the `pyroscope` module.
    """
    fake = FakeAgent()
    monkeypatch.setattr(continuous_profiling, "pyroscope", fake)

    yield fake

    # The SDK holds one agent per process, so a test that leaves the module thinking it started
    # one would make every later test a no-op.
    continuous_profiling.stop_profiling()


@pytest.fixture
def profiling_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn profiling on for the duration of one test, since it is off by default."""
    monkeypatch.setattr(config, "ENABLE_CONTINUOUS_PROFILING", True)


def test_profiling_is_off_by_default(agent: FakeAgent) -> None:
    """With no configuration, no agent is started."""
    configure_profiling(SERVICE_NAME)

    assert agent.configure_calls == []
    assert not continuous_profiling.profiling_active()


def test_enabling_profiling_starts_the_agent(agent: FakeAgent, profiling_on: None) -> None:
    """Enabling profiling starts an agent named for the service."""
    configure_profiling(SERVICE_NAME)

    assert continuous_profiling.profiling_active()
    assert agent.configure_calls[0]["application_name"] == SERVICE_NAME


def test_configured_server_and_rates_reach_the_agent(agent: FakeAgent, profiling_on: None) -> None:
    """The address and timings the agent runs with are the ones common.config resolved."""
    configure_profiling(SERVICE_NAME)

    started = agent.configure_calls[0]

    assert started["server_address"] == config.PYROSCOPE_SERVER_ADDRESS
    assert started["sample_rate"] == config.PYROSCOPE_SAMPLE_RATE
    assert started["upload_interval"] == config.PYROSCOPE_UPLOAD_INTERVAL


def test_cpu_profiling_is_on_and_memory_profiling_is_off(agent: FakeAgent, profiling_on: None) -> None:
    """The default profile selection is CPU only."""
    configure_profiling(SERVICE_NAME)

    assert agent.configure_calls[0]["cpu_enabled"] is True
    assert agent.configure_calls[0]["mem_enabled"] is False


def test_memory_profiling_can_be_turned_on(
    agent: FakeAgent, profiling_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROFILE_MEMORY adds allocation profiling to the agent."""
    monkeypatch.setattr(config, "PROFILE_MEMORY", True)

    configure_profiling(SERVICE_NAME)

    assert agent.configure_calls[0]["mem_enabled"] is True


def test_wall_clock_profiling_can_be_selected(
    agent: FakeAgent, profiling_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROFILE_ON_CPU_ONLY off asks the agent for wall-clock rather than CPU time."""
    monkeypatch.setattr(config, "PROFILE_ON_CPU_ONLY", False)

    configure_profiling(SERVICE_NAME)

    assert agent.configure_calls[0]["oncpu"] is False
