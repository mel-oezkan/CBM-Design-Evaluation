"""ClaudeClient caching against a stub transport (offline): complete responses are cached,
refused or cut-off ones raise and are not."""

from types import SimpleNamespace

import pytest

from cbm_eval.stages.discovery import LLMDiscovery
from cbm_eval.stages.llm import ClaudeClient


class StubTransport:
    def __init__(self, stop_reason="end_turn", text="- feathers\n- beak"):
        self.calls, self.stop_reason, self.text = [], stop_reason, text
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               content=[SimpleNamespace(type="text", text=self.text)])


def test_complete_responses_are_cached(tmp_path):
    stub = StubTransport()
    client = ClaudeClient(cache_dir=tmp_path, client=stub)
    assert client.complete("list bird parts") == stub.text
    assert client.complete("list bird parts") == stub.text
    assert len(stub.calls) == 1


@pytest.mark.parametrize("stop_reason, match", [("max_tokens", "raise `max_tokens`"), ("refusal", "refused")])
def test_cut_off_or_refused_responses_raise_and_are_not_cached(tmp_path, stop_reason, match):
    stub = StubTransport(stop_reason=stop_reason, text="- feathers\n- be")
    client = ClaudeClient(max_tokens=16, cache_dir=tmp_path, client=stub)
    with pytest.raises(RuntimeError, match=match):
        client.complete("list bird parts")
    assert not list(tmp_path.glob("*.json"))


def test_discovery_passes_max_tokens(tmp_path, monkeypatch):
    from cbm_eval.stages import discovery

    stub = StubTransport()
    monkeypatch.setattr(discovery, "ClaudeClient", lambda **kw: ClaudeClient(client=stub, **kw))
    ctx = SimpleNamespace(cache_dir=tmp_path, class_names=["sparrow"])
    LLMDiscovery(prompts=["features"], max_tokens=123).discover(ctx)
    assert stub.calls[0]["max_tokens"] == 123
