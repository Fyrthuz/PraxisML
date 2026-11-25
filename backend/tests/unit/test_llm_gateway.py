"""
Tests del LLMGateway (cliente openai mockeado, provider vllm/openai/ollama).
"""

from types import SimpleNamespace

import pytest

from app.domains.nlp.llm_gateway import LLMGateway


class FakeCompletions:
    async def create(self, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Ho"))],
                    usage=None,
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="la"))],
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
                )

            return _gen()

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hola desde vLLM"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeChat:
    completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = FakeChat()


@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setattr(
        "openai.AsyncOpenAI", lambda *a, **k: FakeOpenAI()
    )
    monkeypatch.setattr(
        "app.domains.nlp.llm_gateway.settings.VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"
    )
    monkeypatch.setattr(
        "app.domains.nlp.llm_gateway.settings.LLM_MODEL_NAME", "llama3"
    )
    monkeypatch.setattr(
        "app.domains.nlp.llm_gateway.settings.LLM_PROVIDER", "vllm"
    )


@pytest.mark.asyncio
async def test_complete_vllm(mock_openai):
    gateway = LLMGateway(provider="vllm")
    result = await gateway.complete(
        [{"role": "user", "content": "Hola"}], system="Eres útil"
    )
    assert result["content"] == "Hola desde vLLM"
    assert result["usage"]["input"] == 10
    assert result["usage"]["output"] == 5


@pytest.mark.asyncio
async def test_stream_vllm(mock_openai):
    gateway = LLMGateway(provider="vllm")
    chunks = []
    usage = None
    async for chunk in gateway.stream([{"role": "user", "content": "Hola"}]):
        if "delta" in chunk:
            chunks.append(chunk["delta"])
        else:
            usage = chunk["usage"]
    assert chunks == ["Ho", "la"]
    assert usage == {"input": 7, "output": 3}


@pytest.mark.asyncio
async def test_factory_by_provider(mock_openai):
    gateway = LLMGateway(provider="ollama")
    assert gateway.provider == "ollama"
    # Ollama también usa cliente OpenAI-compatible
    result = await gateway.complete([{"role": "user", "content": "Hola"}])
    assert result["content"]


def test_unsupported_provider_raises():
    gateway = LLMGateway(provider="noexiste")
    with pytest.raises(ValueError):
        gateway.get_client()


@pytest.mark.asyncio
async def test_record_usage(mock_openai, monkeypatch):
    from unittest.mock import MagicMock

    fake_collector = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.estimate_inference_cost.return_value = 0.0

    monkeypatch.setattr(
        "app.observability.usage_collector.get_usage_collector", lambda: fake_collector
    )
    monkeypatch.setattr(
        "app.observability.cost_calculator.get_cost_calculator", lambda: fake_calculator
    )

    gateway = LLMGateway(provider="vllm")
    db = MagicMock()
    gateway.record_usage(
        db=db, tenant_id="t1", usage={"input": 100, "output": 50},
        latency_ms=200.0, model_id="m1",
    )
    assert fake_collector.record_inference.called
    call = fake_collector.record_inference.call_args.kwargs
    assert call["domain"] == "llm"
    assert call["tokens_input"] == 100
    assert call["tokens_output"] == 50
