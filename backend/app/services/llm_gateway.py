"""
LLM Gateway — abstracción de proveedores LLM.

vLLM (default) expone una API OpenAI-compatible, por lo que usamos el SDK
`openai` con `base_url` apuntando al servidor vLLM. También soporta OpenAI,
Ollama (OpenAI-compatible) y Anthropic (SDK nativo).

Registra uso (tokens) y coste por tenant vía UsageCollector + CostCalculator.
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMGateway:
    """Gateway de LLM multi-provider con tracking de tokens y costes."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.model_name = self._resolve_model()

    # ──────────────────────────────────────────────────────────────────────
    # Clientes
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_model(self) -> str:
        if self.provider == "vllm":
            return settings.VLLM_MODEL_NAME
        if self.provider == "ollama":
            return settings.LLM_MODEL_NAME or "llama3"
        return settings.LLM_MODEL_NAME

    def get_client(self):
        """Devuelve el cliente asíncrono según el provider configurado."""
        if self.provider in ("vllm", "openai"):
            import openai

            base_url = settings.VLLM_BASE_URL if self.provider == "vllm" else None
            return openai.AsyncOpenAI(
                base_url=base_url, api_key=settings.LLM_API_KEY
            )
        if self.provider == "ollama":
            import openai

            return openai.AsyncOpenAI(
                base_url=settings.OLLAMA_BASE_URL, api_key="EMPTY"
            )
        if self.provider == "anthropic":
            import anthropic

            return anthropic.AsyncAnthropic(api_key=settings.LLM_API_KEY)
        raise ValueError(f"Provider LLM no soportado: {self.provider}")

    # ──────────────────────────────────────────────────────────────────────
    # Completions
    # ──────────────────────────────────────────────────────────────────────

    def _messages(self, messages: List[Dict[str, str]], system: Optional[str]):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        return msgs

    async def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Completión no-stream. Devuelve {content, usage}."""
        t0 = time.perf_counter()
        msgs = self._messages(messages, system)
        client = self.get_client()
        usage = {"input": 0, "output": 0}

        if self.provider == "anthropic":
            resp = await client.messages.create(
                model=self.model_name,
                system=system or "",
                messages=[m for m in messages if m["role"] != "system"],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            )
            usage = {
                "input": resp.usage.input_tokens,
                "output": resp.usage.output_tokens,
            }
        else:
            resp = await client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            usage = {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    async def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Completión en streaming. Emite {"delta": str} y al final {"usage": {...}}."""
        msgs = self._messages(messages, system)
        client = self.get_client()

        if self.provider == "anthropic":
            async with client.messages.stream(
                model=self.model_name,
                system=system or "",
                messages=[m for m in messages if m["role"] != "system"],
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
                async for text in stream.text_stream:
                    yield {"delta": text}
            final = await stream.get_final_message()
            yield {
                "usage": {
                    "input": final.usage.input_tokens,
                    "output": final.usage.output_tokens,
                }
            }
            return

        stream = await client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        usage = {"input": 0, "output": 0}
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield {"delta": chunk.choices[0].delta.content}
            if getattr(chunk, "usage", None):
                usage = {
                    "input": chunk.usage.prompt_tokens,
                    "output": chunk.usage.completion_tokens,
                }
        yield {"usage": usage}

    # ──────────────────────────────────────────────────────────────────────
    # Observabilidad y costes
    # ──────────────────────────────────────────────────────────────────────

    def record_usage(
        self,
        db,
        tenant_id: str,
        usage: Dict[str, int],
        latency_ms: float,
        model_id: Optional[str] = None,
        status: str = "success",
        tags: Optional[dict] = None,
    ) -> None:
        """Registra uso y coste de una llamada LLM (dominio llm)."""
        from datetime import datetime, timezone

        from app.observability.cost_calculator import get_cost_calculator
        from app.observability.metrics import (
            llm_latency_ms,
            llm_requests_total,
            llm_tokens_total,
        )
        from app.observability.usage_collector import get_usage_collector

        tokens_in = usage.get("input", 0)
        tokens_out = usage.get("output", 0)
        tags = tags or {}

        get_usage_collector().record_inference(
            db=db,
            tenant_id=tenant_id,
            model_id=model_id,
            domain="llm",
            latency_ms=latency_ms,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            tags={"provider": self.provider, "model": self.model_name, **tags},
        )

        llm_tokens_total.labels(
            tenant_id=tenant_id, model_id=model_id or "none",
            provider=self.provider, direction="input",
        ).inc(tokens_in)
        llm_tokens_total.labels(
            tenant_id=tenant_id, model_id=model_id or "none",
            provider=self.provider, direction="output",
        ).inc(tokens_out)
        llm_requests_total.labels(
            tenant_id=tenant_id, model_id=model_id or "none",
            provider=self.provider, status=status,
        ).inc()
        llm_latency_ms.labels(
            tenant_id=tenant_id, model_id=model_id or "none", provider=self.provider
        ).observe(latency_ms)

        try:
            calculator = get_cost_calculator()
            cost = calculator.estimate_inference_cost(
                db=db,
                domain="llm",
                latency_ms=latency_ms,
                tokens={
                    "model": self.model_name,
                    "input": tokens_in,
                    "output": tokens_out,
                },
            )
            if settings.LLM_CHARGE_COMPUTE:
                cost += calculator.estimate_inference_cost(
                    db=db, domain="nlp", latency_ms=latency_ms,
                    gpu_type=settings.LLM_GPU_TYPE,
                )
            if cost > 0:
                now = datetime.now(timezone.utc)
                calculator.record_cost(
                    db=db, tenant_id=tenant_id, cost_type="llm", amount=cost,
                    source=f"{self.provider}:{self.model_name}",
                    period_start=now, period_end=now, model_id=model_id,
                    metadata={
                        "provider": self.provider,
                        "model": self.model_name,
                        "tokens_input": tokens_in,
                        "tokens_output": tokens_out,
                        "latency_ms": latency_ms,
                    },
                )
        except Exception as exc:
            logger.warning("LLM cost recording failed (non-critical): %s", exc)


_llm_gateway_instance: Optional[LLMGateway] = None


def get_llm_gateway(provider: Optional[str] = None) -> LLMGateway:
    """Factory singleton del LLMGateway."""
    global _llm_gateway_instance
    if _llm_gateway_instance is None or provider:
        return LLMGateway(provider=provider)
    return _llm_gateway_instance
