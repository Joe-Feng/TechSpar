import json
import logging

import requests
from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike

from backend.config import settings

_embedding_instance = None
_llama_llm_instance = None
logger = logging.getLogger("uvicorn")


def _message_to_payload(message) -> dict:
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    else:
        role = getattr(message, "type", None) or getattr(message, "role", "user")
        content = getattr(message, "content", "")

    role = {"human": "user", "ai": "assistant"}.get(role, role)

    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    return {"role": role, "content": content}


def compat_chat_completion(
    messages: list,
    *,
    model: str | None = None,
    temperature: float | None = None,
    timeout: int = 120,
) -> str:
    """Call the configured OpenAI-compatible endpoint via raw HTTP.

    Some gateways reject the official OpenAI SDK request shape while accepting
    plain JSON HTTP requests. Resume-mode flows use this helper as a
    compatibility path.
    """
    base_url = (settings.api_base or "").rstrip("/")
    if not base_url or not settings.api_key or not (model or settings.model):
        raise RuntimeError("LLM provider is not fully configured.")

    payload_messages = [
        payload
        for payload in (_message_to_payload(message) for message in messages)
        if payload["content"].strip()
    ]
    if not payload_messages:
        raise RuntimeError("LLM request contained no non-empty messages.")

    payload = {
        "model": model or settings.model,
        "messages": payload_messages,
        "temperature": settings.temperature if temperature is None else temperature,
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text.strip() or f"HTTP {resp.status_code}"
        logger.warning("Compat LLM request failed: status=%s body=%s", resp.status_code, detail[:500])
        raise RuntimeError(f"LLM request failed: {detail}")

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response contained no choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def get_langchain_llm():
    """LangChain ChatModel for LangGraph nodes (via OpenAI-compatible proxy)."""
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.api_base,
        temperature=settings.temperature,
        streaming=True,
    )


def get_copilot_llm(streaming: bool = False):
    """Copilot 专用 LLM，fallback 到主 LLM。"""
    return ChatOpenAI(
        model=settings.copilot_model or settings.model,
        api_key=settings.copilot_api_key or settings.api_key,
        base_url=settings.copilot_api_base or settings.api_base,
        temperature=settings.copilot_temperature,
        streaming=streaming,
    )


def get_llama_llm():
    """LlamaIndex LLM (singleton)."""
    global _llama_llm_instance
    if _llama_llm_instance is None:
        _llama_llm_instance = OpenAILike(
            model=settings.model,
            api_key=settings.api_key,
            api_base=settings.api_base,
            temperature=settings.temperature,
            is_chat_model=True,
        )
    return _llama_llm_instance


def get_embedding():
    """Embedding model (singleton)."""
    global _embedding_instance
    if _embedding_instance is None:
        if settings.embedding_backend_mode() == "api":
            from llama_index.embeddings.openai import OpenAIEmbedding

            model_name = settings.embedding_api_model_name()
            if not model_name:
                raise RuntimeError("EMBEDDING_API_MODEL is required when EMBEDDING_BACKEND=api")

            kwargs = {
                "model_name": model_name,
                "api_key": settings.embedding_api_key,
            }
            if settings.embedding_api_base:
                kwargs["api_base"] = settings.embedding_api_base

            _embedding_instance = OpenAIEmbedding(**kwargs)
        else:
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings require optional dependencies. "
                    "Install `pip install -r requirements.local-embedding.txt` "
                    "and a torch build that matches your environment."
                ) from exc

            model_path = settings.local_embedding_model_path()
            model_name = settings.local_embedding_model_name()

            if model_path is not None:
                _embedding_instance = HuggingFaceEmbedding(model_name=str(model_path))
            elif model_name:
                _embedding_instance = HuggingFaceEmbedding(model_name=model_name)
            else:
                raise RuntimeError(
                    "LOCAL_EMBEDDING_MODEL or LOCAL_EMBEDDING_PATH is required "
                    "when EMBEDDING_BACKEND=local"
                )
    return _embedding_instance


def _reset_llama_singleton():
    """Reset LlamaIndex LLM singleton so next call picks up new settings."""
    global _llama_llm_instance
    _llama_llm_instance = None
