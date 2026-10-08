"""Typed configuration helpers for the Streamlit chat providers."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


class ChatProvider(str, Enum):
    """Supported chat API provider modes."""

    OPENAI = "OpenAI"
    OPENAI_COMPATIBLE = "OpenAI-compatible"


DEFAULT_OPENAI_CHAT_MODEL = "gpt-3.5-turbo"
DEFAULT_COMPATIBLE_CHAT_MODEL = "llama3.2"
CUSTOM_MODEL_OPTION = "Enter a custom model ID"
OPENAI_CHAT_MODELS = (
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
)
COMPATIBLE_CHAT_MODELS = (
    "llama3.2",
    "mistral",
    "qwen2.5",
)


@dataclass(frozen=True)
class ChatDefaults:
    """Environment-derived defaults for chat and retrieval configuration."""

    provider: ChatProvider
    openai_api_key: str
    compatible_api_key: str
    chat_base_url: str | None
    chat_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None


@dataclass(frozen=True)
class ChatSettings:
    """Validated settings for one chat-completion request."""

    api_key: str
    base_url: str | None
    model: str


@dataclass(frozen=True)
class EmbeddingSettings:
    """Validated settings for retrieval query embeddings."""

    api_key: str | None
    base_url: str | None


def _optional_value(value: str | None) -> str | None:
    """Normalize an optional environment or UI value."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validate_base_url(base_url: str | None) -> str | None:
    """Return a normalized optional HTTP(S) URL or reject invalid input."""
    normalized = _optional_value(base_url)
    if normalized is None:
        return None

    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Base URL must be an absolute http or https URL")
    return normalized


def resolve_chat_defaults(environ: Mapping[str, str]) -> ChatDefaults:
    """Resolve provider defaults from an environment mapping."""
    ollama_base_url = _optional_value(environ.get("OLLAMA_OPENAI_BASE_URL"))
    openai_base_url = _optional_value(environ.get("OPENAI_BASE_URL"))
    custom_base_url = ollama_base_url or openai_base_url
    openai_api_key = _optional_value(environ.get("OPENAI_API_KEY")) or ""
    ollama_api_key = _optional_value(environ.get("OLLAMA_API_KEY"))

    if ollama_base_url:
        provider = ChatProvider.OPENAI_COMPATIBLE
        compatible_api_key = ollama_api_key or "ollama"
        chat_model = (
            _optional_value(environ.get("OLLAMA_CHAT_MODEL"))
            or DEFAULT_COMPATIBLE_CHAT_MODEL
        )
    elif openai_base_url:
        provider = ChatProvider.OPENAI_COMPATIBLE
        compatible_api_key = (
            openai_api_key or ollama_api_key or "ollama"
        )
        chat_model = (
            _optional_value(environ.get("OPENAI_CHAT_MODEL"))
            or DEFAULT_COMPATIBLE_CHAT_MODEL
        )
    else:
        provider = ChatProvider.OPENAI
        compatible_api_key = ollama_api_key or "ollama"
        chat_model = (
            _optional_value(environ.get("OPENAI_CHAT_MODEL"))
            or DEFAULT_OPENAI_CHAT_MODEL
        )

    return ChatDefaults(
        provider=provider,
        openai_api_key=openai_api_key,
        compatible_api_key=compatible_api_key,
        chat_base_url=custom_base_url,
        chat_model=chat_model,
        embedding_api_key=(
            _optional_value(environ.get("CHROMA_OPENAI_API_KEY"))
            or _optional_value(environ.get("OPENAI_API_KEY"))
        ),
        embedding_base_url=_optional_value(
            environ.get("CHROMA_OPENAI_BASE_URL")
        ),
    )


def load_chat_defaults(dotenv_path: str | Path | None = None) -> ChatDefaults:
    """Load `.env` defaults without overriding process-level environment values."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return resolve_chat_defaults(os.environ)


def build_chat_settings(
    provider: ChatProvider,
    api_key: str,
    base_url: str,
    model: str,
) -> ChatSettings:
    """Validate chat settings selected in the Streamlit sidebar."""
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("Chat model must not be empty")

    if provider is ChatProvider.OPENAI:
        normalized_api_key = api_key.strip()
        if not normalized_api_key:
            raise ValueError("An API key is required for OpenAI")
        return ChatSettings(
            api_key=normalized_api_key,
            base_url=None,
            model=normalized_model,
        )

    normalized_base_url = _validate_base_url(base_url)
    if normalized_base_url is None:
        raise ValueError("A base URL is required for an OpenAI-compatible provider")

    return ChatSettings(
        api_key=api_key.strip() or "ollama",
        base_url=normalized_base_url,
        model=normalized_model,
    )


def build_embedding_settings(
    api_key: str,
    base_url: str,
) -> EmbeddingSettings:
    """Validate the independently configured retrieval embedding endpoint."""
    normalized_base_url = _validate_base_url(base_url)
    return EmbeddingSettings(
        api_key=_optional_value(api_key)
        or ("ollama" if normalized_base_url is not None else None),
        base_url=normalized_base_url,
    )
