from pathlib import Path

import pytest

from src.chat_config import (
    ChatProvider,
    build_chat_settings,
    build_embedding_settings,
    load_chat_defaults,
    resolve_chat_defaults,
)


def test_resolves_native_openai_defaults() -> None:
    defaults = resolve_chat_defaults(
        {
            "OPENAI_API_KEY": "openai-key",
        }
    )

    assert defaults.provider is ChatProvider.OPENAI
    assert defaults.openai_api_key == "openai-key"
    assert defaults.compatible_api_key == "ollama"
    assert defaults.chat_base_url is None
    assert defaults.chat_model == "gpt-3.5-turbo"
    assert defaults.embedding_api_key == "openai-key"
    assert defaults.embedding_base_url is None


def test_resolves_custom_provider_and_separate_embedding_settings() -> None:
    defaults = resolve_chat_defaults(
        {
            "OLLAMA_OPENAI_BASE_URL": "http://localhost:11434/v1",
            "OLLAMA_CHAT_MODEL": "llama3.2",
            "OPENAI_API_KEY": "openai-key",
            "CHROMA_OPENAI_BASE_URL": "http://localhost:11434/v1",
            "CHROMA_OPENAI_API_KEY": "embedding-placeholder",
        }
    )

    assert defaults.provider is ChatProvider.OPENAI_COMPATIBLE
    assert defaults.chat_base_url == "http://localhost:11434/v1"
    assert defaults.compatible_api_key == "ollama"
    assert defaults.openai_api_key == "openai-key"
    assert defaults.chat_model == "llama3.2"
    assert defaults.embedding_base_url == "http://localhost:11434/v1"
    assert defaults.embedding_api_key == "embedding-placeholder"


def test_generic_custom_url_uses_openai_key_and_model_defaults() -> None:
    defaults = resolve_chat_defaults(
        {
            "OPENAI_BASE_URL": "https://gateway.example/v1",
            "OPENAI_API_KEY": "gateway-key",
            "OPENAI_CHAT_MODEL": "gateway-model",
        }
    )

    assert defaults.provider is ChatProvider.OPENAI_COMPATIBLE
    assert defaults.openai_api_key == "gateway-key"
    assert defaults.compatible_api_key == "gateway-key"
    assert defaults.chat_base_url == "https://gateway.example/v1"
    assert defaults.chat_model == "gateway-model"
    assert defaults.embedding_api_key == "gateway-key"
    assert defaults.embedding_base_url is None


def test_process_environment_overrides_dotenv_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENAI_API_KEY=dotenv-key\nOPENAI_CHAT_MODEL=gpt-4o-mini\n",
        encoding="utf-8",
    )
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "OLLAMA_API_KEY",
        "OLLAMA_OPENAI_BASE_URL",
        "OLLAMA_CHAT_MODEL",
        "CHROMA_OPENAI_API_KEY",
        "CHROMA_OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    defaults_from_dotenv = load_chat_defaults(dotenv_path)
    assert defaults_from_dotenv.openai_api_key == "dotenv-key"
    assert defaults_from_dotenv.chat_model == "gpt-4o-mini"

    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "process-model")
    defaults_from_process = load_chat_defaults(dotenv_path)

    assert defaults_from_process.openai_api_key == "process-key"
    assert defaults_from_process.chat_model == "process-model"


def test_custom_chat_settings_require_http_url_and_model() -> None:
    settings = build_chat_settings(
        ChatProvider.OPENAI_COMPATIBLE,
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )

    assert settings.api_key == "ollama"
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.model == "llama3.2"

    with pytest.raises(ValueError, match="http or https"):
        build_chat_settings(
            ChatProvider.OPENAI_COMPATIBLE,
            api_key="ollama",
            base_url="localhost:11434/v1",
            model="llama3.2",
        )

    with pytest.raises(ValueError, match="model"):
        build_chat_settings(
            ChatProvider.OPENAI_COMPATIBLE,
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model="  ",
        )

    with pytest.raises(ValueError, match="base URL"):
        build_chat_settings(
            ChatProvider.OPENAI_COMPATIBLE,
            api_key="",
            base_url="",
            model="llama3.2",
        )


def test_native_openai_requires_api_key_and_uses_no_custom_url() -> None:
    settings = build_chat_settings(
        ChatProvider.OPENAI,
        api_key="openai-key",
        base_url="",
        model="gpt-4o-mini",
    )

    assert settings.base_url is None
    assert settings.api_key == "openai-key"

    with pytest.raises(ValueError, match="API key"):
        build_chat_settings(
            ChatProvider.OPENAI,
            api_key=" ",
            base_url="",
            model="gpt-4o-mini",
        )


def test_custom_embedding_endpoint_gets_placeholder_key_by_default() -> None:
    settings = build_embedding_settings("", "http://localhost:11434/v1")

    assert settings.api_key == "ollama"
    assert settings.base_url == "http://localhost:11434/v1"

    with pytest.raises(ValueError, match="http or https"):
        build_embedding_settings("embedding-key", "localhost:11434/v1")
