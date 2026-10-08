from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import chromadb
import httpx
import pytest
from streamlit.testing.v1 import AppTest
from openai import APITimeoutError

from src import rag_client
from src.config import chat_config
from src import llm_client
from src.config.chat_config import (
    CUSTOM_MODEL_OPTION,
    ChatDefaults,
    ChatProvider,
)
from src.rag_types import ChromaBackend


@pytest.mark.unit
def test_custom_provider_and_model_controls_render_without_provider_calls() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "chat.py"
    defaults = ChatDefaults(
        provider=ChatProvider.OPENAI,
        openai_api_key="test-openai-key",
        compatible_api_key="ollama",
        chat_base_url=None,
        chat_model="gpt-3.5-turbo",
        embedding_api_key="test-embedding-key",
        embedding_base_url="http://localhost:11434/v1",
    )
    backends: dict[str, ChromaBackend] = {
        "test": {
            "directory": ".",
            "collection_name": "test_collection",
            "display_name": "test collection",
            "document_count": 1,
        }
    }
    collection = cast(
        chromadb.Collection,
        SimpleNamespace(metadata={"embedding_model": "embeddinggemma"}),
    )

    with (
        patch.object(chat_config, "load_chat_defaults", return_value=defaults),
        patch.object(rag_client, "discover_chroma_backends", return_value=backends),
        patch.object(
            rag_client,
            "initialize_rag_system",
            return_value=(collection, True, None),
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        assert any(item.label == "Chat provider" for item in app.selectbox)
        assert any(item.label == "Chat model" for item in app.selectbox)
        assert any(item.label == "Embedding API base URL" for item in app.text_input)

        app.selectbox(key="chat_provider").set_value(
            ChatProvider.OPENAI_COMPATIBLE
        ).run(timeout=30)
        assert app.text_input(key="compatible_chat_api_key").value == "ollama"
        app.text_input(key="compatible_chat_base_url").set_value(
            "http://localhost:11434/v1"
        ).run(timeout=30)
        app.selectbox(
            key=f"chat_model_choice_{ChatProvider.OPENAI_COMPATIBLE.value}"
        ).set_value(CUSTOM_MODEL_OPTION).run(timeout=30)

        assert any(item.label == "Custom chat model ID" for item in app.text_input)
        assert app.text_input(key="embedding_base_url").value == (
            "http://localhost:11434/v1"
        )
        app.selectbox(key="chat_provider").set_value(
            ChatProvider.OPENAI
        ).run(timeout=30)
        assert app.text_input(key="openai_chat_api_key").value == "test-openai-key"
        assert not app.exception


@pytest.mark.unit
def test_retrieval_error_stops_chat_generation() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "chat.py"
    defaults = ChatDefaults(
        provider=ChatProvider.OPENAI_COMPATIBLE,
        openai_api_key="",
        compatible_api_key="ollama",
        chat_base_url="http://localhost:11434/v1",
        chat_model="qwen3.5:4b",
        embedding_api_key="ollama",
        embedding_base_url="http://localhost:11434/v1",
    )
    backends: dict[str, ChromaBackend] = {
        "test": {
            "directory": ".",
            "collection_name": "test_collection",
            "display_name": "test collection",
            "document_count": 1,
        }
    }
    collection = cast(
        chromadb.Collection,
        SimpleNamespace(metadata={"embedding_model": "embeddinggemma"}),
    )

    with (
        patch.object(chat_config, "load_chat_defaults", return_value=defaults),
        patch.object(rag_client, "discover_chroma_backends", return_value=backends),
        patch.object(
            rag_client,
            "initialize_rag_system",
            return_value=(collection, True, None),
        ),
        patch.object(
            rag_client,
            "retrieve_documents",
            side_effect=RuntimeError("embedding provider unavailable"),
        ),
        patch.object(llm_client, "generate_response", return_value="Unexpected answer")
        as generate_response,
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app.chat_input[0].set_value("What happened on Apollo 13?").run(timeout=30)

    generate_response.assert_not_called()
    assert any(
        "Error retrieving documents" in element.value
        for element in app.error
    )


@pytest.mark.unit
def test_chat_provider_timeout_is_reported_to_user() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "chat.py"
    defaults = ChatDefaults(
        provider=ChatProvider.OPENAI_COMPATIBLE,
        openai_api_key="",
        compatible_api_key="ollama",
        chat_base_url="http://localhost:11434/v1",
        chat_model="qwen3.5:4b",
        embedding_api_key="ollama",
        embedding_base_url="http://localhost:11434/v1",
    )
    backends: dict[str, ChromaBackend] = {
        "test": {
            "directory": ".",
            "collection_name": "test_collection",
            "display_name": "test collection",
            "document_count": 1,
        }
    }
    collection = cast(
        chromadb.Collection,
        SimpleNamespace(metadata={"embedding_model": "embeddinggemma"}),
    )
    embedding_result = {
        "documents": [["A source excerpt."]],
        "metadatas": [[{"mission": "apollo_13", "source": "mission.txt"}]],
        "distances": [[0.1]],
    }

    with (
        patch.object(chat_config, "load_chat_defaults", return_value=defaults),
        patch.object(rag_client, "discover_chroma_backends", return_value=backends),
        patch.object(
            rag_client,
            "initialize_rag_system",
            return_value=(collection, True, None),
        ),
        patch.object(rag_client, "retrieve_documents", return_value=embedding_result),
        patch.object(
            llm_client,
            "generate_response",
            side_effect=APITimeoutError(
                request=httpx.Request(
                    "POST",
                    "http://localhost:11434/v1/chat/completions",
                )
            ),
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app.chat_input[0].set_value("What happened on Apollo 13?").run(timeout=30)

    assert any(
        "did not complete the request" in element.value
        for element in app.error
    )
    assert any(
        element.label == "Chat provider request failed"
        for element in app.get("status")
    )
    assert not app.exception


@pytest.mark.unit
def test_chat_turn_uses_separate_embedding_and_chat_endpoints() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "chat.py"
    defaults = ChatDefaults(
        provider=ChatProvider.OPENAI_COMPATIBLE,
        openai_api_key="",
        compatible_api_key="ollama-chat-key",
        chat_base_url="http://localhost:11434/v1",
        chat_model="qwen3.5:4b",
        embedding_api_key="ollama-embedding-key",
        embedding_base_url="http://localhost:11435/v1",
    )
    backends: dict[str, ChromaBackend] = {
        "test": {
            "directory": ".",
            "collection_name": "test_collection",
            "display_name": "test collection",
            "document_count": 1,
        }
    }
    collection = cast(
        chromadb.Collection,
        SimpleNamespace(metadata={"embedding_model": "embeddinggemma"}),
    )
    embedding_result = {
        "documents": [["Apollo 13 lost oxygen pressure."]],
        "metadatas": [[{"mission": "apollo_13", "source": "mission.txt"}]],
        "distances": [[0.1]],
    }

    with (
        patch.object(chat_config, "load_chat_defaults", return_value=defaults),
        patch.object(rag_client, "discover_chroma_backends", return_value=backends),
        patch.object(
            rag_client,
            "initialize_rag_system",
            return_value=(collection, True, None),
        ),
        patch.object(
            rag_client,
            "retrieve_documents",
            return_value=embedding_result,
        ) as retrieve_documents,
        patch.object(
            llm_client,
            "generate_response",
            return_value="Apollo 13 lost oxygen pressure. [Source 1]",
        ) as generate_response,
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app.chat_input[0].set_value("What happened on Apollo 13?").run(timeout=30)

    retrieve_documents.assert_called_once()
    retrieval_kwargs = retrieve_documents.call_args.kwargs
    assert retrieval_kwargs["openai_key"] == "ollama-embedding-key"
    assert retrieval_kwargs["openai_base_url"] == "http://localhost:11435/v1"
    generation_kwargs = generate_response.call_args.kwargs
    assert generation_kwargs["openai_base_url"] == "http://localhost:11434/v1"
    assert generate_response.call_args.args[0] == "ollama-chat-key"
    assert generate_response.call_args.args[4] == "qwen3.5:4b"
    assert any(
        "Apollo 13 lost oxygen pressure" in element.value
        for element in app.markdown
    )
    assert not app.exception
