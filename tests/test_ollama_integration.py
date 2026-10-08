import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import patch
from urllib.request import urlopen

import chromadb
import pytest

from src import llm_client, rag_client


class OllamaModel(TypedDict):
    name: str


class OllamaTags(TypedDict):
    models: list[OllamaModel]


def _ollama_configuration() -> tuple[str, str]:
    base_url = os.getenv(
        "OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"
    ).rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", "")
    strict = os.getenv("NASA_RAG_REQUIRE_OLLAMA") == "1"
    if not model:
        message = "OLLAMA_CHAT_MODEL is not set"
        if strict:
            pytest.fail(message)
        pytest.skip(message)

    models_url = base_url.removesuffix("/v1") + "/api/tags"
    try:
        with urlopen(models_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError as error:
        message = f"Ollama is not available at {models_url}: {error}"
        if strict:
            pytest.fail(message)
        pytest.skip(message)

    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("models"), list)
        or any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str)
            for item in payload["models"]
        )
    ):
        raise ValueError("Ollama returned an invalid model-list response")
    available: OllamaTags = payload
    model_names = {item["name"] for item in available["models"]}
    if model not in model_names and not any(
        installed.split(":")[0] == model.split(":")[0] for installed in model_names
    ):
        message = f"Ollama model {model!r} is not installed"
        if strict:
            pytest.fail(message)
        pytest.skip(message)
    return base_url, model


@pytest.mark.integration
def test_ollama_answers_using_retrieved_local_chroma_context(
    tmp_path: Path,
) -> None:
    base_url, model = _ollama_configuration()
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("ollama-fixture")
    source = (
        "Apollo 13 suffered an oxygen tank failure after launch. "
        "The crew used the lunar module as a lifeboat."
    )
    collection.add(
        ids=["apollo13-oxygen-0"],
        documents=[source],
        metadatas=[
            {
                "mission": "apollo_13",
                "source": "oxygen_system",
                "document_category": "systems",
            }
        ],
        embeddings=[[0.25, 0.75, 0.5]],
    )

    with patch.object(rag_client, "OpenAI") as embedding_client:
        embedding_client.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25, 0.75, 0.5])]
        )
        results = rag_client.retrieve_documents(
            collection,
            "What system failed on Apollo 13?",
            n_results=1,
            mission_filter="apollo_13",
            openai_key="ollama",
        )

    assert results["documents"] is not None
    assert results["metadatas"] is not None
    
    context = rag_client.format_context(
        results["documents"][0], results["metadatas"][0]
    )
    answer = llm_client.generate_response(
        "ollama",
        "What failed, and what did the crew use as a lifeboat?",
        context,
        [],
        model=model,
        openai_base_url=base_url,
    )

    lowered_answer = answer.lower()
    assert answer.strip()
    assert "oxygen" in lowered_answer
    assert "lifeboat" in lowered_answer or "lunar module" in lowered_answer
    assert "[source 1]" in lowered_answer
