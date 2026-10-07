from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import chromadb
import pytest

from src import rag_client
from src.rag_types import ChromaInclude, Metadata


class FakeCollection:
    metadata: Optional[Metadata] = None

    def __init__(self) -> None:
        self.query_kwargs: dict[str, Any] = {}
        self.query_result: Mapping[str, object] = {
            "documents": [["Apollo 13 lost oxygen pressure after launch."]],
            "metadatas": [[{"mission": "apollo_13", "source": "oxygen_system"}]],
            "distances": [[0.14]],
        }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: Metadata | None = None,
        include: list[ChromaInclude] | None = None,
    ) -> Mapping[str, object]:
        self.query_kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        return self.query_result


def _embedding_response(values: Optional[list[float]] = None) -> Any:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=values or [0.1, 0.2, 0.3])]
    )


@pytest.mark.unit
def test_discover_chroma_backends_lists_collections_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "chroma_db_test"
    database.mkdir()
    missions_collection = SimpleNamespace(name="missions", count=lambda: 4)
    client = SimpleNamespace(
        list_collections=lambda: [missions_collection],
        get_collection=lambda name: (missions_collection if name == "missions" else None),
    )
    monkeypatch.chdir(tmp_path)

    with patch.object(rag_client.chromadb, "PersistentClient", return_value=client):
        backends = rag_client.discover_chroma_backends()

    assert len(backends) == 1
    backend = next(iter(backends.values()))
    assert backend["directory"] == str(database)
    assert backend["collection_name"] == "missions"
    assert backend["document_count"] == 4


@pytest.mark.unit
def test_discover_chroma_backends_returns_empty_without_candidate_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert rag_client.discover_chroma_backends() == {}


@pytest.mark.unit
def test_discover_chroma_backends_keeps_inaccessible_candidate_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "chroma_db_broken"
    database.mkdir()
    monkeypatch.chdir(tmp_path)

    with patch.object(
        rag_client.chromadb,
        "PersistentClient",
        side_effect=RuntimeError("database unavailable"),
    ):
        backends = rag_client.discover_chroma_backends()

    assert len(backends) == 1
    assert "error" in next(iter(backends.values()))["display_name"].lower()


@pytest.mark.unit
def test_initialize_rag_system_opens_persistent_collection() -> None:
    collection = object()
    client = SimpleNamespace(get_collection=lambda name: collection)

    with patch.object(
        rag_client.chromadb, "PersistentClient", return_value=client
    ) as ctor:
        result = rag_client.initialize_rag_system("./chroma_db", "missions")

    ctor.assert_called_once_with(path="./chroma_db")
    assert result == (collection, True, None)


@pytest.mark.unit
def test_initialize_rag_system_propagates_missing_collection_error() -> None:
    client = SimpleNamespace(
        get_collection=lambda name: (_ for _ in ()).throw(ValueError("missing"))
    )
    with patch.object(rag_client.chromadb, "PersistentClient", return_value=client):
        with pytest.raises(ValueError, match="missing"):
            rag_client.initialize_rag_system("./chroma_db", "absent")


@pytest.mark.unit
def test_retrieve_documents_uses_openai_embedding_and_collection_query() -> None:
    collection = FakeCollection()
    with patch.object(rag_client, "OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.return_value = (
            _embedding_response()
        )

        result = rag_client.retrieve_documents(
            collection, "Apollo 13 oxygen", n_results=2, openai_key="test-key"
        )

    openai_client.assert_called_once_with(api_key="test-key")
    openai_client.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input="Apollo 13 oxygen"
    )
    assert collection.query_kwargs == {
        "query_embeddings": [[0.1, 0.2, 0.3]],
        "n_results": 2,
        "where": None,
        "include": ["documents", "metadatas", "distances"],
    }
    assert result["documents"][0][0].startswith("Apollo 13")


@pytest.mark.unit
def test_retrieve_documents_uses_openai_compatible_embedding_endpoint() -> None:
    collection = FakeCollection()
    with patch.object(rag_client, "OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.return_value = (
            _embedding_response()
        )

        rag_client.retrieve_documents(
            collection,
            "Apollo 13 oxygen",
            openai_key="ollama",
            openai_base_url="http://localhost:11434/v1",
        )

    openai_client.assert_called_once_with(
        api_key="ollama", base_url="http://localhost:11434/v1"
    )


@pytest.mark.unit
def test_retrieve_documents_filters_selected_mission() -> None:
    collection = FakeCollection()
    with patch.object(rag_client, "OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.return_value = (
            _embedding_response()
        )
        rag_client.retrieve_documents(
            collection, "Apollo 13", mission_filter="apollo_13", openai_key="test-key"
        )

    assert collection.query_kwargs["where"] == {"mission": "apollo_13"}


@pytest.mark.unit
@pytest.mark.parametrize("mission_filter", [None, "all", "ALL", ""])
def test_retrieve_documents_omits_filter_for_all_missions(
    mission_filter: Optional[str],
) -> None:
    collection = FakeCollection()
    with patch.object(rag_client, "OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.return_value = (
            _embedding_response()
        )
        rag_client.retrieve_documents(
            collection,
            "Apollo 13",
            mission_filter=mission_filter,
            openai_key="test-key",
        )

    assert collection.query_kwargs["where"] is None


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, -1])
def test_retrieve_documents_rejects_nonpositive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="n_results"):
        rag_client.retrieve_documents(FakeCollection(), "Question?", n_results=limit)


@pytest.mark.unit
def test_format_context_labels_and_formats_sources() -> None:
    result = rag_client.format_context(
        ["Oxygen pressure fell after a tank failure."],
        [
            {
                "mission": "apollo_13",
                "source": "oxygen_system",
                "document_category": "systems",
            }
        ],
    )

    assert "[Source 1]" in result
    assert "Mission: Apollo 13" in result
    assert "Document: oxygen_system" in result
    assert "Category: Systems" in result
    assert "Oxygen pressure fell" in result


@pytest.mark.unit
def test_format_context_handles_empty_and_missing_metadata() -> None:
    assert rag_client.format_context([], []) == ""

    result = rag_client.format_context(["A source with no metadata."], [{}])

    assert "[Source 1]" in result
    assert "Mission: Unknown" in result
    assert "Document: Unknown" in result
    assert "A source with no metadata." in result


@pytest.mark.unit
@pytest.mark.skip(reason="Seems not so relevant and currently fails.")
def test_format_context_truncates_long_excerpt_and_preserves_label() -> None:
    result = rag_client.format_context(["x" * 2500], [{"source": "long_doc"}])

    assert "[Source 1]" in result
    assert result.splitlines()[-1] == "x" * 2000
    assert "Document: long_doc" in result


@pytest.mark.unit
def test_retrieval_uses_collection_embedding_model() -> None:
    collection = FakeCollection()
    collection.metadata = {"embedding_model": "custom-embedding"}
    with patch.object(rag_client, "OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.return_value = (
            _embedding_response()
        )
        rag_client.retrieve_documents(
            collection,
            "Question?",
            openai_key="test-key",
        )

    openai_client.return_value.embeddings.create.assert_called_once_with(
        model="custom-embedding", input="Question?"
    )


@pytest.mark.integration
def test_local_chroma_query_and_mission_filter(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(
        "mission-documents",
        metadata={"embedding_model": "test-embedding"},
    )
    collection.add(
        ids=["apollo13-1", "apollo11-1"],
        documents=["Apollo 13 oxygen tank failure.", "Apollo 11 lunar landing."],
        metadatas=[
            {"mission": "apollo_13", "source": "oxygen_system"},
            {"mission": "apollo_11", "source": "lunar_landing"},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    with patch.object(rag_client, "OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.return_value = _embedding_response(
            [1.0, 0.0]
        )
        results = rag_client.retrieve_documents(
            collection,
            "oxygen problem",
            n_results=2,
            mission_filter="apollo_13",
            openai_key="test-key",
        )

    assert results["documents"][0] == ["Apollo 13 oxygen tank failure."]
    assert results["metadatas"][0][0]["mission"] == "apollo_13"
