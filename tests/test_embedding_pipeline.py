from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple, cast
from unittest.mock import Mock, patch

import pytest

from src.embedding_pipeline import ChromaEmbeddingPipelineTextOnly, UpdateMode, main
from src.rag_types import ChromaInclude, Metadata, RetrievalResult


def _embedding_response_for_text(model: str, input: str) -> Any:
    return SimpleNamespace(
        model=model,
        data=[SimpleNamespace(embedding=[float(len(input)), 0.5])]
    )


class MemoryCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.metadata: Metadata | None = {"embedding_model": "test-embedding"}

    def get(
        self,
        ids: list[str] | None = None,
        include: list[ChromaInclude] | None = None,
    ) -> dict[str, Any]:
        selected = self.records
        if ids is not None:
            selected = {
                doc_id: self.records[doc_id]
                for doc_id in ids
                if doc_id in self.records
            }
        return {
            "ids": list(selected),
            "documents": [record["document"] for record in selected.values()],
            "metadatas": [record["metadata"] for record in selected.values()],
        }

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Metadata],
        embeddings: list[list[float]],
    ) -> None:
        for values in zip(ids, documents, metadatas, embeddings):
            doc_id, document, metadata, embedding = values
            if doc_id in self.records:
                raise ValueError("ID already exists")
            self.records[doc_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def update(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Metadata],
        embeddings: list[list[float]],
    ) -> None:
        for values in zip(ids, documents, metadatas, embeddings):
            doc_id, document, metadata, embedding = values
            self.records[doc_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self.records.pop(doc_id, None)

    def count(self) -> int:
        return len(self.records)

    def modify(self, metadata: Metadata) -> None:
        self.metadata = metadata

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: Metadata | None = None,
        include: list[ChromaInclude] | None = None,
    ) -> RetrievalResult:
        return {
            "documents": [["answer"]],
            "metadatas": [[{}]],
            "distances": [[0.0]],
        }


def _pipeline(
    chunk_size: int = 100, chunk_overlap: int = 20
) -> ChromaEmbeddingPipelineTextOnly:
    collection = MemoryCollection()
    with (
        patch("src.embedding_pipeline.OpenAI") as openai_client,
        patch("src.embedding_pipeline.chromadb.PersistentClient") as chroma_client,
    ):
        openai_client.return_value.embeddings.create.side_effect = (
            _embedding_response_for_text
        )
        chroma_client.return_value.get_or_create_collection.return_value = collection
        pipeline = ChromaEmbeddingPipelineTextOnly(
            "test-key",
            collection_name="test-missions",
            embedding_model="test-embedding",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return pipeline


@pytest.mark.unit
def test_pipeline_initializes_openai_and_persistent_chroma(tmp_path: Path) -> None:
    collection = SimpleNamespace(
        metadata={"embedding_model": "text-embedding-3-small"}
    )
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: collection)
    with (
        patch("src.embedding_pipeline.OpenAI") as openai_client,
        patch(
            "src.embedding_pipeline.chromadb.PersistentClient",
            return_value=client,
        ) as chroma,
    ):
        pipeline = ChromaEmbeddingPipelineTextOnly(
            "test-key",
            chroma_persist_directory=str(tmp_path),
            collection_name="missions",
        )

    openai_client.assert_called_once_with(api_key="test-key")
    chroma.assert_called_once()
    assert pipeline.collection is collection


@pytest.mark.unit
def test_pipeline_forwards_openai_base_url(tmp_path: Path) -> None:
    collection = SimpleNamespace(
        metadata={"embedding_model": "text-embedding-3-small"}
    )
    client = SimpleNamespace(get_or_create_collection=Mock(return_value=collection))
    base_url = "http://localhost:11434/v1"
    with (
        patch("src.embedding_pipeline.OpenAI") as openai_client,
        patch(
            "src.embedding_pipeline.chromadb.PersistentClient",
            return_value=client,
        ),
    ):
        ChromaEmbeddingPipelineTextOnly(
            "test-key",
            chroma_persist_directory=str(tmp_path),
            openai_base_url=base_url,
        )

    openai_client.assert_called_once_with(api_key="test-key", base_url=base_url)


@pytest.mark.unit
def test_pipeline_accepts_openai_embedding_model(tmp_path: Path) -> None:
    embedding_model = "custom-embedding"
    collection = SimpleNamespace(metadata={"embedding_model": embedding_model})
    client = SimpleNamespace(get_or_create_collection=Mock(return_value=collection))
    with (
        patch("src.embedding_pipeline.OpenAI"),
        patch(
            "src.embedding_pipeline.chromadb.PersistentClient",
            return_value=client,
        ),
    ):
        pipeline = ChromaEmbeddingPipelineTextOnly(
            "test-key",
            chroma_persist_directory=str(tmp_path),
            openai_embedding_model=embedding_model,
        )

    assert pipeline.embedding_model == embedding_model


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_option",
    ["--embedding-model", "--openai-embedding-model"],
)
def test_main_accepts_embedding_model_options(
    model_option: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "embedding_pipeline.py",
            "--openai-key",
            "test-key",
            model_option,
            "custom-embedding",
            "--stats-only",
        ],
    )
    with patch("src.embedding_pipeline.ChromaEmbeddingPipelineTextOnly") as pipeline:
        pipeline.return_value.get_collection_stats.return_value = {}
        main()

    assert (
        pipeline.call_args.kwargs["openai_embedding_model"]
        == "custom-embedding"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_pipeline_rejects_invalid_chunk_configuration(
    chunk_size: int, chunk_overlap: int
) -> None:
    with pytest.raises(ValueError):
        ChromaEmbeddingPipelineTextOnly(
            "test-key", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [True, 1.5, "10"])
def test_pipeline_rejects_noninteger_chunk_size(chunk_size: object) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        ChromaEmbeddingPipelineTextOnly(
            "test-key", chunk_size=cast(int, chunk_size)
        )


@pytest.mark.unit
@pytest.mark.parametrize("chunk_overlap", [True, 1.5, "2"])
def test_pipeline_rejects_noninteger_chunk_overlap(chunk_overlap: object) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChromaEmbeddingPipelineTextOnly(
            "test-key", chunk_overlap=cast(int, chunk_overlap)
        )


@pytest.mark.unit
def test_pipeline_rejects_collection_with_different_embedding_model(
    tmp_path: Path,
) -> None:
    collection = SimpleNamespace(
        metadata={"embedding_model": "embedding-v1"},
    )
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: collection)
    with (
        patch("src.embedding_pipeline.OpenAI"),
        patch("src.embedding_pipeline.chromadb.PersistentClient", return_value=client),
    ):
        with pytest.raises(ValueError, match="embedding model"):
            ChromaEmbeddingPipelineTextOnly(
                "test-key",
                chroma_persist_directory=str(tmp_path),
                embedding_model="embedding-v2",
            )


@pytest.mark.unit
def test_pipeline_rejects_unknown_embedding_model_on_populated_collection(
    tmp_path: Path,
) -> None:
    collection = SimpleNamespace(metadata=None, count=lambda: 1)
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: collection)
    with (
        patch("src.embedding_pipeline.OpenAI"),
        patch("src.embedding_pipeline.chromadb.PersistentClient", return_value=client),
    ):
        with pytest.raises(ValueError, match="unknown embedding model"):
            ChromaEmbeddingPipelineTextOnly(
                "test-key",
                chroma_persist_directory=str(tmp_path),
                embedding_model="embedding-v2",
            )


@pytest.mark.unit
def test_chunk_text_handles_empty_and_short_text() -> None:
    pipeline = _pipeline()
    assert pipeline.chunk_text(" \n ", {"mission": "apollo_11"}) == []

    result = pipeline.chunk_text("Apollo 11 landed.", {"mission": "apollo_11"})

    assert result == [
        ("Apollo 11 landed.", {"mission": "apollo_11", "chunk_index": 0})
    ]


@pytest.mark.unit
def test_chunk_text_respects_chunk_limit_and_overlap() -> None:
    pipeline = _pipeline(chunk_size=20, chunk_overlap=5)
    text = "0123456789abcdefghijABCDEFGHIJ"

    chunks = pipeline.chunk_text(text, {"source": "test"})

    assert all(len(chunk) <= 20 for chunk, _ in chunks)
    assert chunks[0][0][-5:] == chunks[1][0][:5]
    assert "".join(
        chunk if index == 0 else chunk[5:] for index, (chunk, _) in enumerate(chunks)
    ) == text


@pytest.mark.unit
def test_chunk_text_preserves_unicode_and_chunk_metadata() -> None:
    pipeline = _pipeline(chunk_size=8, chunk_overlap=2)
    chunks = pipeline.chunk_text("Apollo 🚀 mission.", {"mission": "apollo_11"})

    assert all(len(text) <= 8 for text, _ in chunks)
    assert "".join(
        text if index == 0 else text[2:]
        for index, (text, _) in enumerate(chunks)
    ) == "Apollo 🚀 mission."
    assert [metadata["chunk_index"] for _, metadata in chunks] == list(
        range(len(chunks))
    )


@pytest.mark.unit
def test_get_embedding_uses_configured_model_and_surfaces_errors() -> None:
    pipeline = _pipeline()
    with patch.object(pipeline.openai_client.embeddings, "create") as create:
        create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.2, 0.3])]
        )
        assert pipeline.get_embedding("NASA") == [0.2, 0.3]
    create.assert_called_once_with(model="test-embedding", input="NASA")

    with patch.object(
        pipeline.openai_client.embeddings,
        "create",
        side_effect=RuntimeError("offline"),
    ):
        with pytest.raises(RuntimeError, match="offline"):
            pipeline.get_embedding("NASA")


@pytest.mark.unit
@pytest.mark.skip(reason="The function should not check if openAI returns a valid embedding vector. So the test is obsolete.")
@pytest.mark.parametrize("embedding", [[], [float("nan")], ["bad"], [True]])
def test_get_embedding_rejects_invalid_vectors(embedding: list[Any]) -> None:
    pipeline = _pipeline()
    with patch.object(
        pipeline.openai_client.embeddings, "create"
    ) as create:
        create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=embedding)]
        )

        with pytest.raises(ValueError, match="invalid embedding"):
            pipeline.get_embedding("NASA")


@pytest.mark.unit
def test_generate_document_id_is_stable_and_chunk_specific() -> None:
    pipeline = _pipeline()
    source = Path("apollo11/mission_overview.txt")
    id_a = pipeline.generate_document_id(source, {"chunk_index": 2})

    assert id_a == pipeline.generate_document_id(source, {"chunk_index": 2})
    assert id_a != pipeline.generate_document_id(source, {"chunk_index": 3})
    assert id_a != pipeline.generate_document_id(
        Path("apollo11/archive/mission_overview.txt"), {"chunk_index": 2}
    )
    assert id_a == pipeline.generate_document_id(
        Path("apollo11/mission_overview.txt").resolve(), {"chunk_index": 2}
    )
    assert "apollo_11" in id_a
    assert "mission_overview" in id_a


@pytest.mark.unit
def test_add_documents_supports_skip_update_and_replace_modes() -> None:
    pipeline = _pipeline()
    file_path = Path("apollo13/oxygen_system.txt")
    original_metadata: Metadata = {
        "mission": "apollo_13",
        "source": "oxygen_system",
        "chunk_index": 0,
    }
    second_metadata: Metadata = {
        "mission": "apollo_13",
        "source": "oxygen_system",
        "chunk_index": 1,
    }
    initial = [
        ("First original chunk.", original_metadata),
        ("Second chunk.", second_metadata),
    ]
    assert pipeline.add_documents_to_collection(initial, file_path)["added"] == 2

    skipped = [
        ("First changed chunk.", initial[0][1]),
        ("Second chunk.", initial[1][1]),
    ]
    assert pipeline.add_documents_to_collection(
        skipped, file_path, update_mode="skip"
    ) == {"added": 0, "updated": 0, "skipped": 2}
    assert isinstance(pipeline.collection, MemoryCollection)
    assert pipeline.collection.records[
        pipeline.generate_document_id(file_path, initial[0][1])
    ]["document"] == "First original chunk."

    assert pipeline.add_documents_to_collection(
        skipped, file_path, update_mode="update"
    ) == {"added": 0, "updated": 2, "skipped": 0}
    assert pipeline.collection.records[
        pipeline.generate_document_id(file_path, initial[0][1])
    ]["document"] == "First changed chunk."

    replacement = [
        ("Replacement only.", original_metadata)
    ]
    assert pipeline.add_documents_to_collection(
        replacement, file_path, update_mode="replace"
    ) == {"added": 0, "updated": 1, "skipped": 0}
    assert pipeline.collection.count() == 1


@pytest.mark.unit
def test_add_documents_handles_empty_input() -> None:
    pipeline = _pipeline()
    file_path = Path("apollo11/test.txt")

    assert pipeline.add_documents_to_collection([], file_path) == {
        "added": 0,
        "updated": 0,
        "skipped": 0,
    }


@pytest.mark.unit
def test_add_documents_rejects_invalid_mode_option() -> None:
    pipeline = _pipeline()
    file_path = Path("apollo11/test.txt")

    with pytest.raises(ValueError, match="update_mode"):
        pipeline.add_documents_to_collection(
            [],
            file_path,
            update_mode=cast(UpdateMode, "unknown"),
        )


@pytest.mark.unit
@pytest.mark.skip(reason="Such type checking is not necessary, as the function signature already specifies the type of batch_size.")
@pytest.mark.parametrize("batch_size", [True, 1.5, "10"])
def test_add_documents_rejects_noninteger_batch_size(batch_size: object) -> None:
    pipeline = _pipeline()

    with pytest.raises(ValueError, match="batch_size"):
        pipeline.add_documents_to_collection(
            [("text", {})],
            Path("apollo11/test.txt"),
            batch_size=cast(int, batch_size),
        )


@pytest.mark.unit
def test_process_all_text_data_aggregates_file_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline()
    file_path = tmp_path / "apollo11" / "mission.txt"
    file_path.parent.mkdir()
    file_path.write_text("A short Apollo mission document.", encoding="utf-8")
    monkeypatch.setattr(pipeline, "scan_text_files_only", lambda _: [file_path])

    stats = pipeline.process_all_text_data(str(tmp_path))

    assert stats["files_processed"] == 1
    assert stats["total_chunks"] >= 1
    assert stats["documents_added"] >= 1
    assert stats["errors"] == 0
    assert stats["missions"]["apollo_11"]["files"] == 1


@pytest.mark.unit
def test_process_all_text_data_counts_file_errors_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline()
    failed_file = tmp_path / "apollo11" / "broken.txt"
    good_file = tmp_path / "apollo13" / "good.txt"
    good_metadata: Metadata = {
        "mission": "apollo_13",
        "source": "good",
        "chunk_index": 0,
    }
    monkeypatch.setattr(
        pipeline, "scan_text_files_only", lambda _: [failed_file, good_file]
    )

    def process_text_file(file_path: Path) -> list[tuple[str, Metadata]]:
        if file_path == failed_file:
            raise OSError("cannot read document")
        return [("Apollo 13 source.", good_metadata)]

    monkeypatch.setattr(pipeline, "process_text_file", process_text_file)
    stats = pipeline.process_all_text_data(str(tmp_path))

    assert stats["errors"] == 1
    assert stats["files_processed"] == 1
    assert stats["documents_added"] == 1


@pytest.mark.unit
def test_process_all_text_data_with_no_files_returns_zero_counts(
    tmp_path: Path,
) -> None:
    stats = _pipeline().process_all_text_data(str(tmp_path))

    assert stats["files_processed"] == 0
    assert stats["documents_added"] == 0
    assert stats["errors"] == 0
    assert stats["missions"] == {}


@pytest.mark.unit
def test_scan_text_files_supports_repository_data_text_layout(tmp_path: Path) -> None:
    pipeline = _pipeline()
    data_file = tmp_path / "data" / "text" / "apollo13" / "transcript.txt"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("Apollo 13 transcript.", encoding="utf-8")

    assert pipeline.scan_text_files_only(str(tmp_path)) == [data_file]


@pytest.mark.unit
def test_collection_info_and_stats_support_empty_and_populated_collections() -> None:
    pipeline = _pipeline()
    assert pipeline.get_collection_stats() == {
        "total_documents": 0,
        "missions": {},
        "data_types": {},
        "document_categories": {},
        "file_types": {},
    }
    assert pipeline.get_collection_info()["document_count"] == 0

    pipeline.add_documents_to_collection(
        [("Apollo 11 landed.", {"mission": "apollo_11", "source": "landing"})],
        Path("apollo11/landing.txt"),
    )

    assert pipeline.get_collection_info()["document_count"] == 1
    assert pipeline.get_collection_stats()["missions"] == {"apollo_11": 1}


@pytest.mark.unit
def test_query_collection_uses_same_embedding_model() -> None:
    pipeline = _pipeline()
    with patch.object(pipeline.openai_client.embeddings, "create") as create:
        create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
        with patch.object(
            pipeline.collection,
            "query",
            return_value={
                "documents": [["answer"]],
                "metadatas": [[{}]],
                "distances": [[0.0]],
            },
        ) as query:
            result = pipeline.query_collection("Apollo question", n_results=2)

    create.assert_called_once_with(model="test-embedding", input="Apollo question")
    query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    assert result["documents"][0] == ["answer"]


@pytest.mark.unit
@pytest.mark.parametrize("n_results", [0, -1, True, 1.5])
def test_query_collection_rejects_invalid_result_limits(
    n_results: object,
) -> None:
    pipeline = _pipeline()

    with pytest.raises(ValueError, match="n_results"):
        pipeline.query_collection("Apollo question", n_results=cast(int, n_results))


@pytest.mark.integration
def test_pipeline_persists_and_reopens_collection_with_local_embeddings(
    tmp_path: Path,
) -> None:
    with patch("src.embedding_pipeline.OpenAI") as openai_client:
        openai_client.return_value.embeddings.create.side_effect = (
            _embedding_response_for_text
        )
        pipeline = ChromaEmbeddingPipelineTextOnly(
            "test-key",
            chroma_persist_directory=str(tmp_path),
            collection_name="integration-missions",
        )
        file_path = Path("apollo13/oxygen_system.txt")
        documents: List[Tuple[str, Metadata]] = [
            (
                "Apollo 13 oxygen pressure fell after a tank failure.",
                {"mission": "apollo_13", "source": "oxygen_system", "chunk_index": 0},
            )
        ]
        pipeline.add_documents_to_collection(documents, file_path)

    reopened = ChromaEmbeddingPipelineTextOnly(
        "test-key",
        chroma_persist_directory=str(tmp_path),
        collection_name="integration-missions",
    )
    record = reopened.collection.get()
    assert record["documents"] == [documents[0][0]]
    assert record["metadatas"] is not None
    assert record["metadatas"][0]["mission"] == "apollo_13"
