"""Shared type definitions for the NASA RAG pipeline."""

import math
from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict

from chromadb.api.types import Include as ChromaInclude


class ConversationTurn(TypedDict):
    """A supported role/content pair sent to the chat completion API."""

    role: Literal["user", "assistant"]
    content: str


Metadata = dict[str, Any]


class ChromaBackend(TypedDict):
    """A discovered persisted ChromaDB collection shown in the chat UI."""

    directory: str
    collection_name: str
    display_name: str
    document_count: int | str


class RetrievalResult(TypedDict):
    """Subset of the ChromaDB query response consumed by this application."""

    documents: list[list[str | None]]
    metadatas: list[list[Metadata | None]]
    distances: list[list[float]]


def normalize_retrieval_result(result: Mapping[str, object]) -> RetrievalResult:
    """Validate the required fields returned by ChromaDB queries."""
    raw_documents = result.get("documents")
    raw_metadatas = result.get("metadatas")
    raw_distances = result.get("distances")
    if (
        not isinstance(raw_documents, list)
        or not isinstance(raw_metadatas, list)
        or not isinstance(raw_distances, list)
    ):
        raise ValueError("ChromaDB returned a malformed query result")

    documents: list[list[str | None]] = []
    for row in raw_documents:
        if not isinstance(row, list):
            raise ValueError("ChromaDB returned a malformed query result")
        normalized_row: list[str | None] = []
        for document in row:
            if document is not None and not isinstance(document, str):
                raise ValueError("ChromaDB returned a malformed query result")
            normalized_row.append(document)
        documents.append(normalized_row)

    metadatas: list[list[Metadata | None]] = []
    for row in raw_metadatas:
        if not isinstance(row, list):
            raise ValueError("ChromaDB returned a malformed query result")
        normalized_row_metadata: list[Metadata | None] = []
        for metadata in row:
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("ChromaDB returned a malformed query result")
            normalized_row_metadata.append(metadata)
        metadatas.append(normalized_row_metadata)

    distances: list[list[float]] = []
    for row in raw_distances:
        if not isinstance(row, list):
            raise ValueError("ChromaDB returned a malformed query result")
        normalized_row_distances: list[float] = []
        for distance in row:
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(distance)
            ):
                raise ValueError("ChromaDB returned a malformed query result")
            normalized_row_distances.append(float(distance))
        distances.append(normalized_row_distances)

    if (
        len(documents) != len(metadatas)
        or len(documents) != len(distances)
        or any(
            len(document_row) != len(metadata_row)
            or len(document_row) != len(distance_row)
            for document_row, metadata_row, distance_row in zip(
                documents, metadatas, distances
            )
        )
    ):
        raise ValueError("ChromaDB returned a malformed query result")

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
    }


class DocumentUpdateStats(TypedDict):
    """Counts produced while adding or updating chunks."""

    added: int
    updated: int
    skipped: int


class MissionProcessStats(TypedDict):
    """Per-mission document-processing counters."""

    files: int
    chunks: int
    added: int
    updated: int
    skipped: int


class PipelineStats(TypedDict):
    """Aggregate counters returned by a full embedding-pipeline run."""

    files_processed: int
    documents_added: int
    documents_updated: int
    documents_skipped: int
    errors: int
    total_chunks: int
    missions: dict[str, MissionProcessStats]


class CollectionInfo(TypedDict):
    """Persistent ChromaDB collection summary."""

    collection_name: str
    document_count: int
    metadata: Metadata


class CollectionStats(TypedDict):
    """Aggregated metadata counts for the indexed collection."""

    total_documents: int
    missions: dict[str, int]
    data_types: dict[str, int]
    document_categories: dict[str, int]
    file_types: dict[str, int]


class RetrievalCollection(Protocol):
    """ChromaDB query surface required by the RAG client."""

    @property
    def metadata(self) -> Metadata | None:
        """Return collection-level configuration metadata."""

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: Metadata | None = None,
        include: list[ChromaInclude] | None = None,
    ) -> Mapping[str, object]:
        """Retrieve nearest vectors, optionally filtering by metadata."""


class ChromaCollection(RetrievalCollection, Protocol):
    """ChromaDB collection operations used by indexing and retrieval."""

    def count(self) -> int:
        """Return the number of stored chunks."""

    def get(
        self,
        ids: list[str] | None = None,
        include: list[ChromaInclude] | None = None,
    ) -> Mapping[str, Any]:
        """Fetch records by id or retrieve collection records."""

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Metadata],
        embeddings: list[list[float]],
    ) -> None:
        """Add new records to the collection."""

    def update(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Metadata],
        embeddings: list[list[float]],
    ) -> None:
        """Replace existing records in the collection."""

    def delete(self, ids: list[str]) -> None:
        """Delete records by id."""

    def modify(self, metadata: Metadata) -> None:
        """Update collection metadata."""
