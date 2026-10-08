"""ChromaDB discovery, retrieval, and source-context formatting."""

import math
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

import chromadb
from chromadb.api.models import Collection as chromadb_collection
import chromadb.api.types as chromadb_types
from openai import OpenAI

from src.config.api_config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT_SECONDS
from src.rag_types import ChromaBackend


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CONTEXT_EXCERPT_LENGTH = 2000


def discover_chroma_backends() -> dict[str, ChromaBackend]:
    """Discover immediate project-child ChromaDB directories and collections."""
    backends: dict[str, ChromaBackend] = {}
    for directory in sorted(Path(".").iterdir()):
        if not directory.is_dir() or not directory.name.startswith("chroma_db"):
            continue

        directory_path = str(directory)
        try:
            directory_path = str(directory.resolve())
            client = chromadb.PersistentClient(path=directory_path)
            for collection in client.list_collections():
                try:
                    document_count = collection.count()
                    error = None
                except Exception as collection_error:
                    document_count = "Unknown"
                    error = str(collection_error)

                key = f"{directory.name}:{collection.name}"
                display_name = (
                    f"{directory.name} / {collection.name} "
                    f"({document_count} documents)"
                )
                if error:
                    display_name += f" - count unavailable: {error[:120]}"
                backends[key] = {
                    "directory": directory_path,
                    "collection_name": collection.name,
                    "display_name": display_name,
                    "document_count": document_count,
                }
        except Exception as error:
            key = f"{directory.name}:error"
            backends[key] = {
                "directory": directory_path,
                "collection_name": "",
                "display_name": f"{directory.name} - error: {str(error)[:160]}",
                "document_count": "Unknown",
            }
    return backends


def initialize_rag_system(
    chroma_dir: str, collection_name: str
) -> tuple[chromadb_collection.Collection, bool, Optional[str]]:
    """Open a persistent ChromaDB collection for retrieval."""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name)
    return collection, True, None


# TODO: It does not look good to have the embedding model selection logic in the retrieval function. 
#  Consider moving it to a separate function or configuration.
def retrieve_documents(
    collection: chromadb_collection.Collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    openai_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> chromadb.QueryResult:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    """Retrieve top matching chunks using the index's configured embedding model."""
    if not isinstance(n_results, int) or n_results <= 0:
        raise ValueError("n_results must be a positive integer")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if embedding_model is not None and not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string")
    selected_embedding_model = (
        embedding_model
        or (collection.metadata or {}).get("embedding_model")
        or DEFAULT_EMBEDDING_MODEL
    )

    api_key = (
        openai_key
        or os.getenv("CHROMA_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "An API key is required for query embeddings; set "
            "CHROMA_OPENAI_API_KEY or OPENAI_API_KEY"
        )

    client_options: dict[str, Any] = {}
    if openai_base_url:
        client_options["base_url"] = openai_base_url
    embedding_response = OpenAI(
        api_key=api_key,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
        **client_options,
    ).embeddings.create(
        model=selected_embedding_model,
        input=query,
    )
    try:
        query_embedding = embedding_response.data[0].embedding
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("Embedding provider returned a malformed response") from error
    if not query_embedding:
        raise ValueError("Embedding provider returned an empty embedding")
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for value in query_embedding
    ):
        raise ValueError("Embedding provider returned an invalid embedding vector")

    where: Optional[chromadb_types.Where] = None
    normalized_mission = mission_filter.strip() if mission_filter else ""
    if normalized_mission and normalized_mission.lower() != "all":
        where = {"mission": normalized_mission}

    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return query_result


def format_context(
    documents: List[str],
    metadatas: List[chromadb_types.Metadata],
) -> str:
    """Format retrieved excerpts with stable source labels and attribution."""
    if not documents or not metadatas:
        return ""
    if len(documents) != len(metadatas):
        raise ValueError(
            "Documents and metadatas must have the same length for formatting"
        )

    context_parts = ["Retrieved NASA source excerpts:"]
    metadata_values = metadatas or []
    for index, document in enumerate(documents, start=1):
        get_metadata: Callable[[str], Optional[Any]] = lambda key: (
            metadata_values[index - 1].get(key)
            if index - 1 < len(metadata_values) and (key in metadata_values[index - 1])
            else None
        )
        mission = str(get_metadata("mission") or "Unknown")
        mission_label = mission.replace("_", " ").title()
        source = str(get_metadata("source") or get_metadata("file_path") or "Unknown")
        category = str(
            get_metadata("document_category")
            or "Unknown"
        ).replace("_", " ").title()
        excerpt = document or ""
        if len(excerpt) > MAX_CONTEXT_EXCERPT_LENGTH:
            excerpt = excerpt[: MAX_CONTEXT_EXCERPT_LENGTH - 3].rstrip() + "..."

        context_parts.extend(
            [
                f"[Source {index}] Mission: {mission_label}; "
                f"Document: {source}; Category: {category}",
                excerpt,
            ]
        )
    return "\n".join(context_parts)
