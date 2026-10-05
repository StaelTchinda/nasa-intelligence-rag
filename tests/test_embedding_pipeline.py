from pathlib import Path

from src.embedding_pipeline import ChromaEmbeddingPipelineTextOnly


def test_chunk_text_returns_chunks_with_metadata():
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key="test-key",
        collection_name="unit-test-collection",
    )

    text = (
        "Apollo 11 was the first crewed mission to land on the Moon. "
        "Neil Armstrong and Buzz Aldrin explored the lunar surface. "
        "The mission returned samples and valuable scientific data."
    )

    chunks = pipeline.chunk_text(text, {"mission": "apollo11", "source": "mission_overview"})

    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    for chunk, metadata in chunks:
        assert isinstance(chunk, str)
        assert metadata["mission"] == "apollo11"
        assert metadata["source"] == "mission_overview"


def test_generate_document_id_returns_stable_string():
    pipeline = ChromaEmbeddingPipelineTextOnly(openai_api_key="test-key")
    doc_id = pipeline.generate_document_id(Path("apollo11/mission_overview.txt"), {"chunk_index": 2})

    assert isinstance(doc_id, str)
    assert "apollo11" in doc_id.lower()
    assert "mission_overview" in doc_id.lower()
