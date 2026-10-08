from unittest.mock import patch

import pytest

from src import chat
from src.rag_types import ConversationTurn


@pytest.mark.unit
def test_chat_wrapper_can_generate_and_evaluate() -> None:
    with patch.object(
        chat.llm_client,
        "generate_response",
        return_value="Mock answer about Apollo 13",
    ) as generate_response:
        with patch.object(
            chat.ragas_evaluator,
            "evaluate_response_quality",
            return_value={"faithfulness": 0.97, "response_relevancy": 0.93},
        ) as evaluate_response_quality:
            response = chat.generate_response(
                "fake-key",
                "What happened on Apollo 13?",
                "Apollo 13 lost oxygen after launch.",
                [
                    ConversationTurn(
                        role="user",
                        content="Tell me about Apollo 13.",
                    )
                ],
                openai_base_url="http://localhost:11434/v1",
            )
            evaluation = chat.evaluate_response_quality(
                "What happened on Apollo 13?",
                "Mock answer about Apollo 13",
                ["Apollo 13 lost oxygen after launch."],
            )

    generate_response.assert_called_once_with(
        "fake-key",
        "What happened on Apollo 13?",
        "Apollo 13 lost oxygen after launch.",
        [
            ConversationTurn(
                role="user",
                content="Tell me about Apollo 13.",
            )
        ],
        "gpt-3.5-turbo",
        openai_base_url="http://localhost:11434/v1",
    )
    evaluate_response_quality.assert_called_once()

    assert response == "Mock answer about Apollo 13"
    assert evaluation["faithfulness"] >= 0.9


@pytest.mark.unit
def test_chat_wrapper_preserves_legacy_generation_call() -> None:
    with patch.object(
        chat.llm_client, "generate_response", return_value="Legacy-compatible answer"
    ) as generate_response:
        result = chat.generate_response("fake-key", "Question?", "Context", [])

    generate_response.assert_called_once_with(
        "fake-key", "Question?", "Context", [], "gpt-3.5-turbo"
    )
    assert result == "Legacy-compatible answer"


@pytest.mark.unit
def test_retrieval_wrapper_passes_embedding_provider_settings() -> None:
    collection = object()
    result = object()
    with patch.object(
        chat.rag_client, "retrieve_documents", return_value=result
    ) as retrieve_documents:
        actual = chat.retrieve_documents(
            collection,
            "What happened on Apollo 13?",
            5,
            openai_key="embedding-key",
            openai_base_url="http://localhost:11434/v1",
        )

    retrieve_documents.assert_called_once_with(
        collection,
        "What happened on Apollo 13?",
        5,
        None,
        openai_key="embedding-key",
        openai_base_url="http://localhost:11434/v1",
    )
    assert actual is result
