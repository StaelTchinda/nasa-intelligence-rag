from types import SimpleNamespace
from typing import Any, Optional, List, cast
from unittest.mock import patch

import pytest

from src import llm_client
from src.rag_types import ConversationTurn


def _completion(content: str | None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.unit
def test_generate_response_uses_default_model_and_legacy_client_kwargs() -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = _completion(
            "Apollo 13 lost oxygen pressure. [Source 1]"
        )

        response = llm_client.generate_response(
            "fake-key", "What happened?", "Apollo 13 lost oxygen.", []
        )

    openai_client.assert_called_once_with(
        api_key="fake-key", timeout=120.0, max_retries=0
    )
    request = openai_client.return_value.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-3.5-turbo"
    assert response == "Apollo 13 lost oxygen pressure. [Source 1]"


@pytest.mark.unit
def test_generate_response_streams_answer_and_reports_reasoning_without_content() -> None:
    stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, reasoning="private thoughts")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="Apollo 13", reasoning=None))
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=" returned safely.", reasoning=None))
            ]
        ),
    ]
    progress: list[str] = []

    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = stream

        response = llm_client.generate_response(
            "fake-key",
            "What happened?",
            "Retrieved source.",
            [],
            on_progress=progress.append,
        )

    create_call = openai_client.return_value.chat.completions.create
    assert create_call.call_args.kwargs["stream"] is True
    assert response == "Apollo 13 returned safely."
    assert progress == ["thinking", "answering", "answering"]


@pytest.mark.unit
def test_generate_response_forwards_explicit_openai_base_url() -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = _completion(
            "Answer."
        )

        llm_client.generate_response(
            "ollama",
            "Question?",
            "Context",
            [],
            openai_base_url="http://localhost:11434/v1",
        )

    openai_client.assert_called_once_with(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        timeout=120.0,
        max_retries=0,
    )


@pytest.mark.unit
@pytest.mark.parametrize("base_url", [None, ""])
def test_generate_response_omits_empty_or_none_base_url(
    base_url: Optional[str],
) -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = _completion(
            "Answer."
        )

        llm_client.generate_response(
            "fake-key", "Question?", "Context", [], openai_base_url=base_url
        )

    openai_client.assert_called_once_with(
        api_key="fake-key", timeout=120.0, max_retries=0
    )


@pytest.mark.unit
def test_generate_response_includes_context_history_and_question_in_order() -> None:
    history: list[ConversationTurn] = [
        {"role": "user", "content": "Tell me about Apollo 13."},
        {"role": "assistant", "content": "What do you want to know?"},
    ]
    context = (
        "[Source 1] Mission: Apollo 13; Document: oxygen_system. "
        "Oxygen pressure fell."
    )

    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = _completion(
            "Answer."
        )
        llm_client.generate_response("fake-key", "Why?", context, history)

    messages = openai_client.return_value.chat.completions.create.call_args.kwargs[
        "messages"
    ]
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert "NASA mission-history assistant" in messages[0]["content"]
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert context in messages[-1]["content"]
    assert "Question:\nWhy?" in messages[-1]["content"]


@pytest.mark.unit
def test_generate_response_handles_empty_context_explicitly() -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = _completion(
            "I do not have enough evidence."
        )
        llm_client.generate_response("fake-key", "Question?", "", [])

    messages = openai_client.return_value.chat.completions.create.call_args.kwargs[
        "messages"
    ]
    user_message = messages[-1]["content"]
    assert "No retrieved NASA source excerpts were provided." in user_message


@pytest.mark.unit
def test_generate_response_rejects_missing_conversation_history() -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        with pytest.raises(Exception):
            llm_client.generate_response(
                "fake-key",
                "Question?",
                "Context",
                cast(List[ConversationTurn], None),
            )

    openai_client.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        ),
    ],
)
def test_generate_response_rejects_missing_or_none_completion_content(
    response: Any,
) -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = response

        with pytest.raises(ValueError, match="completion"):
            llm_client.generate_response("fake-key", "Question?", "Context", [])


@pytest.mark.unit
def test_generate_response_surfaces_openai_errors() -> None:
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.side_effect = RuntimeError(
            "provider unavailable"
        )

        with pytest.raises(RuntimeError, match="provider unavailable"):
            llm_client.generate_response("fake-key", "Question?", "Context", [])
