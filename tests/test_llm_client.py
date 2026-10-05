from types import SimpleNamespace
from unittest.mock import patch

from src import llm_client


def test_generate_response_returns_string_and_uses_context():
    with patch.object(llm_client, "OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Apollo 13 suffered an oxygen tank failure."
                    )
                )
            ]
        )
        response = llm_client.generate_response(
            "fake-key",
            "What happened on Apollo 13?",
            "Apollo 13 encountered an oxygen system issue after launch.",
            [{"role": "user", "content": "Tell me about Apollo 13."}],
            model="gpt-3.5-turbo",
        )

    openai_client.assert_called_once_with(api_key="fake-key")
    assert isinstance(response, str)
    assert "Apollo 13" in response
