from unittest.mock import patch

from src import chat


def test_chat_wrapper_can_generate_and_evaluate():
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
                [{"role": "user", "content": "Tell me about Apollo 13."}],
            )
            evaluation = chat.evaluate_response_quality(
                "What happened on Apollo 13?",
                "Mock answer about Apollo 13",
                ["Apollo 13 lost oxygen after launch."],
            )

    generate_response.assert_called_once()
    evaluate_response_quality.assert_called_once()

    assert response == "Mock answer about Apollo 13"
    assert evaluation["faithfulness"] >= 0.9
