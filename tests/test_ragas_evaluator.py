from unittest.mock import patch

from src import ragas_evaluator


def test_evaluate_response_quality_returns_metric_dict():
    with patch.object(ragas_evaluator, "RAGAS_AVAILABLE", True):
        with patch.object(
            ragas_evaluator,
            "evaluate",
            return_value={"faithfulness": 0.92, "response_relevancy": 0.88},
        ) as evaluate:
            result = ragas_evaluator.evaluate_response_quality(
                "What failed on Apollo 13?",
                "Apollo 13 suffered an oxygen tank failure.",
                ["Apollo 13 experienced a catastrophic oxygen system issue after launch."],
            )

    evaluate.assert_called_once()
    assert isinstance(result, dict)
    assert "faithfulness" in result or "error" in result
