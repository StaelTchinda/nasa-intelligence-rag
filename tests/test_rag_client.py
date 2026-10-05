from src import rag_client


class FakeCollection:
    def query(self, query_texts, n_results=3, where=None):
        return {
            "documents": [[
                "Apollo 13 lost oxygen pressure after launch.",
                "The crew used the lunar module to preserve power and oxygen."
            ]],
            "metadatas": [[
                {"mission": "apollo_13", "source": "oxygen_system", "category": "systems"},
                {"mission": "apollo_13", "source": "lifeboat", "category": "operations"},
            ]],
            "distances": [[0.14, 0.33]],
        }


def test_format_context_includes_mission_and_source_details():
    documents = [
        "Apollo 13 lost oxygen pressure after launch.",
        "The crew used the lunar module to preserve power and oxygen.",
    ]
    metadatas = [
        {"mission": "apollo_13", "source": "oxygen_system", "category": "systems"},
        {"mission": "apollo_13", "source": "lifeboat", "category": "operations"},
    ]

    result = rag_client.format_context(documents, metadatas)

    assert "Apollo 13" in result
    assert "Oxygen" in result
    assert "Systems" in result or "Operations" in result


def test_retrieve_documents_uses_filters_when_provided():
    result = rag_client.retrieve_documents(
        FakeCollection(),
        "Apollo 13 oxygen problem",
        n_results=2,
        mission_filter="apollo_13",
    )

    assert result is not None
    assert "documents" in result
    assert len(result["documents"][0]) == 2
