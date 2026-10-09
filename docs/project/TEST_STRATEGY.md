# Test Strategy for the NASA Intelligence RAG System

## 1. Objective

This project must be validated from the start, not only after the final interface is complete. The testing strategy is designed to:

- verify each component independently,
- validate integrations between retrieval, generation, and evaluation,
- check that the product answers real NASA-domain questions with factual grounding,
- ensure each new feature is driven by a concrete failing test before implementation.

The system is a Retrieval-Augmented Generation (RAG) application built around:

- `src/embedding_pipeline.py`
- `src/rag_client.py`
- `src/llm_client.py`
- `src/ragas_evaluator.py`
- `src/chat.py`

The goal is not just to make the app run, but to prove it solves the expected task reliably and repeatedly.

---

## 2. Testing Principles

1. Test-first development
   - Every feature starts with a failing test.
   - The implementation is only considered complete when the test passes.

2. Requirement-driven validation
   - Each user story or task must map to at least one acceptance test.
   - A feature is not accepted unless it has a test proving the desired behavior.

3. Layered validation
   - Unit tests validate logic in isolation.
   - Integration tests validate interactions among modules and external tools.
   - Evaluation tests validate answer quality.
   - End-to-end tests validate the user-visible workflow.

4. Regression prevention
   - New changes must not break existing behavior.
   - Re-run the targeted suite that covers the modified component and relevant integration paths.

5. Evidence over assumption
   - For answer quality, rely on factual checks, retrieval validation, and evaluation metrics rather than only “looks good” inspection.

---

## 3. Testing Frameworks and Tools

### Core test runner
- `pytest`
  - Primary framework for unit and integration tests.
  - Clear fixtures, parametrization, and assertions.
  - Easy to run in CI and locally.

### Coverage
- `pytest-cov`
  - Tracks line, branch, and function coverage.
  - Enforces a minimum target for the codebase.

### Mocking and isolation
- `unittest.mock`
  - Standard-library mocking for OpenAI API calls, Chroma operations, and file IO.
- `pytest-mock` or `monkeypatch`
  - Useful for injecting mocked dependencies in a clean, readable way.

### Data generation and edge-case testing
- `faker` (optional)
  - Generates realistic text and metadata for tests.
- `hypothesis` (optional, for advanced cases)
  - Useful for property-based tests around chunking, text processing, and filtering.

### RAG-specific evaluation
- `ragas`
  - Used to validate retrieval and answer quality with metrics such as:
    - faithfulness
    - answer relevance
    - context precision
    - context recall

### Frontend/UI smoke tests
- `playwright` or `streamlit` app smoke checks
  - For validating the chat interface and critical user journeys.
  - Useful for ensuring the application still works when the whole stack is assembled.

### Static quality
- `ruff` or `flake8`
  - Linting for import hygiene and code quality.
- `mypy` (optional but recommended)
  - Type safety for complex data structures and API responses.

---

## 4. Test Pyramid for This Project

The project should use a layered strategy similar to this:

- 70% unit tests
- 20% integration tests
- 8% evaluation/regression tests
- 2% end-to-end smoke tests

This keeps the system fast to validate while still checking real-world behavior.

### 4.1 Unit tests
Apply these to small, isolated pieces of logic:

- text chunking behavior
- metadata extraction from paths
- context formatting
- prompt construction
- response parsing
- score normalization
- filtering logic

### 4.2 Integration tests
Validate the interaction between components:

- embedding pipeline writes chunks and metadata to Chroma
- rag client retrieves relevant chunks from Chroma
- llm client packages retrieved context + conversation history correctly
- evaluator reads question, answer, and context and computes scores
- chat application integrates all services together with mocks

### 4.3 Evaluation tests
The RAG system is only useful if it answers well. This layer uses NASA-domain questions and expected behavior checks:

- the answer uses the retrieved context,
- the answer stays grounded in source documents,
- the answer is relevant to the user’s prompt,
- the retrieval step returns useful source chunks.

### 4.4 End-to-end tests
Exercise the main user flow:

- launch app,
- submit a question,
- retrieve context,
- generate answer,
- show score,
- verify the response and no crash occurs.

---

## 5. Tests to Write by Component

### 5.1 Embedding pipeline tests
Focus on data preparation and indexing quality.

Required tests:
- splits a document into chunks with a configurable size and overlap,
- preserves metadata like mission, file name, and source path,
- handles empty documents without crashing,
- skips or logs invalid input files,
- creates embeddings with batch processing correctly,
- stores documents in Chroma with stable IDs,
- gracefully handles failed embedding calls.

Example scenarios:
- single document with many paragraphs,
- empty text file,
- large transcript with repeated phrases,
- metadata missing or partial.

### 5.2 RAG client tests
Focus on retrieval quality and prompt context preparation.

Required tests:
- returns the top K most relevant chunks for a query,
- filters by mission or metadata when requested,
- formats retrieved chunks into LLM-ready context,
- handles no-match retrieval gracefully,
- sorts results by relevance score,
- rejects invalid query types,
- returns predictable structure for downstream code.

Example scenarios:
- question about Apollo 13 oxygen tank problems,
- query with spelling variations,
- database returns zero matches,
- metadata filter excludes unrelated missions.

### 5.3 LLM client tests
Focus on request construction and failure handling.

Required tests:
- sends system prompt and user question to the model,
- includes conversation history when requested,
- preserves chat order and role information,
- handles timeouts and API errors,
- trims or formats long prompts correctly,
- returns normalized response objects,
- does not leak raw internal context into user-facing output.

Example scenarios:
- first user turn,
- multi-turn conversation,
- OpenAI rate limit,
- invalid API key.

### 5.4 RAGAS evaluator tests
Focus on measurable answer quality.

Required tests:
- produces scalar or structured scores within expected bounds,
- flags unfaithful answers,
- detects irrelevant context,
- handles missing answer or missing context without crashing,
- returns consistent metrics for the same inputs,
- can compare multiple candidate answers for regression checks.

Example scenarios:
- supported answer grounded in retrieved context,
- hallucinated answer not supported by docs,
- answer unrelated to the question,
- no relevant chunks returned.

### 5.5 Chat application tests
Focus on the user experience and integration flow.

Required tests:
- question submission triggers retrieval and generation,
- answer is displayed in the UI,
- errors are surfaced clearly,
- empty input is rejected,
- loading and status states are handled,
- evaluation metrics are visible when available,
- the app remains responsive after a failed retrieval.

Example scenarios:
- valid question,
- invalid or blank question,
- retrieval failure,
- model failure, UI fallback state.

### 5.6 Local Ollama and OpenAI-compatible endpoint tests

The LLM client accepts an optional `openai_base_url`. Unit tests completely mock
the OpenAI SDK and verify that the URL is omitted for ordinary OpenAI calls and
forwarded unchanged when supplied. These tests must never make network requests.

The `tests/test_ollama_integration.py` test is an opt-in local integration test.
It creates a temporary ChromaDB collection with deterministic fixture embeddings,
retrieves a labeled NASA excerpt, and sends that context to a locally running
Ollama chat model through its OpenAI-compatible API.

Prerequisites:

1. Install and start Ollama independently of pytest.
2. Pull a locally available chat model, for example `ollama pull llama3.2`.
3. Set the model and, if needed, endpoint environment variables in PowerShell:

   ```powershell
   $env:OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"
   $env:OLLAMA_CHAT_MODEL = "llama3.2"
   ```

4. Run the Ollama check with required-service mode so an unavailable endpoint
   cannot be mistaken for a pass:

   ```powershell
   $env:NASA_RAG_REQUIRE_OLLAMA = "1"
   python -m pytest tests/test_ollama_integration.py -m integration
   ```

Without `NASA_RAG_REQUIRE_OLLAMA=1`, the Ollama test skips with the reason when
the model or local endpoint is unavailable. Chroma-only integration tests do not
need Ollama or an OpenAI key:

```powershell
python -m pytest -m unit
python -m pytest -m integration -k "not ollama"
python -m pytest
```

The Ollama test uses a non-secret placeholder API key (`ollama`). Never place a
real provider key in test source, test data, or checked-in configuration.

---

## 6. Test Structure and Organization

Use a clear project structure:

```text
/tests
  /unit
    test_embedding_pipeline.py
    test_rag_client.py
    test_llm_client.py
    test_ragas_evaluator.py
    test_chat_utils.py
  /integration
    test_embedding_to_chroma.py
    test_rag_end_to_end.py
    test_llm_integration.py
  /e2e
    test_chat_flow.py
  /fixtures
    nasa_sample_documents.py
    sample_queries.json
    expected_answers.json
  /golden
    apollo11_questions.json
    apollo13_questions.json
    challenger_questions.json
  conftest.py
  pytest.ini
```

### Recommended fixtures
- `sample_documents` for a minimal NASA dataset,
- `mock_chroma_client` for vector DB tests,
- `mock_openai_client` for LLM tests,
- `query_set` for retrieval evaluation tests,
- `chat_session_state` for Streamlit UI tests.

### Naming conventions
- test files should be named after the module under test,
- test names should describe behavior rather than implementation,
- place asserts on observable outputs, not internal state.

---

## 7. Acceptance Criteria for Each Feature

Before a feature is considered complete, it must satisfy all of the following:

1. There is a failing test proving the expected behavior.
2. The implementation passes that test.
3. Related unit and integration tests still pass.
4. Relevant evaluation questions still pass.
5. Coverage for the changed component is above the project threshold.
6. Edge cases have been tested.
7. If the feature changes user-visible behavior, a UI smoke test confirms it.

A practical rule:

- No feature is “done” without a requirement-to-test mapping.
- No feature is “done” without evidence that it solves the task.

---

## 8. Coverage Target and Quality Gates

A good baseline for this project is:

- minimum 80% overall line coverage,
- 75%+ branch coverage for critical logic,
- 100% coverage for safety-critical helper functions (error handling, input validation, retrieval formatting, prompt construction).

Recommended CI gate:

- Ruff linting for `src` and `tests` must pass,
- the pytest suite must pass, excluding the opt-in Ollama integration test
  because it requires an external local service and model,
- branch protection must require the **Lint** and **Tests** status checks on
  `master` and `dev`.

The GitHub Actions workflow runs this gate on pull requests targeting `master` or
`dev` and on pushes to those branches. The Ollama test remains available for
manual runs where its service prerequisites are installed.

---

## 9. Golden Dataset Strategy

The RAG system should be validated on a small benchmark dataset of curated NASA questions.

For each mission:

- create 10-20 high-value questions,
- record the expected answer or an acceptance criterion,
- test retrieval returns relevant evidence,
- test the final answer remains faithful to the source material.

Example dataset categories:
- factual recall,
- cause-and-effect reasoning,
- mission timeline questions,
- troubleshooting and engineering issues,
- comparison questions across missions.

This dataset becomes the regression guardrail for future feature work.

---

## 10. Suggested Execution Flow in Practice

For each feature or bug fix:

1. Write a failing test for the expected behavior.
2. Run the smallest relevant subset of tests.
3. Confirm the test fails for the right reason.
4. Implement the smallest fix.
5. Re-run the same test subset.
6. Run the relevant integration and evaluation checks.
7. Check coverage for the touched module.
8. Only then merge or continue to the next feature.

This creates a disciplined red-green-refactor loop that keeps the project stable while still moving fast.

---

## 11. Recommended First Sprint of Tests

Before building full functionality, the project should start with a minimal but meaningful test suite:

- chunking test for text documents,
- retrieval test for relevant document chunks,
- LLM request test for prompt correctness,
- evaluator test for faithfulness/relevance scoring,
- chat app smoke test for the end-to-end flow,
- a small golden set of NASA questions.

This creates a strong baseline from day one and ensures the system is being assessed from the beginning instead of only after implementation.

---

## 12. Final Recommendation

For this project, the most effective stack is:

- `pytest` for the test runner,
- `pytest-cov` for coverage,
- `unittest.mock` / `pytest-mock` for dependency isolation,
- `ragas` for answer and retrieval quality validation,
- `playwright` for critical UI smoke coverage.

The key is not simply to add tests at the end, but to structure them as a living quality system that checks functional correctness, retrieval quality, and user-visible behavior from the beginning of development.

This ensures that each feature development directly solves an expected task and is backed by evidence.
