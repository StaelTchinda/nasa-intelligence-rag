# Phase 2: Evaluation and Interface — Construction Plan

## Problem and proposed approach

The repository has completed the Phase 1 infrastructure (LLM client, retrieval client, and embedding pipeline), but Phase 2 remains open: the system needs a working RAGAS evaluation layer and a Streamlit-based chat interface that ties the retrieval, LLM, and scoring pieces together. The plan is to implement the evaluator first as a self-contained, dependency-safe scoring module, then wire the Streamlit chat app to the existing backend and evaluation functions, and finally verify behavior with targeted checks that avoid live provider dependence unless a separate opt-in run is approved.

## Todo plan

1. Audit the current Phase 2 integration points and constraints.
   - Confirm the public APIs in `src/ragas_evaluator.py` and `src/chat.py`.
   - Check how conversation history, retrieved contexts, and model configuration flow into the app.
   - Identify compatibility constraints such as the RAGAS/LangChain version pin and the current ChromaDB/LLM setup.

2. Implement the RAGAS evaluator.
   - Add the evaluator LLM and embedding wrapper setup with the project’s supported OpenAI-compatible configuration.
   - Define the evaluation metrics that match the project goals (for example relevance and faithfulness, plus any additional metrics already in the codebase).
   - Validate that the evaluator handles error cases cleanly and returns structured score dictionaries instead of silently failing.

3. Build the interactive chat application.
   - Create the Streamlit UI for mission selection, backend selection, model choice, and conversation history.
   - Connect chat inputs to the retrieval flow, LLM generation, and evaluation pipeline.
   - Ensure the app surfaces both answers and evaluation scores clearly in the interface without breaking the Phase 1 provider abstractions.

4. Run focused verification and polish.
   - Exercise the evaluator end-to-end with deterministic inputs.
   - Confirm the chat app still works with the existing wrappers and no longer bypasses config passing.
   - Check error handling for missing dependencies, empty context, invalid inputs, and provider failures.

## Notes and considerations

- The repository already includes `RAGAS_AVAILABLE` checks and a partial `src/chat.py` wrapper. The implementation should extend those patterns rather than replacing them wholesale.
- The RAGAS integration must respect the project’s environment guidance: keep compatibility with the pinned library versions and avoid introducing new heavy dependencies.
- The app should preserve the current user experience while exposing only the minimal configuration that is necessary for the mission-specific chat flow.
- Testing should stay local and deterministic. No live OpenAI call should be required for routine verification, and any live-provider validation should be clearly separated and opt-in.
- The final phase should prove that the evaluation and UI work together cohesively: retrieval results feed the LLM, the answer is scored, and the metrics are visible to the user without blocking the conversation.
