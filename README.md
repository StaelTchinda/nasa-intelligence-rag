# NASA Intelligence RAG

A Streamlit chat application that uses retrieval-augmented generation to answer
questions about NASA missions from a local ChromaDB collection.

## Features

- Retrieves source passages from indexed NASA mission documents.
- Grounds chat responses in the retrieved context and shows source references.
- Supports OpenAI and OpenAI-compatible chat and embedding endpoints.
- Lets you choose among available local ChromaDB collections.

## Quick start

Prerequisites: Python 3.12 and an API-compatible chat and embedding provider.
The default setup uses OpenAI; local Ollama setup is described in the
[provider setup guide](./docs/SETUP.md).

From the repository root, create a virtual environment and install dependencies
(PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Create your local environment file from the template, then add your provider
credentials:

```powershell
Copy-Item .env.example .env
```

Only run this when you do not already have a `.env` file, so existing local
settings are not overwritten.

Configure your provider as described under [Configuration](#configuration),
then index the mission documents as described under
[Indexing data](#indexing-data). Start the chat application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/chat.py
```

## Configuration

Settings can be supplied in a root `.env` file or as process environment
variables. Process environment variables take precedence. Do not commit real
API keys.

Use the checked-in [`.env.example`](./.env.example) as the template for your
local `.env` file. Replace its OpenAI placeholder with your own key, or
uncomment and fill in the optional OpenAI-compatible/Ollama settings as needed.
Keep `.env` private; it is ignored by Git.

<!-- AUTO-GENERATED: Supported environment variables are sourced from src/config/chat_config.py and src/rag_client.py. -->

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | For native OpenAI chat | OpenAI chat key; also the default key for retrieval embeddings |
| `OPENAI_CHAT_MODEL` | No | Initial OpenAI chat model (default: `gpt-3.5-turbo`) |
| `OPENAI_BASE_URL` | No | Base URL for a generic OpenAI-compatible chat endpoint |
| `OLLAMA_OPENAI_BASE_URL` | No | Ollama-compatible chat endpoint; selects compatible-provider mode |
| `OLLAMA_CHAT_MODEL` | No | Initial Ollama chat model (default: `llama3.2`) |
| `OLLAMA_API_KEY` | No | Key for an OpenAI-compatible chat endpoint (defaults to `ollama` if omitted) |
| `CHROMA_OPENAI_BASE_URL` | No | Custom OpenAI-compatible endpoint for retrieval embeddings |
| `CHROMA_OPENAI_API_KEY` | No | Retrieval embedding key (falls back to `OPENAI_API_KEY`) |
<!-- END AUTO-GENERATED -->

Set the embedding endpoint and key independently when the collection uses a
custom embedding provider. The embedding model used to query a collection must
match the model that created its stored vectors. See the
[provider setup guide](./docs/SETUP.md) for OpenAI-compatible and Ollama
examples.

## Usage

Launch the Streamlit app, select a ChromaDB collection in the sidebar, choose
the chat and retrieval settings, then ask a question. The app discovers
collections in immediate project-root directories whose names start with
`chroma_db`. Retrieval and generation progress are shown separately; if
retrieval fails, the app reports the error rather than sending an ungrounded
question to the chat model.

## Indexing data

The repository includes NASA mission text under `data/text`. To index it with
OpenAI embeddings, set `OPENAI_API_KEY` in the environment and run (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m src.embedding_pipeline `
  --data-path . `
  --openai-key $env:OPENAI_API_KEY `
  --chroma-dir .\chroma_db_openai `
  --collection-name nasa_space_missions_text
```

The default embedding model is `text-embedding-3-small`. For Ollama or another
compatible embedding provider, supply `--openai-base-url` and select a model
supported by that endpoint. See the [provider setup guide](./docs/SETUP.md) for
a complete Ollama example. The embedding model must remain consistent between
indexing and querying the collection.

## Development and testing

Run the same checks as CI:

```powershell
python -m ruff check src tests
python -m pytest -k "not ollama"
```

<!-- AUTO-GENERATED: CI checks mirror .github/workflows/ci.yml. -->
GitHub Actions runs Ruff and pytest on pull requests and pushes targeting
`master` and `dev`. The Ollama integration test is excluded from CI because it
requires a running Ollama service and model. Configure branch protection for
both branches to require the **Lint** and **Tests** checks.
<!-- END AUTO-GENERATED -->

See the [test strategy](./docs/project/TEST_STRATEGY.md) for test types and
instructions for running the opt-in Ollama integration test.

## Project documentation

- [Provider setup guide](./docs/SETUP.md)
- [Test strategy](./docs/project/TEST_STRATEGY.md)
- [General project instructions](./docs/project/GENERAL_INSTRUCTIONS.md)
- [Code instructions](./docs/project/CODE_INSTRUCTIONS.md)

## Security

Never commit real provider API keys or credentials. Keep local values in an
untracked `.env` file or inject them through your environment. The literal
`ollama` key shown for a local Ollama endpoint is a placeholder for the
OpenAI-compatible client, not a secret.
