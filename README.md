# NASA Intelligence RAG

A Streamlit chat application that uses retrieval-augmented generation to answer
questions about NASA missions from a local ChromaDB collection.

## Features

- Retrieves source passages from indexed NASA mission documents.
- Grounds chat responses in the retrieved context and shows source references.
- Supports OpenAI and OpenAI-compatible chat and embedding endpoints.
- Lets you choose among available local ChromaDB collections.

## Quick start

Prerequisites: Python 3.12. Choose either **online** (OpenAI-hosted chat and
embeddings) or **local/offline inference** (Ollama running on your machine).
The local option requires downloading Ollama and its models first; after that,
chat and embedding requests run locally.

From the repository root, create a virtual environment and install
dependencies. These commands use POSIX shell syntax and work in Linux and Git
Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

On Windows, use Git Bash and activate the environment with
`source .venv/Scripts/activate`. If `python3` is not available in Git Bash, use
`python` to create the environment. Copy the environment template if you do
not already have a local `.env`:

```bash
cp .env.example .env
```

Configure the chosen provider in `.env`. Before indexing, export those settings
into the current shell:

```bash
set -a
source .env
set +a
```

Then index the mission documents as described under
[Indexing data](#indexing-data), and start the app:

```bash
python -m streamlit run src/chat.py
```

## Configuration

Choose and configure one of these provider setups in `.env`:

### Online setup (OpenAI)

Use OpenAI-hosted chat and embedding APIs. This requires internet access and an
OpenAI API key:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_CHAT_MODEL=gpt-3.5-turbo
```

### Local/offline setup (Ollama)

Install Ollama and download the models before using the app offline. Then
configure Ollama for both chat and retrieval embeddings:

```dotenv
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_API_KEY=ollama
CHROMA_OPENAI_BASE_URL=http://localhost:11434/v1
CHROMA_OPENAI_API_KEY=ollama
```

A starter template with both sets of settings is provided in
[`.env.example`](./.env.example). See the [provider setup guide](./docs/SETUP.md)
for installation and indexing steps. Keep `.env` private; it is ignored by
Git. In the app, process environment variables take precedence over `.env`.

## Usage

Launch the Streamlit app, select a ChromaDB collection in the sidebar, choose
the chat and retrieval settings, then ask a question. The app discovers
collections in immediate project-root directories whose names start with
`chroma_db`. Retrieval and generation progress are shown separately; if
retrieval fails, the app reports the error rather than sending an ungrounded
question to the chat model.

## Indexing data

The repository includes NASA mission text under `data/text`. For online
indexing, set `OPENAI_API_KEY` in your shell and run:

```bash
python -m src.embedding_pipeline \
  --data-path . \
  --openai-key "$OPENAI_API_KEY" \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text
```

For local Ollama indexing, see the [offline setup steps](./docs/SETUP.md).
The embedding model used to query a collection must match the model used to
create its stored vectors.

## Development and testing

Run the same checks as CI:

```bash
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
