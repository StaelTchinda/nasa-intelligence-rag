# Setup: online and local/offline options

Choose one provider setup:

- **Online:** OpenAI provides chat completions and embeddings. Requests require
  internet access and an OpenAI API key.
- **Local/offline inference:** Ollama provides chat completions and embeddings
  on your computer. Ollama and the selected models must be downloaded before
  using this setup offline; no OpenAI service or API key is needed afterward.

The shell examples use POSIX syntax and work on Linux and in Git Bash on
Windows. In Git Bash, a Windows-created virtual environment uses
`.venv/Scripts/activate`; on Linux use `.venv/bin/activate`.

## Common installation

From the repository root, create and activate a virtual environment and install
the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

For Git Bash on Windows, replace the activation command with:

```bash
source .venv/Scripts/activate
```

Copy the checked-in environment template once; skip this if `.env` already
exists so local settings are not overwritten:

```bash
cp .env.example .env
```

After editing `.env` for the selected provider, load its values into the
current shell before running the indexing pipeline:

```bash
set -a
source .env
set +a
```

The application also loads `.env` itself. Process environment variables take
precedence over values in the file. Never commit real credentials.

## Online setup: OpenAI

In `.env`, set the OpenAI key and optional chat model:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_CHAT_MODEL=gpt-3.5-turbo
```

Index the mission documents using OpenAI embeddings:

```bash
python -m src.embedding_pipeline \
  --data-path . \
  --openai-key "$OPENAI_API_KEY" \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text
```

The embedding pipeline defaults to `text-embedding-3-small`. Start the app and
select `nasa_space_missions_text`:

```bash
python -m streamlit run src/chat.py
```

This option sends chat and embedding requests to online APIs. Keep your API key
private and monitor provider usage and billing.

## Local/offline setup: Ollama

Install and start Ollama while you have internet access. Download the chat and
embedding models you want to use; these models remain available locally for
offline inference:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

In `.env`, enable local Ollama for chat and embeddings:

```dotenv
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_API_KEY=ollama
CHROMA_OPENAI_BASE_URL=http://localhost:11434/v1
CHROMA_OPENAI_API_KEY=ollama
```

The `ollama` value is a placeholder key required by the OpenAI-compatible
client; it is not a real credential. Index mission documents into a separate
local collection:

```bash
python -m src.embedding_pipeline \
  --data-path . \
  --openai-key ollama \
  --openai-base-url http://localhost:11434/v1 \
  --chroma-dir ./chroma_db_ollama \
  --collection-name nasa_space_missions_ollama \
  --openai-embedding-model nomic-embed-text
```

Start Ollama locally and launch the app:

```bash
python -m streamlit run src/chat.py
```

Select `nasa_space_missions_ollama` in the app. Keep the collection paired
with the embedding model that indexed it; query vectors from a different model
are incompatible. Once dependencies and Ollama models are installed, inference
uses the local Ollama service rather than an online model provider.

## Generic OpenAI-compatible endpoint

The app also supports a remote OpenAI-compatible chat service. Configure
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_CHAT_MODEL` in `.env` with
values supplied by that service. Retrieval embeddings use OpenAI by default;
set `CHROMA_OPENAI_BASE_URL` and `CHROMA_OPENAI_API_KEY` as well if embeddings
are provided by a separate compatible endpoint.
