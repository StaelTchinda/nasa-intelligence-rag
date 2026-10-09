# Provider setup

The application uses separate settings for chat completions and retrieval
embeddings. They may use the same provider, but the embedding model must match
the model used to create the selected ChromaDB collection.

Store local configuration in the repository-root `.env` file. Process
environment variables take precedence over values in that file. Do not commit
real provider credentials.

## OpenAI

For chat and retrieval using OpenAI:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_CHAT_MODEL=gpt-3.5-turbo
```

The embedding pipeline defaults to `text-embedding-3-small`. Chat and embedding
models are configured independently.

## Generic OpenAI-compatible chat endpoint

For a compatible chat service, configure its base URL and API key:

```dotenv
OPENAI_BASE_URL=https://provider.example/v1
OPENAI_API_KEY=your-provider-api-key
OPENAI_CHAT_MODEL=your-chat-model
```

Replace the example URL and model with values supported by your provider.

## Ollama chat and embeddings

Install and start Ollama, then pull a chat model and an embedding model. For
example:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Configure both endpoints:

```dotenv
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_API_KEY=ollama
CHROMA_OPENAI_BASE_URL=http://localhost:11434/v1
CHROMA_OPENAI_API_KEY=ollama
```

Index mission documents into a separate collection:

```powershell
.\.venv\Scripts\python.exe -m src.embedding_pipeline `
  --data-path . `
  --openai-key ollama `
  --openai-base-url http://localhost:11434/v1 `
  --chroma-dir .\chroma_db_ollama `
  --collection-name nasa_space_missions_ollama `
  --openai-embedding-model nomic-embed-text
```

Start the Streamlit application and select `nasa_space_missions_ollama` from
the sidebar. Keep the collection paired with the embedding model used to index
it; query embeddings from a different model are not compatible with its stored
vectors.

The `ollama` key is a non-secret placeholder required by the OpenAI-compatible
client. Use an actual key if your compatible endpoint requires authentication.
