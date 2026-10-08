#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

from collections.abc import Mapping, Sequence
from time import monotonic
from typing import Callable, Optional, cast, List

import chromadb
from chromadb.api.models import Collection as chromadb_collection
import chromadb.api.types as chromadb_types
from openai import APIError
import streamlit as st

import src.ragas_evaluator as ragas_evaluator
import src.rag_client as rag_client
import src.llm_client as llm_client
from src.config.chat_config import (
    COMPATIBLE_CHAT_MODELS,
    CUSTOM_MODEL_OPTION,
    OPENAI_CHAT_MODELS,
    ChatProvider,
    build_chat_settings,
    build_embedding_settings,
    load_chat_defaults,
)

from src.rag_types import (
    ChromaBackend,
    ConversationTurn,
)

# RAGAS imports
try:
    import ragas

    RAGAS_AVAILABLE = ragas is not None
except ImportError:
    RAGAS_AVAILABLE = False
    st.warning("RAGAS not available. Install with: pip install ragas")

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

def discover_chroma_backends() -> dict[str, ChromaBackend]:
    """Discover available ChromaDB backends in the project directory"""

    return rag_client.discover_chroma_backends()

#@st.cache_resource
def initialize_rag_system(
    chroma_dir: str, collection_name: str
) -> tuple[Optional[chromadb_collection.Collection], bool, Optional[str]]:
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
       return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(
    collection: chromadb_collection.Collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> Optional[chromadb.QueryResult]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        embedding_options: dict[str, str] = {}
        if openai_key is not None:
            embedding_options["openai_key"] = openai_key
        if openai_base_url is not None:
            embedding_options["openai_base_url"] = openai_base_url
        return rag_client.retrieve_documents(
            collection,
            query,
            n_results,
            mission_filter,
            **embedding_options,
        )
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[chromadb_types.Metadata]) -> str:
    """Format retrieved documents into context"""
    
    return rag_client.format_context(documents, metadatas)

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[ConversationTurn],
    model: str = "gpt-3.5-turbo",
    openai_base_url: Optional[str] = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Generate response using OpenAI with context"""
    if openai_base_url is None and on_progress is None:
        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model,
        )
    if openai_base_url is None:
        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model,
            on_progress=on_progress,
        )
    if on_progress is None:
        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model,
            openai_base_url=openai_base_url,
        )
    return llm_client.generate_response(
        openai_key,
        user_message,
        context,
        conversation_history,
        model,
        openai_base_url=openai_base_url,
        on_progress=on_progress,
    )

def evaluate_response_quality(
    question: str, answer: str, contexts: Sequence[str]
) -> dict[str, float | str]:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def display_evaluation_metrics(scores: Mapping[str, float | str]) -> None:
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return
    
    st.sidebar.subheader("📊 Response Quality")
    
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{score:.3f}",
                delta=None
            )
            
            # Add progress bar
            st.sidebar.progress(score)

def main() -> None:
    defaults = load_chat_defaults()
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    messages = cast(list[ConversationTurn], st.session_state.messages)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Discover available backends
        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()
        
        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n`python run_text_embedding.py`")
            st.stop()
        
        # Backend selection
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )
        
        selected_backend = available_backends[selected_backend_key]
        
        st.subheader("Chat Provider Settings")
        provider = st.selectbox(
            "Chat provider",
            options=list(ChatProvider),
            index=list(ChatProvider).index(defaults.provider),
            format_func=lambda value: value.value,
            help="Choose native OpenAI or an OpenAI-compatible service such as Ollama",
            key="chat_provider",
        )

        if provider is ChatProvider.OPENAI:
            chat_base_url = ""
            chat_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=defaults.openai_api_key,
                help="Required for the native OpenAI API",
                key="openai_chat_api_key",
            )
            model_options = (*OPENAI_CHAT_MODELS, CUSTOM_MODEL_OPTION)
        else:
            chat_base_url = st.text_input(
                "Chat API base URL",
                value=defaults.chat_base_url or "",
                help="An OpenAI-compatible HTTP(S) API base URL",
                key="compatible_chat_base_url",
            )
            chat_api_key = st.text_input(
                "Chat API key",
                type="password",
                value=defaults.compatible_api_key,
                help="Use the provider key, or leave as `ollama` for local Ollama",
                key="compatible_chat_api_key",
            )
            model_options = (*COMPATIBLE_CHAT_MODELS, CUSTOM_MODEL_OPTION)

        configured_model = defaults.chat_model
        default_model_choice = (
            configured_model
            if configured_model in model_options
            else CUSTOM_MODEL_OPTION
        )
        selected_model = st.selectbox(
            "Chat model",
            options=list(model_options),
            index=list(model_options).index(default_model_choice),
            help="Choose a preset or enter a provider-specific model ID",
            key=f"chat_model_choice_{provider.value}",
        )
        if selected_model == CUSTOM_MODEL_OPTION:
            model_choice = st.text_input(
                "Custom chat model ID",
                value=(
                    ""
                    if configured_model in model_options
                    else configured_model
                ),
                help="Enter the exact model identifier supported by the selected endpoint",
                key=f"custom_chat_model_{provider.value}",
            )
        else:
            model_choice = selected_model

        try:
            chat_settings = build_chat_settings(
                provider,
                chat_api_key,
                chat_base_url,
                model_choice,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

        st.subheader("Retrieval Embedding Settings")
        embedding_base_url = st.text_input(
            "Embedding API base URL",
            value=defaults.embedding_base_url or "",
            help="Independent from chat; blank uses the default OpenAI API endpoint",
            key="embedding_base_url",
        )
        embedding_api_key = st.text_input(
            "Embedding API key",
            type="password",
            value=defaults.embedding_api_key or "",
            help=(
                "Uses CHROMA_OPENAI_API_KEY or OPENAI_API_KEY by default. "
                "For local providers, a placeholder such as `ollama` is sufficient."
            ),
            key="embedding_api_key",
        )
        try:
            embedding_settings = build_embedding_settings(
                embedding_api_key
                or (
                    chat_settings.api_key
                    if provider is ChatProvider.OPENAI
                    else ""
                ),
                embedding_base_url,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)
        
        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE)
        
        # Initialize RAG system when backend changes
        if (st.session_state.current_backend != selected_backend_key):
            st.session_state.current_backend = selected_backend_key
            # Clear cache to force reinitialization
            st.cache_resource.clear()
    
    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):

        collection, success, error = initialize_rag_system(
            selected_backend["directory"], 
            selected_backend["collection_name"]
        )
    
    if not success or collection is None:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    collection_embedding_model = (
        (collection.metadata or {}).get("embedding_model")
        or rag_client.DEFAULT_EMBEDDING_MODEL
    )
    st.sidebar.caption(
        f"Collection query embedding model: `{collection_embedding_model}`"
    )

    # Display evaluation metrics if available
    last_evaluation = st.session_state.last_evaluation
    if last_evaluation and enable_evaluation:
        display_evaluation_metrics(cast(Mapping[str, float | str], last_evaluation))
    
    # Display chat messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about NASA space missions..."):
        # Add user message to chat history
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        response: str | None = None
        contexts_list: list[str] = []
        with st.chat_message("assistant"):
            with st.status(
                "Creating query embedding and searching documents...",
                expanded=False,
            ) as request_status:
                docs_result = retrieve_documents(
                    collection,
                    prompt,
                    n_docs,
                    openai_key=embedding_settings.api_key,
                    openai_base_url=embedding_settings.base_url,
                )

                if docs_result is None:
                    request_status.update(
                        label="Document retrieval failed",
                        state="error",
                    )
                else:
                    context = ""
                    if docs_result["documents"] and docs_result["metadatas"]:
                        retrieved_documents = docs_result["documents"][0]
                        retrieved_metadatas = docs_result["metadatas"][0]
                        context = format_context(
                            retrieved_documents, retrieved_metadatas
                        )
                        contexts_list = [
                            document
                            for document in retrieved_documents
                            if document is not None
                        ]
                        st.session_state.last_contexts = contexts_list

                    request_status.update(
                        label=f"Generating response with {chat_settings.model}...",
                        state="running",
                    )

                    last_progress_update = 0.0
                    last_progress_phase: str | None = None

                    def update_generation_progress(phase: str) -> None:
                        nonlocal last_progress_update, last_progress_phase
                        now = monotonic()
                        if (
                            phase == last_progress_phase
                            and now - last_progress_update < 1.0
                        ):
                            return
                        label = (
                            f"{chat_settings.model} is reasoning; the answer "
                            "will appear when ready..."
                            if phase == "thinking"
                            else f"Receiving answer from {chat_settings.model}..."
                        )
                        request_status.update(label=label, state="running")
                        last_progress_phase = phase
                        last_progress_update = now

                    try:
                        response = generate_response(
                            chat_settings.api_key,
                            prompt,
                            context,
                            messages[:-1],
                            chat_settings.model,
                            openai_base_url=chat_settings.base_url,
                            on_progress=update_generation_progress,
                        )
                    except APIError as error:
                        request_status.update(
                            label="Chat provider request failed",
                            state="error",
                        )
                        st.error(
                            "The chat provider did not complete the request. "
                            f"{type(error).__name__}: {error}"
                        )
                    else:
                        request_status.update(
                            label="Response generated",
                            state="complete",
                        )
                        st.markdown(response)

            if response is not None and enable_evaluation and RAGAS_AVAILABLE:
                with st.spinner("Evaluating response quality..."):
                    evaluation_scores = evaluate_response_quality(
                        prompt,
                        response,
                        contexts_list,
                    )
                    st.session_state.last_evaluation = evaluation_scores

        if response is not None:
            messages.append({"role": "assistant", "content": response})
            st.rerun()


if __name__ == "__main__":
    main()
