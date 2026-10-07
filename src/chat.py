#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Optional, cast, List

import chromadb
from chromadb.api.models import Collection as chromadb_collection
import chromadb.api.types as chromadb_types
import streamlit as st

import src.ragas_evaluator as ragas_evaluator
import src.rag_client as rag_client
import src.llm_client as llm_client

from src.rag_types import (
    ChromaBackend,
    ChromaCollection,
    ConversationTurn,
    Metadata,
    RetrievalResult,
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
) -> Optional[chromadb.QueryResult]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
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
) -> str:
    """Generate response using OpenAI with context"""
    return llm_client.generate_response(
        openai_key,
        user_message,
        context,
        conversation_history,
        model,
        openai_base_url=openai_base_url,
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
        
        # API Key input
        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        else:
            os.environ["CHROMA_OPENAI_API_KEY"] = openai_key
        
        # Model selection
        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Choose the OpenAI model for responses"
        )
        
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
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Retrieve relevant documents
                docs_result = retrieve_documents(
                    collection, 
                    prompt, 
                    n_docs
                )
                
                # Format context
                context = ""
                contexts_list: list[str] = []
                if docs_result and docs_result["documents"] and docs_result["metadatas"]:
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
                
                # Generate response
                response = generate_response(
                    openai_key, 
                    prompt, 
                    context, 
                    messages[:-1],
                    model_choice
                )
                st.markdown(response)
                
                # Evaluate response quality if enabled
                if enable_evaluation and RAGAS_AVAILABLE:
                    with st.spinner("Evaluating response quality..."):
                        evaluation_scores = evaluate_response_quality(
                            prompt, 
                            response, 
                            contexts_list
                        )
                        st.session_state.last_evaluation = evaluation_scores
        
        # Add assistant response to chat history
        messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
