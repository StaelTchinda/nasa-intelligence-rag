# Project: NASA Mission Intelligence: Developing a RAG-Based Chat System

---

## Project Overview

### Your Mission: Build a NASA Intelligence Chat System

For this project, you'll step into the shoes of a NASA mission operations specialist. Your task is to build a Q&A system that can answer questions about some of NASA's most historic space missions. You'll be working with actual mission transcripts and technical documents from Apollo 11, Apollo 13, and the Challenger missions.

The goal is to create a tool that allows astronauts, researchers, or even a curious historian to ask a question in plain English—like "What problems did Apollo 13 encounter?"—and get an accurate, detailed answer sourced directly from NASA's own archives.

To do this, you are going to build a complete **Retrieval-Augmented Generation (RAG)** system.

### The Project Blueprint

You'll be working through a series of Python files, each with a specific job.

Here’s a high-level look at what you'll be building, piece by piece:

1. **The Embedding Pipeline** First, you'll take all those raw NASA text files and process them. You'll write code to break them into smaller, manageable chunks and then convert those chunks into numerical representations—or embeddings and store them in **ChromaDB**.
2. **The RAG Client** This is the core of your retrieval system. You'll build the logic that takes a user's question, searches the ChromaDB database to find the most relevant document chunks, and then formats that information neatly to be used as context.
3. **The LLM Client** Here, you'll connect to the OpenAI API. This component will take the user's question and the context from your RAG client and generate a helpful, human-readable answer.
4. **The RAGAS Evaluator** How do you know if your RAG system is any good? You'll implement a real-time evaluation system using RAGAS.
5. **The Chat Application** Finally, you'll bring everything together in an interactive chat interface using Streamlit.

---

## Environment Setup

### Project Environment

Here’s a breakdown of the technical environment you’ll be using in this project.

#### Programming Language
- **Python**: All of the code you'll write for this project will be in Python.

#### Core Libraries and SDKs

You'll be using several key software development kits (SDKs) to interact with the different services and components of your RAG system:

- **OpenAI Python SDK**: This is the official library for interacting with the OpenAI API. You'll use it to access the embedding models (for converting text to vectors) and the chat completion models that will generate the final answers.
- **Chroma SDK**: This library allows you to work with ChromaDB, the open-source vector database you'll use to store and query your document embeddings. You'll use it to build your document index and perform semantic searches.
- **RAGAS SDK**: This is the toolkit you'll use to evaluate the performance of your RAG system. It provides the functions needed to calculate metrics like faithfulness, answer relevancy, and context precision.

### Prerequisites

Before you start, make sure you have the following:

- Python 3.10 or a more recent version installed.
- An active OpenAI API key. Instructions for accessing an OpenAI Key with a budget are on the following pages.

### Installation Steps :

**Install the Required Libraries:** All the Python libraries you need are listed in the requirements.txt file. You can install them all with a single command using pip.

`pip install -r requirements.txt`

---

## Instructions

Here's a step-by-step implemetation path you can follow.

### Phase 1: Core Infrastructure

This first phase is all about building the foundational components of your RAG system.

#### 1. Implement the LLM Client (`llm_client.py`)

- Here, you'll connect to the OpenAI API. You'll learn how to write a good system prompt to give the AI its "persona" as a NASA expert. This component will take the user's question and the context from your RAG client and generate a helpful, human-readable answer.
- **What You'll Do:** Your first step is to create a connection to the OpenAI API. You'll write the code to send a user's question, along with any relevant context and conversation history, to an LLM like GPT-3.5 or GPT-4.
- **Tasks:**
  - Define a system prompt that tells the model to act as a NASA expert.
  - Manage conversation history so the model can remember previous turns.
  - Write the function that sends the request to OpenAI and returns the model's response.

#### 2. Build the RAG Client (`rag_client.py`)

- This is the core of your retrieval system. You'll build the logic that takes a user's question, searches the ChromaDB database to find the most relevant document chunks, and then formats that information neatly to be used as context.
- **What You'll Do:** Next, you'll build the "retrieval" part of the RAG system. This component is responsible for searching the vector database to find the most relevant documents to answer a user's question.
- **Tasks:**
  - Connect to the ChromaDB backend.
  - Implement the semantic search function that takes a question and finds the best matching document chunks.
  - Format the retrieved documents into a clean context string that can be passed to the LLM.

#### 3. Create the Embedding Pipeline (`embedding_pipeline.py`)

- You'll take all those raw NASA text files and process them. You'll write code to break them into smaller, manageable chunks and then convert those chunks into numerical representations—or embeddings—using an OpenAI model. You will then store all of this in a specialized vector database called **ChromaDB**. This is the foundation of your system's "memory."
- **What You'll Do:** This is the most extensive part of the setup. You'll write a script that takes all the raw NASA .txt files, processes them, creates embeddings, and saves them into the ChromaDB database.
- **Tasks:**
  - Implement a text chunking strategy to break large documents into smaller, more manageable pieces.
  - Use the OpenAI API to generate embeddings for each text chunk.
  - Manage the creation and population of collections within ChromaDB.
  - Build a command-line interface so you can easily run this pipeline from your terminal.

### Phase 2: Evaluation and Interface

Once the core system is built, you'll focus on evaluating its performance and creating a user-friendly interface.

#### 4. Develop the RAGAS Evaluator (`ragas_evaluator.py`)

- How do you know if your RAG system is any good? You'll implement a real-time evaluation system using a framework called **RAGAS**. This will automatically score your system's answers on metrics like faithfulness (Is it sticking to the facts?) and relevancy (Is the answer actually helpful?).
- **What You'll Do:** You'll build the system that scores how good your RAG system's answers are. This will give you real-time feedback on your system's quality.
- **Key Tasks:**
  - Integrate the RAGAS framework.
  - Define the evaluation metrics you want to use, such as answer relevancy and faithfulness.
  - Write the function that takes a question, an answer, and the context and returns a set of quality scores.

#### 5. Build the Chat Application (chat.py)

- Finally, you'll bring everything together in an interactive chat interface using Streamlit. This is where your project comes to life! Users will be able to select different models, choose a mission to focus on, and see the evaluation scores for each answer in real-time.
- **What You'll Do:** You'll bring everything together into a single, interactive web application using Streamlit.
- **Tasks:**
  - Build the chat interface where a user can type in questions.
  - Integrate all the components you've built: the RAG client, the LLM client, and the RAGAS evaluator.
  - Display the AI's answer and its real-time quality scores in the interface.

### Submission Instructions

When you are ready to submit your project, please follow these steps to make sure everything is included.

#### Submission Checklist

1. **Implement All TODO Items:** Go through each of the Python files (`llm_client.py`, `rag_client.py`, `embedding_pipeline.py`, `ragas_evaluator.py`, and `chat.py`) and make sure you have completed all the `TODO` comments.
2. **End-to-End Testing:** Before submitting, run the entire workflow to confirm that everything works together.
  - First, run your embedding pipeline to process the documents.
  - Then, launch the chat application and test it with several questions to make sure it responds correctly and displays the evaluation scores.
3. **Provide Sample Questions:** In a text file called "evaluation_dataset.txt", Include a few sample questions that you used for testing and show the responses you expected the system to provide.
4. **Prepare Your Files:**
  - Make sure all your code is clean.
  - Zip up all your project files, including your report, into a single archive.

---

## Project Rubric

Use this project rubric to understand and assess the project criteria.

### Embedding & Data Pipeline

| Criteria | Submission Requirements |
| --- | --- |
| Chunking strategy | <ul><li>Use configurable <code>chunk_size</code> and <code>chunk_overlap</code> parameters at runtime (CLI flags or configuration).</li><li>Ensure chunks never exceed <code>chunk_size</code> characters or tokens.</li><li>Apply <code>chunk_overlap</code> consistently between consecutive chunks.</li></ul> |
| Embedding creation with metadata | <ul><li>Call an OpenAI embedding model to vectorize each chunk.</li><li>Store per-chunk metadata that includes at least the source/filepath and mission (Apollo 11, Apollo 13, or Challenger).</li><li>Handle existing documents according to <code>--update-mode</code>, with one of: <code>skip</code>, <code>update</code>, or <code>replace</code>.</li></ul> |
| ChromaDB persistence and inspection | <ul><li>Write embeddings to a ChromaDB collection at the configured <code>--chroma-dir</code> and <code>--collection-name</code>.</li><li>Provide a <code>--stats-only</code> option (or equivalent) that prints the collection size and at least one aggregate, such as the number of documents or chunks.</li></ul> |

### Retrieval & LLM Integration

| Criteria | Submission Requirements |
| --- | --- |
| Semantic retrieval from ChromaDB | <ul><li>Connect to the ChromaDB backend and issue similarity queries using the user question embedding.</li><li>Return the top <code>k</code> results, with <code>k</code> configurable at runtime.</li><li>If mission filtering is selected, restrict results to the chosen mission using metadata filtering.</li></ul> |
| Clean context construction for the LLM | <ul><li>Format retrieved chunks into a single context string with clear separators and source attributions.</li><li>Deduplicate results or sort them by score to avoid repeated snippets.</li></ul> |
| System prompt and conversation management | <ul><li>Implement a system prompt that positions the assistant as a NASA mission expert who cites retrieved sources.</li><li>Maintain conversation history across turns (role and content per turn) and include only the necessary context for each call.</li></ul> |
| Grounded LLM answers | <ul><li>Pass the constructed context and user query to the LLM and return an answer.</li><li>Instruct the model to rely on the provided context and indicate uncertainty when the context is insufficient.</li></ul> |

### Real-Time Evaluation

| Criteria | Submission Requirements |
| --- | --- |
| Integration of RAGAS metrics | <ul><li>Compute at least Response Relevancy and Faithfulness for each answer.</li><li>Support additional documented metrics, such as BLEU, ROUGE, or Precision.</li></ul> |
| Evaluation of a (question, context, answer) triple | <ul><li>Accept a question, retrieved context, and model answer; return a structured result containing metric names and values.</li><li>Handle empty or malformed inputs with a clear error message rather than crashing.</li></ul> |
| Batch evaluation using a test set | <ul><li>Load test questions (for example, from <code>test_questions.json</code> or <code>evaluation_dataset.txt</code>) and compute metrics end-to-end.</li><li>Output a summary for each question and an aggregate (mean or distribution) for each metric.</li></ul> |
| Evaluation dataset | <ul><li>Include <code>evaluation_dataset.txt</code> or <code>test_questions.json</code> with at least five mission-relevant questions spanning multiple categories: overview, emergency, disaster analysis, crew, technical, and timeline.</li><li>Ensure the file loads without errors and is referenced by the evaluation flow or README.</li></ul> |

