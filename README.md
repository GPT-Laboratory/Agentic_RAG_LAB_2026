# Advanced Agentic RAG System

An advanced Retrieval-Augmented Generation (RAG) system with multi-hop reasoning, quality control, and web search fallback.

## Features

- **Multi-hop Reasoning**: Decomposes complex questions into sub-questions
- **Quality Control**: Document relevance grading and answer hallucination detection
- **Self-Reflection**: Automatic answer improvement through self-critique
- **Web Search Fallback**: Retrieves additional context when needed
- **LangGraph Workflow**: State-based agent orchestration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Groq API key:
```bash
export GROQ_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GROQ_API_KEY=your-api-key-here
```

## Usage

Run the interactive RAG system:
```bash
python main.py
```

Enter your questions when prompted. Press Enter (empty line) to exit.

## Project Structure

- `1_groq_models.py` - Groq API client and model configuration
- `2_rag_backend.py` - Document loading, chunking, embedding, and retrieval
- `3_agents.py` - Agent implementations (router, decompose, answer, grade, reflect)
- `4_graph_workflow.py` - LangGraph workflow orchestration
- `main.py` - Main entry point
- `query.txt` - Sample questions for testing

## Requirements

- Python 3.8+
- Groq API key
- See `requirements.txt` for full dependency list
