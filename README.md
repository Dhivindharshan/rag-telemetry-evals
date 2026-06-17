# RAG Telemetry & Evals

Production-grade Retrieval-Augmented Generation (RAG) system built with:

- ChromaDB Vector Store
- Sentence Transformers Embeddings
- FastAPI
- MLflow Tracking
- Retrieval Evaluation
- Generation Evaluation
- Telemetry Monitoring

## Features

- Document Chunking
- Embedding Generation
- Vector Search
- Retrieval Evaluation
- Generation Evaluation
- MLflow Experiment Tracking
- API Deployment
- Telemetry Logging

## Architecture

User Query
↓
Retriever
↓
ChromaDB
↓
Top-K Documents
↓
LLM
↓
Generated Answer
↓
Telemetry + Evaluation

## Tech Stack

- Python
- FastAPI
- ChromaDB
- Sentence Transformers
- MLflow
- Docker

## Running the Project

```bash
pip install -r requirements.txt
python api/main.py
