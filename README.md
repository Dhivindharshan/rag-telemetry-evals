# RAG Telemetry & Evals
## Overview

This project implements a production-style Retrieval-Augmented Generation (RAG) pipeline with telemetry, retrieval evaluation, generation evaluation, MLflow experiment tracking, and FastAPI deployment.
The system supports document ingestion, semantic retrieval using ChromaDB, answer generation using LLMs, and comprehensive evaluation of retrieval and generation quality.

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

pip install -r requirements.txt
python api/main.py

## Retrieval Evaluation

python evals/run_evals.py --mode retrieval --top-k 3

## Generation Evaluation

python evals/run_evals.py --mode generation --top-k 3

## MLFlow

mlflow ui

## Project Structure

api/            - FastAPI service
src/            - Core RAG pipeline
telemetry/      - Logging and tracing
evals/          - Retrieval and generation evaluation
prompts/        - LLM prompts
data/           - Knowledge base documents
dashboard/      - Visualization components

## Evaluation Results

### Retrieval Metrics

- Precision@3: 0.74
- Recall@3: 0.67
- MRR: 0.85
- NDCG@3: 0.82
- Hit Rate: 0.92

### Generation Metrics

Evaluated using an LLM-as-a-Judge framework with:

- Faithfulness
- Relevance
- Completeness
- Correctness
