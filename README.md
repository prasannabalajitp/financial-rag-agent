# Financial RAG Agent

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for financial documents. It uses LangChain, Chroma vector store, SentenceTransformers, and Ollama’s Mistral LLM via FastAPI.

## Project Structure
- `data/` - Contains PDF financial reports for ingestion.
- `venv/` - Python virtual environment (excluded from Git).
- `ingest.py` - Script to extract text, create embeddings, and build vector store.
- `main.py` - FastAPI app serving a query endpoint using RAG.
- `requirements.txt` - Python dependencies to install.

## Setup Instructions
1. Clone the repository.
2. Create and activate a Python virtual environment:
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Place PDF documents in the `data/` folder.
5. Run ingestion script to build vector store:
   python ingest.py
6. Run FastAPI server:
   uvicorn main:app --reload

## Usage
- Send POST requests to `/query` with JSON body:
   {
     "query": "What was Google's revenue in 2023?"
   }
- The API will return an answer based on retrieved documents and the LLM.

## Notes
- Make sure Ollama and the Mistral model are installed and running locally.
- The `venv/` folder is excluded from version control.

## License
[Add your preferred license here]
