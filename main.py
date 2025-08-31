from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from typing import List, Dict

app = FastAPI()

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#loading the vector-store
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function,
)

#initialing local LLM model here
llm = OllamaLLM(model="mistral")

class QueryRequest(BaseModel):
    query: str

class SourceInfo(BaseModel):
    source: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    reasoning: str
    sub_queries: List[str]
    sources: List[SourceInfo]


def retrieve_relevant_docs(query: str, k: int = 3):
    docs = vectordb.similarity_search(query, k=k)
    return docs

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):

    docs = retrieve_relevant_docs(request.query)
    context_texts = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"Use the following context to answer the query:\n{context_texts}\n\nQuery: {request.query}\nAnswer:"
    answer = llm(prompt)

    sources = [SourceInfo(source=doc.metadata.get("source", "unknown")) for doc in docs]
    return QueryResponse(
        query=request.query,
        answer=answer,
        reasoning="Retrieved relevant documents and generated answer with Ollama Mistral.",
        sub_queries=[request.query],
        sources=sources
    )
