from fastapi import FastAPI
from pydantic import BaseModel
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from typing import List
import os, logging
os.environ["LANGCHAIN_HANDLER_TELEMETRY"] = "false"

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="../chroma_db",  
    embedding_function=embedding_function
)
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
    if not request.query.strip():
        return QueryResponse(
            query=request.query,
            answer="No query provided.",
            reasoning="Empty query was received.",
            sub_queries=[],
            sources=[]
        )
    docs = retrieve_relevant_docs(request.query)
    for i, doc in enumerate(docs):
        snippet = doc.page_content[:500].replace('\n', ' ')
        print(f"Retrieved doc #{i} from source {doc.metadata.get('source', 'unknown')}: {snippet} ...")
        
    context_texts = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    You are an expert financial document assistant specialized in analyzing SEC filings and financial reports.
    Your role is to provide accurate and precise answers strictly based on the context provided below.
    If the information requested by the user is NOT contained in the provided context, respond with:
    'Information not available.'

    Context:
    {context_texts}

    User Query: {request.query}

    Answer:
    """


    answer = llm(prompt)
    sources = [SourceInfo(source=doc.metadata.get("source", "unknown")) for doc in docs]
    return QueryResponse(
        query=request.query,
        answer=answer,
        reasoning="Retrieved relevant documents and generated answer with Ollama Mistral.",
        sub_queries=[request.query],
        sources=sources
    )
