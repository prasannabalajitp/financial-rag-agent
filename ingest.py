from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os, pdfplumber

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(DATA_DIR, filename)
        print(f"Extracting text from {filename}...")
        text = extract_text_from_pdf(full_path)
        if text.strip():
            docs.append({"filename": filename, "text": text})
    return docs

def chunk_texts(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

def main():
    print("Loading documents...")
    documents = load_documents()

    print("Loading embedding model...")
    embedder = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    texts = []
    metadatas = []
    for doc in documents:
        chunks = chunk_texts(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc["filename"]})

    print(f"Adding {len(texts)} chunks to Chroma vector store...")
    vectordb = Chroma(
        embedding_function=embedder,
        persist_directory=PERSIST_DIR,
    )

    vectordb.add_texts(texts, metadatas=metadatas)
    vectordb.persist()

    print("Ingestion completed successfully.")

if __name__ == "__main__":
    main()