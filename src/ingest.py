import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils import fetch_10k_html_selenium

COMPANY_CIKS = {
    "GOOGL": "1652044",
    "MSFT": "789019",
    "NVDA": "1045810"
}
YEARS = [2022, 2023, 2024]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def clean_html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    texts = []

    for p in soup.find_all('p'):
        texts.append(p.get_text(separator="\n"))

    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cols = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            rows.append('\t'.join(cols))
        table_text = '\n'.join(rows)
        texts.append(table_text)

    for div in soup.find_all('div'):
        texts.append(div.get_text(separator="\n"))

    return '\n\n'.join(texts)

def load_documents():
    documents = []
    for company, cik in COMPANY_CIKS.items():
        for year in YEARS:
            html_content = fetch_10k_html_selenium(cik, "10-K", year)
            if html_content:
                text = clean_html_to_text(html_content)
                documents.append({"filename": f"{company}_10K_{year}.html", "text": text})
            else:
                continue
    return documents

def chunk_texts(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

def main():
    documents = load_documents()
    if not documents:
        return

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    texts, metadatas = [], []
    for doc in documents:
        chunks = chunk_texts(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc["filename"]})

    if not texts:
        return

    vectordb = Chroma(
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIR,
    )
    vectordb.add_texts(texts, metadatas=metadatas)
    # vectordb.persist()

if __name__ == "__main__":
    main()

