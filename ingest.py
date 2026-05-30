"""
Run once to embed your medical PDF into Pinecone.

Usage:
    python ingest.py --pdf path/to/Medical_book.pdf
"""

import argparse
import os
from dotenv import load_dotenv
load_dotenv()

import pypdf
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import Config


def ingest(pdf_path: str):
    pdf_path = os.path.abspath(pdf_path)
    print(f"[1/4] Loading PDF: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": i + 1, "source": pdf_path}))
    print(f"      Loaded {len(documents)} pages.")

    print("[2/4] Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"      {len(chunks)} chunks created.")

    print("[3/4] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"[4/4] Uploading to Pinecone index '{Config.PINECONE_INDEX_NAME}'…")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=Config.PINECONE_INDEX_NAME,
    )
    print("Done! Pinecone index is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the medical PDF")
    args = parser.parse_args()
    ingest(args.pdf)
