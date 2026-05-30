"""
RAG pipeline: retrieve from Pinecone → generate with Gemini.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import Config

_chain = None

PROMPT_TEMPLATE = """You are Medi-Chat, a knowledgeable and empathetic medical information assistant.
Use the following excerpts from verified medical literature to answer the user's question accurately.
If the answer is not covered by the provided context, say so honestly and advise the user to consult
a qualified healthcare professional.

Context:
{context}

Question: {question}

Answer (be clear, concise, and cite page numbers if available):"""


def _build_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = PineconeVectorStore(
        index_name=Config.PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=Config.GEMINI_API_KEY,
        temperature=0.2,
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_answer(query: str) -> str:
    global _chain
    if _chain is None:
        _chain = _build_chain()
    try:
        return _chain.invoke(query)
    except Exception as e:
        return f"An error occurred while processing your query: {str(e)}"
