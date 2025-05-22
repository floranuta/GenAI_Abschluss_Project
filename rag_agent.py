from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import os
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# 🤖 Web-Agent mit Gemini 2.0 Flash
gemini_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

persistent_client = chromadb.PersistentClient(path="chroma/")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = Chroma(
    client=persistent_client,
    collection_name="big_tech_financial_reports",
    embedding_function=embeddings,
)

# Функция: отвечает на финансовый запрос с цитированием источников
def answer_financial_query(query: str) -> str:
    # Используем глобальные vector_store и llm
    global vector_store,  gemini_model

    query_embedding = embeddings.embed_query(query) 
    retrieved_docs = vector_store.similarity_search_by_vector(query_embedding, k=5)

    context = "\n\n".join([
        f"[{doc.metadata['company']}, {doc.metadata['year']}, {doc.metadata['type']}, {doc.metadata['source']}]:\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    prompt = f"""
You are a financial assistant. Based only on the following financial report excerpts, answer the user's query.
Use a clear and concise tone and cite the company, year, document type, and source for any fact.

User Query: {query}

Documents:
{context}

Answer:
"""

    response = gemini_model([HumanMessage(content=prompt)])
    return response.content

financial_rag_tool = Tool(
    name="analyze_financial_report",
    func=answer_financial_query,
    description=(
        "Beantworte Fragen zu Finanzberichten, Bilanzen, Quartalszahlen und Jahresabschlüssen "
        "von Apple, Microsoft, Google, NVIDIA und Meta in den letzten fünf Jahren. "
        "Die Antworten enthalten genaue Quellenangaben zum Bericht."
    )
)

financial_rag_agent = create_react_agent(
    model=gemini_model, 
    tools=[financial_rag_tool],
    name="financial_rag_agent",
    prompt=(
        "Du bist ein spezialisierter Finanzassistent.\n"
        "Du beantwortest ausschließlich Fragen zu den Finanzberichten von Apple, Microsoft, Google, NVIDIA und Meta.\n"
        "Nutze ausschließlich das Tool 'analyze_financial_report', um Informationen aus diesen Quellen zu beziehen.\n"
        "Gib stets eine präzise Antwort mit Angabe der Quelle (Unternehmen, Jahr, Berichtstyp, Dateiname)."
    )
)
