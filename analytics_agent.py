# ✅ Step 1: Import Libraries
import os, json, getpass
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import chromadb



# ✅ Step 2: API Key Input
load_dotenv()


# ✅ Step 3: Load Existing Chroma DB from Local Filesystem

persistent_client = chromadb.PersistentClient(path="chroma/")
collection = persistent_client.get_or_create_collection(name="big_tech_financial_reports")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = Chroma(
    client=persistent_client,
    collection_name="big_tech_financial_reports",
    embedding_function=embeddings,
)

# ✅ Step 3: Load Existing Chroma DB from Local Filesystem
persistent_client = chromadb.PersistentClient(path="chroma/")
collection = persistent_client.get_or_create_collection(name="big_tech_financial_reports")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = Chroma(
    client=persistent_client,
    collection_name="big_tech_financial_reports",
    embedding_function=embeddings,
)

# ✅ Step 4: Gemini-powered Financial RAG QA with Sources
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ✅ Step 5: LangGraph Tools – Define Tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def forecast(query: str):
    """Generate forward-looking financial insights or predictive observations."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    context = "\n---\n".join(doc.page_content for doc in retrieved_docs)
    prompt = (
        "You are a financial forecasting assistant. Based on the following context, "
        "provide a forecast or predictive insight.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response, retrieved_docs

# ✅ Step 6: Agent Creation with LangGraph
analytics_agent = create_react_agent(
    model=llm,
    tools=[forecast],
    name="fanalytics_agent",
    prompt=(
        "You are a data science & analytics agent specializing in market forecasting.\n"
        "Use the 'forecast' tool to generate insights based on company financial data.\n"
        "Your job is to answer user questions involving trends, predictions, market dynamics, and revenue projections.\n"
        "Respond clearly with citations where appropriate."
    )
)

# def analyze_company(query: str) -> str:
#     return f"📊 Analytische Auswertung für: {query}"

# analytics_tool = Tool(
#     name="analyze_company",
#     func=analyze_company,
#     description="Verwende dieses Tool für Marktanalysen, Prognosen oder statistische Bewertungen."
# )

# analytics_agent = create_react_agent(
#     model=gemini_model,
#     tools=[analytics_tool],
#     name="analytics_agent",
#     prompt=(
#         "Du bist ein Finanzanalyst.\n"
#         "Du beantwortest nur Fragen zur Marktanalyse, Vorhersagen, Trends oder Finanzkennzahlen.\n"
#         "Verwende ausschließlich das Tool 'analyze_company'."
#     )
# )


