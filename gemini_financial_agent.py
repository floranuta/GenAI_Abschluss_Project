# ✅ Step 0: Installation Instructions (for users)
# To run this script, make sure you have all required dependencies installed:
# Run this in your terminal:
# pip install "unstructured[all-docs]" langchain langchain_community chromadb langchain-experimental \
# sentence-transformers ftfy PyMuPDF torch torchvision torchaudio faiss-cpu \
# git+https://github.com/openai/CLIP.git langchain-google-genai gradio langgraph

# ✅ Step 1: Import Libraries
import os, json, re, getpass, time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import chromadb
import gradio as gr

# ✅ Step 2: API Key Input
load_dotenv()
def set_env_variable(key_name, prompt_text):
    if not os.environ.get(key_name):
        os.environ[key_name] = getpass.getpass(prompt_text)
set_env_variable("GOOGLE_API_KEY", "Enter your GOOGLE_API_KEY: ")

# ✅ Step 3: Load Existing Chroma DB from Local Filesystem
chroma_dir = chroma_dir = r"C:\Users\Victoria\OneDrive\Desktop\Projekt_2_Hussam\GenerativeAI-II-Project\chroma_new\chroma"
persistent_client = chromadb.PersistentClient(path=chroma_dir = r"C:\Users\Victoria\OneDrive\Desktop\Projekt_2_Hussam\GenerativeAI-II-Project\chroma_new\chroma")
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
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ✅ Predefined Questions
predefined_questions = [
    "What were Apple's key financial highlights in fiscal year 2024?",
    "Summarize Google's investment strategy mentioned in their 2024 10-K.",
    "How did Microsoft describe AI integration in their business in 2024?",
    "What are the strategic priorities for NVIDIA mentioned in the 2024 annual report?",
    "What were Meta's main business segments and how did they perform financially in 2022?"
]

print("\n🧠 Predefined Gemini Answers:")
for q in predefined_questions:
    print(f"\n❓ Question: {q}")
    result = qa_chain.invoke({"question": q})
    print("📘 Answer:")
    print(result["answer"])
    print(f"📌 Source: {result['sources']}")

# ✅ Step 5: LangGraph Agent Setup with Custom Tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# ✅ LangGraph Agent Construction with Gemini
agent = create_react_agent(
    model=llm,
    tools=[retrieve],
    name="financial_agent",
    prompt=(
        "You are a financial assistant. Use the retrieval tool to answer questions based on financial reports.\n"
        "Answer clearly and cite sources when available.\n"
    )
)

# ✅ Step 6: Gradio Interface
chat_history = []

def chat_interface(user_input):
    try:
        if user_input.strip():
            result = agent.invoke({"input": user_input})
            output_text = result.get("output", "No answer returned.")

            # Extract sources if available in output text
            sources = "\n\n".join([
                f"📌 Source: {doc.metadata.get('source', 'Unknown')}"
                for doc in result.get("__raw__", {}).get("artifacts", [])
                if hasattr(doc, 'metadata')
            ])

            if sources:
                output_text += f"\n\n{sources}"

            chat_history.append((user_input, output_text))
        return "\n\n".join([f"❓ {q}\n📘 {a}" for q, a in chat_history])
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

interface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=3, placeholder="Type your question about the reports..."),
    outputs="text",
    title="💼 Gemini Financial Report Q&A",
    description="Ask questions about the financial data embedded from major tech companies."
)

interface.launch()
