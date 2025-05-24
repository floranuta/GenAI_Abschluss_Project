# ✅ Before running in Visual Studio Code:
# Manually install required packages via terminal or requirements.txt:
# pip install pandas numpy matplotlib prophet statsmodels gradio chromadb sentence-transformers python-dotenv
# pip install "unstructured[all-docs]" langchain langchain_community langchain-experimental
# pip install sentence-transformers ftfy PyMuPDF torch torchvision torchaudio faiss-cpu
# pip install git+https://github.com/openai/CLIP.git
# pip install langchain-google-genai gradio langgraph

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import gradio as gr
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import chromadb
from langchain.vectorstores import Chroma

# ✅ Load Environment Variables
load_dotenv()
def set_env_variable(key_name, prompt_text):
    if not os.environ.get(key_name):
        import getpass
        os.environ[key_name] = getpass.getpass(prompt_text)

set_env_variable("GOOGLE_API_KEY", "Enter your GOOGLE_API_KEY: ")

# ✅ Setup Gemini Model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load Chroma DB
chroma_dir = "./chroma_new/chroma"
persistent_client = chromadb.PersistentClient(path=chroma_dir)
collection = persistent_client.get_or_create_collection(name="big_tech_financial_reports")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = Chroma(client=persistent_client, collection_name="big_tech_financial_reports", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ Forecasting Functions
def parse_yearly_data(text):
    pattern = r"(\d{4})[^0-9]{1,10}?([\d.,]+)[\s]?[BbMm]?"
    matches = re.findall(pattern, text)
    data = {}
    for year, value in matches:
        try:
            value = float(value.replace(',', ''))
            data[int(year)] = value
        except ValueError:
            continue
    return dict(sorted(data.items()))

def forecast_next_value(data):
    years = list(data.keys())
    values = list(data.values())
    if len(years) < 2:
        raise ValueError("❌ Error: Need at least 2 years of data.")
    delta = values[-1] - values[-2]
    next_year = years[-1] + 1
    data[next_year] = values[-1] + delta
    return data

def forecast_with_prophet(data):
    try:
        df = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in data]), "y": list(data.values())})
        df = df.dropna()
        if len(df) < 2:
            return "⚠️ Not enough valid data for Prophet."
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=1, freq='Y')
        forecast = model.predict(future)
        return round(forecast.iloc[-1]['yhat'], 1)
    except Exception as e:
        return f"⚠️ Prophet forecast failed: {e}"

def forecast_with_arima(data):
    try:
        values = list(data.values())
        if len(values) < 3:
            return "⚠️ Not enough data for ARIMA."
        model = ARIMA(values, order=(1, 1, 1))
        model_fit = model.fit()
        return round(model_fit.forecast()[0], 1)
    except Exception as e:
        return f"⚠️ ARIMA forecast failed: {e}"

def plot_forecast(data, title="Financial Forecast"):
    years = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(8, 5))
    plt.plot(years, values, marker='o')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()

# ✅ Forecast Tool
last_data_source = ""

def forecast_tool_func(query):
    global last_data_source
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_content = "\n".join(doc.page_content for doc in retrieved_docs)
    sources = ", ".join(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))

    combined_query = f"{retrieved_content}\n\nUser query: {query}"
    data = parse_yearly_data(combined_query)
    if len(data) < 2:
        print("⚠️ Not enough data in retrieved docs. Falling back to user query only.")
        data = parse_yearly_data(query)

    if not data or len(data) < 2:
        raise ValueError("❌ Error: Need at least 2 years of numerical data.")

    extrapolated = forecast_next_value(data.copy())
    prophet_result = forecast_with_prophet(data)
    arima_result = forecast_with_arima(data)
    plot_forecast(extrapolated)
    last_data_source = query

    result = (
        f"📈 Forecast complete using Prophet and ARIMA.\n"
        f"🔮 Prophet prediction for {max(data.keys()) + 1}: ${prophet_result}B\n"
        f"📊 ARIMA prediction for {max(data.keys()) + 1}: ${arima_result}B\n"
        f"🖼️ See image below for forecast plot.\n\n"
        f"📌 User query: '{query}'\n"
        f"📖 Retrieved context from: {sources}\n"
        f"🔢 Parsed numeric data: {data}\n"
        f"📘 Note: ARIMA relies on short-term trends; Prophet captures seasonality and holidays."
    )
    return result

# ✅ LangChain Agent
forecast_tool = Tool(name="FinancialForecaster", func=forecast_tool_func, description="Forecast financials using historical data")
forecast_agent = initialize_agent(tools=[forecast_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# ✅ Gradio UI
chat_history = []

def chat_interface_with_plot(user_input):
    try:
        if user_input.strip():
            result = forecast_agent.run(user_input)
            chat_history.append((user_input, result))
        return "\n\n".join([f"❓ {q}\n📘 {a}" for q, a in chat_history]), "forecast.png"
    except Exception as e:
        return f"⚠ {e}", None

if __name__ == "__main__":
    gr.Interface(
        fn=chat_interface_with_plot,
        inputs=gr.Textbox(lines=3, placeholder="Ask a forecast question..."),
        outputs=["text", "image"],
        title="📈 Financial Forecasting Assistant",
        description="Ask financial projection questions based on historical earnings reports."
    ).launch()
