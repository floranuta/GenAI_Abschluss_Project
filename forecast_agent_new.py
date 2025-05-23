# ✅ Step 0: Installation Instructions (for users)
# To run this script, make sure you have all required dependencies installed:
# Run this in your terminal (or Colab cell):
#
# pip install -q pandas numpy matplotlib prophet statsmodels gradio chromadb sentence-transformers python-dotenv
# pip install -q "unstructured[all-docs]" langchain langchain_community chromadb langchain-experimental
# pip install -q sentence-transformers ftfy PyMuPDF
# pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install -q faiss-cpu
# pip install -q git+https://github.com/openai/CLIP.git
# pip install -U langchain-google-genai
# pip install -U langchain langchain-google-genai chromadb langchain-experimental
# pip install gradio langgraph

# ✅ Install Forecasting and Visualization Tools
!pip install -q pandas numpy matplotlib prophet statsmodels gradio chromadb sentence-transformers python-dotenv

# ✅ Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import gradio as gr
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from google.colab import files
import zipfile
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Load Environment Variables from .env or set securely
load_dotenv()
def set_env_variable(key_name, prompt_text):
    if not os.environ.get(key_name):
        from getpass import getpass
        os.environ[key_name] = getpass(prompt_text)

set_env_variable("GOOGLE_API_KEY", "Enter your GOOGLE_API_KEY: ")

# ✅ Setup Gemini Model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Upload and Load Chroma Vector Store
uploaded = files.upload()
for filename in uploaded.keys():
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall("/content/chroma_db")
        print(f"✅ Extracted: {filename} to /content/chroma_db")

persistent_client = chromadb.PersistentClient(path="/content/chroma_db")
vectorstore = Chroma(
    client=persistent_client,
    collection_name="financial_reports",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# ✅ Forecasting utility functions
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

def forecast_next_value(data: dict):
    years = list(data.keys())
    values = list(data.values())
    if len(years) < 2:
        raise ValueError("❌ Error: Need at least 2 years of numerical data in your query to generate a forecast.\n\n👉 Try rephrasing and include examples like 'Revenue was 100B in 2022 and 120B in 2023'.")
    delta = values[-1] - values[-2]
    next_year = years[-1] + 1
    next_value = values[-1] + delta
    data[next_year] = next_value
    return data

def forecast_with_prophet(data):
    try:
        df = pd.DataFrame({"ds": pd.to_datetime([f"{year}-01-01" for year in data.keys()], errors='coerce'), "y": list(data.values())})
        df = df.dropna()
        if len(df) < 2:
            return "⚠️ Not enough valid data for Prophet."
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=1, freq='Y')
        forecast = model.predict(future)
        return forecast.iloc[-1]['yhat']
    except Exception as e:
        return f"⚠️ Prophet forecast failed: {str(e)}"

def forecast_with_arima(data):
    try:
        values = list(data.values())
        if len(values) < 3:
            return "⚠️ Not enough data for ARIMA."
        model = ARIMA(values, order=(1, 1, 1))
        model_fit = model.fit()
        return model_fit.forecast()[0]
    except Exception as e:
        return f"⚠️ ARIMA forecast failed: {str(e)}"

def plot_forecast(data: dict, title="Financial Forecast"):
    years = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(8, 5))
    plt.plot(years, values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()

# ✅ Forecasting Tool Function
last_data_source = ""

def forecast_tool_func(query: str) -> str:
    global last_data_source
    data = parse_yearly_data(query)
    if not data or len(data) < 2:
        raise ValueError("❌ Error: Need at least 2 years of numerical data in your query to generate a forecast.\n\n👉 Try rephrasing and include examples like 'Revenue was 100B in 2022 and 120B in 2023'.")

    extrapolated = forecast_next_value(data.copy())
    prophet_result = forecast_with_prophet(data)
    arima_result = forecast_with_arima(data)
    plot_forecast(extrapolated, title="Forecast based on financial data")
    last_data_source = query

    result_text = (
        f"📈 Forecast complete using Prophet and ARIMA.\n"
        f"🔮 Prophet prediction for {max(data.keys()) + 1}: {prophet_result}\n"
        f"📊 ARIMA prediction for {max(data.keys()) + 1}: {arima_result}\n"
        f"🖼️ Plot saved as forecast.png.\n"
        f"\n📌 Source data extracted from your query: '{query}'"
        f"\n🔢 Parsed numeric data: {data}"
    )
    return result_text

# ✅ LangChain Tool and Agent
forecast_tool = Tool(
    name="FinancialForecaster",
    func=forecast_tool_func,
    description="Use this tool to forecast financial metrics based on yearly values given in the user's query. It parses numerical data and extrapolates one year forward using Prophet and ARIMA, also saving a plot."
)

forecast_agent = initialize_agent(
    tools=[forecast_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ Gradio Forecasting Dashboard
chat_history = []

def chat_interface_with_plot(user_input):
    try:
        if user_input.strip():
            result = forecast_agent.run(user_input)
            chat_history.append((user_input, result))
        return "\n\n".join([f"❓ {q}\n📘 {a}" for q, a in chat_history]), "forecast.png"
    except Exception as e:
        return f"⚠ {str(e)}", None

gr.Interface(
    fn=chat_interface_with_plot,
    inputs=gr.Textbox(lines=3, placeholder="Stelle eine Prognosefrage zur Umsatzentwicklung..."),
    outputs=["text", "image"],
    title="📈 Finanzprognose-Dashboard",
    description="Frage nach Prognosen zur Umsatzentwicklung basierend auf geschätzten Jahresdaten."
).launch()
