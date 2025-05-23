from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
import os
# Falls du einen Analytics-Agent verwendest, kannst du diesen ebenfalls importieren
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent



gemini_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

def analyze_company(query: str) -> str:
    return f"📊 Analytische Auswertung für: {query}"

analytics_tool = Tool(
    name="analyze_company",
    func=analyze_company,
    description="Verwende dieses Tool für Marktanalysen, Prognosen oder statistische Bewertungen."
)

analytics_agent = create_react_agent(
    model=gemini_model,
    tools=[analytics_tool],
    name="analytics_agent",
    prompt=(
        "Du bist ein Finanzanalyst.\n"
        "Du beantwortest nur Fragen zur Marktanalyse, Vorhersagen, Trends oder Finanzkennzahlen.\n"
        "Verwende ausschließlich das Tool 'analyze_company'."
    )
)
