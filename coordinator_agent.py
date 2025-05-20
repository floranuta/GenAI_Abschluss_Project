# agents/coordinator_agent.py
# coordinator_agent.py

from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from web_agent import web_agent  # <-- импортируй deinen Web-Agent mit Tool
import os
# Falls du einen Analytics-Agent verwendest, kannst du diesen ebenfalls importieren
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent


# Einfache Beispiel-Funktion für Analytics
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

# 🧠 Koordinator-Agent mit Gemini 2.0 Flash
coordinator_agent = create_supervisor(
    agents=[web_agent, analytics_agent],
    model=gemini_model,
    prompt=(
        "Du bist ein Koordinator-Agent, der zwei spezialisierte Agenten verwaltet:\n"
        "- web_agent für Nachrichten und aktuelle Informationen\n"
        "- analytics_agent für Analyse, Prognosen und Finanzbewertung\n"
        "\n"
        "Wähle den passenden Agenten basierend auf der Nutzerfrage.\n"
        "Gib die Antworten und Ergebnisse von Agenten zurück und zeight mir die Antworten."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
