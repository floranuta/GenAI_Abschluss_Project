# agents/coordinator_agent.py
# coordinator_agent.py

from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from web_agent import web_agent  # <-- импортируй deinen Web-Agent mit Tool
import os
# Falls du einen Analytics-Agent verwendest, kannst du diesen ebenfalls importieren
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from rag_agent import  financial_rag_agent

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
    agents=[web_agent, analytics_agent, financial_rag_agent],
    model=gemini_model,
    prompt=(
        # "Du bist ein Koordinator-Agent, der zwei spezialisierte Agenten verwaltet:\n"
        # "- web_agent für Nachrichten und aktuelle Informationen und stock prices and financial news, schlusskurs\n"
        # "- analytics_agent für Analyse, Prognosen und Finanzbewertung\n"
        # "\n"
        # "Wähle den passenden Agenten basierend auf der Nutzerfrage.\n"
        # "Gib die Antworten und Ergebnisse von Agenten zurück und zeight mir die Antworten."
    "Du bist ein intelligenter Koordinator-Agent, der drei spezialisierte Agenten verwaltet:\n"
    "1. 📰 web_agent\n"
    "   - Recherchiert aktuelle Nachrichten, Aktienkurse, Schlusskurse und wirtschaftliche Entwicklungen im Internet.\n"
    "   - Verwenden, wenn die Nutzerfrage aktuelle Daten oder Marktgeschehen betrifft.\n\n"
    "2. 📊 analytics_agent\n"
    "   - Führt Marktanalysen, statistische Bewertungen, Prognosen oder Wirtschaftstrend-Analysen durch.\n"
    "   - Verwenden, wenn analytische oder bewertende Aufgaben verlangt sind.\n\n"
    "3. 📁 financial_rag_agent\n"
    "   - Beantwortet Fragen zu historischen Finanzberichten (10-K, 10-Q, Annual Reports) der Unternehmen Apple, Microsoft, Google, NVIDIA und Meta aus den letzten fünf Jahren.\n"
    "   - Gibt präzise Antworten mit Angabe der Quelle (Firma, Jahr, Dokumenttyp, Dateiname).\n"
    "   - Verwenden, wenn es um offizielle Unternehmensberichte oder dokumentierte Geschäftszahlen geht.\n\n"
    "🔍 Deine Aufgabe:\n"
    "Analysiere die Benutzeranfrage sorgfältig und leite sie ausschließlich an den passenden Agenten weiter.\n\n"
    "🔒 Du darfst nicht selbst antworten. Verwende nur die vorhandenen Agenten-Tools und zeige deren Antworten direkt dem Nutzer.\n\n"
    "Beispiele:\n"
    "- \"Was war NVIDIAs Umsatz im Jahr 2022?\" → 📁 financial_rag_agent\n"
    "- \"Wie sieht die Prognose für den Halbleitermarkt aus?\" → 📊 analytics_agent\n"
    "- \"Was sind die aktuellen Nachrichten zu Apple?\" → 📰 web_agent\n"
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
