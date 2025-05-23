📊 Multimodales Marktanalyse-KI-System
🚀 Projektübersicht
Dieses System ist eine multimodale, agentenbasierte Generative-KI-Plattform, die reale Investor-Relations-Daten (IR) von führenden Technologieunternehmen – Apple, Microsoft, Google, NVIDIA und Meta – aus den Jahren 2020 bis 2024 analysiert. Es beantwortet finanzielle Anfragen, erstellt datenbasierte Prognosen, visualisiert Markttrends und ruft aktuelle Stimmungsdaten in Echtzeit aus dem Web ab.

Das System wurde modular mit spezialisierten Agenten entwickelt und wird über Gradio auf Hugging Face Spaces bereitgestellt. Es spiegelt reale KI-Anwendungen aus der Finanzanalyse wider.

🧠 Systemarchitektur
🧩 Agenten und deren Aufgaben
🌐 1. Multimodaler RAG-Spezialist
Bearbeitet Fragen zu Text, Tabellen, Grafiken und PDFs.

Verwendet einen Vektor-Datenbankansatz mit Chroma/FAISS zur Dokumentensuche.

Gibt präzise Antworten mit Quellenangabe aus IR-Dokumenten zurück.

📈 2. Data-Science- und Prognose-Agent
Extrahiert Zeitreihen aus Finanzdaten.

Führt Vorhersagen mit Prophet und ARIMA durch.

Erstellt Visualisierungen mit Matplotlib oder Plotly.

🌍 3. Echtzeit-Markt-Agent
Ruft aktuelle Finanznachrichten über NewsAPI, SerpAPI usw. ab.

Fasst Marktstimmungen und -entwicklungen mit Quellenangabe zusammen.

🧠 4. Koordinator-Agent
Zerlegt komplexe Anfragen in Teilaufgaben.

Koordiniert die Agentenkommunikation.

Integriert alle Ergebnisse zu einer fundierten Gesamtantwort.

🔄 Beispiel-Workflow
Benutzeranfrage:
„Basierend auf dieser Grafik und aktuellen Nachrichten – wie entwickelt sich Metas Ausblick für Q1 2025?“

Antwort des Systems:
RAG-Agent: fasst Metas IR-Präsentation zusammen.

Web-Agent: liefert aktuelle Schlagzeilen.

Forecast-Agent: erstellt eine Umsatzprognose.

Koordinator: vereint alles in eine strukturierte, zitierte Antwort.

🧰 Verwendeter Tech-Stack
Komponente	Technologien
Agenten-Orchestrierung	LangChain, LangGraph
Sprachmodell	Gemini (via langchain-google-genai)
Vektorspeicher	ChromaDB, SentenceTransformers
Zeitreihenprognose	Prophet, ARIMA
Visualisierung	Matplotlib, Plotly
Websuche & News	SerpAPI, NewsAPI, newspaper3k
Bereitstellung	Gradio + Hugging Face Spaces

📚 Datensatz
Investor Relations-Dokumente (2020–2024) von:

Apple

Microsoft

Google

NVIDIA

Meta

Dokumenttypen:

Jahresberichte (10-K), Quartalsberichte (10-Q)

Earnings Calls & Präsentationen

Charts & Investorenfolien

🎓 Projektzeitplan (Agile)
Woche	Meilenstein
1	Datensatzbeschaffung & Chroma-Indexierung
1	RAG-Agent mit Quellenangabe
1	Forecast-Agent: Prognosen + Visualisierung
2	Websuche & Echtzeit-Agent
2	Koordinator-Agent + QA-Modul
2	Gradio-Interface + Bereitstellung

🗃️ Projektstruktur (Empfehlung)
bash
Копировать
Редактировать
📁 projekt-root
│
├── chroma_db/               # Ungezipte Chroma-Datenbank
├── daten/                   # IR-PDF-Dokumente
├── src/
│   ├── forecast_agent.py    # Vorhersage-Logik
│   ├── rag_agent.py         # RAG-Agent mit Chroma
│   └── coordinator.py       # Aufgabenverteilung
├── .env                     # API-Schlüssel (nicht in Git)
├── requirements.txt
└── README.md
📦 Abzugebende Ergebnisse
✅ Gradio-App auf Hugging Face Spaces

✅ Vollständiges GitHub-Repository

✅ Visualisierte Finanzprognosen

✅ Demo-Präsentation

✅ Dokumentation (Architektur & Reflexion)

✅ Lernergebnisse
Multimodale Dokumentverarbeitung

Finanzdatenanalyse & Zeitreihenprognose

Integration von Echtzeitdatenquellen

Agentenbasiertes Design mit LangChain

Teamarbeit nach Scrum/Jira

Deployment mit Gradio & Hugging Face

📬 Kontakt
Bei Fragen zur Umsetzung, Kooperation oder Nutzung: bitte wenden Sie sich an die Projektleitung oder öffnen Sie ein Issue im GitHub-Repository.


