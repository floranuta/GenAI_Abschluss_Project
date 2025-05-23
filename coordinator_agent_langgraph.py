# 💼 Multimodal Market Analyst – LangGraph pipeline
# Последовательность: financial_rag_agent → analytics_agent → web_agent

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from web_agent import web_agent  # <-- импортируй deinen Web-Agent mit Tool
import os
from rag_agent import  financial_rag_agent
from analytics_agent import analytics_agent


class AgentState(dict):
    pass

def start_node(state: AgentState):
    print("🚀 Start-Node erhalten:", state)
    return state

def financial_node(state: AgentState):
    print("📌 STATE IN financial_node:", state)
    query = state.get("input", "")
    if not query:
        raise ValueError("⚠️ Kein 'input' im State übergeben!")
    query = state["input"]
    response = financial_rag_agent.invoke({"input": query})
    state["financial_result"] = response.content
    return state

# Узел 2: Analytics Agent

def analytics_node(state: AgentState):
    query = state["input"] + "\n" + state.get("financial_result", "")
    response = analytics_agent.invoke({"input": query})
    state["analytics_result"] = response.content
    return state

# Узел 3: Web Agent

def web_node(state: AgentState):
    query = state["input"]
    response = web_agent.invoke({"input": query})
    state["web_result"] = response.content
    return state

# Узел 4: Финальный – объединить ответы

def final_node(state: AgentState):
    result = ""
    if "financial_result" in state:
        result += f"\n📁 Finanzdaten:\n{state['financial_result']}\n"
    if "analytics_result" in state:
        result += f"\n📊 Analyse:\n{state['analytics_result']}\n"
    if "web_result" in state:
        result += f"\n🌐 Aktuelle Infos:\n{state['web_result']}"
    return {"response": result.strip()}

# Сборка графа

graph = StateGraph(AgentState)
graph.add_node("start", start_node)
graph.add_node("financial", financial_node)
graph.add_node("analytics", analytics_node)
graph.add_node("web", web_node)
graph.add_node("final", final_node)

graph.set_entry_point("start")

graph.add_edge("start","financial")
graph.add_edge("financial", "analytics")
graph.add_edge("analytics", "web")
graph.add_edge("web", "final")
graph.add_edge("final", END)

# Компилируем
coordinator_graph = graph.compile()

# Пример вызова:
# result = coordinator_graph.invoke({"input": "Wie war die Performance von Apple im Jahr 2023?"})
# print(result["response"])