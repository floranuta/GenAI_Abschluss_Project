from coordinator_agent import coordinator_agent
from coordinator_agent_langgraph import  AgentState, coordinator_graph
from dotenv import load_dotenv
from langchain_core.messages import convert_to_messages
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()



# demo = gr.Interface(
#     fn=lambda query: "\n\n".join(
#         f"🔹 {msg.name if hasattr(msg, 'name') else 'Agent'}:\n{msg.content}"
#         for msg in coordinator_agent.invoke({"messages": [{"role": "user", "content": query}]}).get("messages", [])
#         if hasattr(msg, "content") and msg.content
#     ),
#     inputs=gr.Textbox(label="📝 Frag etwas zum Markt", placeholder="z.B. Was gibt es Neues bei NVIDIA?"),
#     outputs=gr.Textbox(label="🤖 Antwort der Agenten"),
#     title="📊 Multimodaler Markt-Analyst (Gradio)",
#     description="Ein intelligentes System zur Analyse von Marktinformationen mit mehreren Agenten.",
# )

#   Langchain Version
def run_supervisor_full(query):
    result = coordinator_agent.invoke({"messages": [{"role": "user", "content": query}]})
    history = result.get("messages", [])

    chunks = []
    for msg in history:
        name = getattr(msg, "name", "Agent")
        if hasattr(msg, "content") and msg.content:
            content = msg.content
        elif hasattr(msg, "tool_call_id"):
            content = f"[→ Übergabe an Tool: {msg.tool}]"
        else:
            continue
        chunks.append(f"🔹 {name}:\n{content}")

    return "\n\n".join(chunks)
# demo = gr.Interface(
#     fn=lambda query: "\n\n".join(
#         f"🔹 {msg.name if hasattr(msg, 'name') else 'Agent'}:\n"
#         f"{msg.content}"
#         + (f"\n\n🔗 Quelle: {msg.metadata.get('source')}" if hasattr(msg, "metadata") and "source" in msg.metadata else "")
#         for msg in coordinator_agent.invoke({"messages": [{"role": "user", "content": query}]}).get("messages", [])
#         if hasattr(msg, "content") and msg.content
#     ),
#     inputs=gr.Textbox(label="📝 Frag etwas zum Markt", placeholder="z.B. Was gibt es Neues bei NVIDIA?"),
#     outputs=gr.Textbox(label="🤖 Antwort der Agenten"),
#     title="📊 Multimodaler Markt-Analyst (Gradio)",
#     description="Ein intelligentes System zur Analyse von Marktinformationen mit mehreren Agenten.",
# )

demo = gr.Interface(
    fn=run_supervisor_full,
    inputs=gr.Textbox(label="📝 Marktfrage", placeholder="z. B. Wie war die Performance von NVIDIA 2023?"),
    outputs=gr.Textbox(label="🤖 Antwortverlauf"),
    title="🧠 Koordinator-Agent mit LangGraph",
    description="Automatisches Routing zu spezialisierten Agenten mit vollständigem Nachrichtenverlauf."
)


# Langgraph Version
# def run_coordinator(user_input):
#     result = coordinator_graph.invoke(AgentState({"input": user_input}))
#     return result.get("response", "Keine Antwort erhalten.")
# demo = gr.Interface(
#     fn=run_coordinator,
#     inputs=gr.Textbox(label="📝 Deine Marktfrage", placeholder="z. B. Wie war NVIDIAs Ergebnis 2023?"),
#     outputs=gr.Textbox(label="🤖 Antwort vom System"),
#     title="📊 Multimodaler Markt-Analyst (LangGraph)",
#     description="Ein intelligentes System mit mehreren spezialisierten Agenten (Finanz, Analyse, Web)."
# )

if __name__ == "__main__":
    # user_question = "Wie war die Performance von Apple im Jahr 2023?"
    # print("▶️ Direktaufruf:")
    # print(run_coordinator(user_question))
    demo.launch()
    
