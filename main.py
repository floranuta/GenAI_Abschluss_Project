from coordinator_agent import coordinator_agent
from dotenv import load_dotenv
from IPython.display import display, Image
from langchain_core.messages import convert_to_messages
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI 
from rag_agent import  financial_rag_agent
load_dotenv()



demo = gr.Interface(
    fn=lambda query: "\n\n".join(
        f"🔹 {msg.name if hasattr(msg, 'name') else 'Agent'}:\n{msg.content}"
        for msg in coordinator_agent.invoke({"messages": [{"role": "user", "content": query}]}).get("messages", [])
        if hasattr(msg, "content") and msg.content
    ),
    inputs=gr.Textbox(label="📝 Frag etwas zum Markt", placeholder="z.B. Was gibt es Neues bei NVIDIA?"),
    outputs=gr.Textbox(label="🤖 Antwort der Agenten"),
    title="📊 Multimodaler Markt-Analyst (Gradio)",
    description="Ein intelligentes System zur Analyse von Marktinformationen mit mehreren Agenten.",
)
if __name__ == "__main__":

    demo.launch()
    
