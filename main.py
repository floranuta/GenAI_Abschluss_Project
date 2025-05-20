from coordinator_agent import coordinator_agent
from dotenv import load_dotenv
from IPython.display import display, Image
from langchain_core.messages import convert_to_messages
from web_agent import handle_company_news
from web_agent import web_agent
import gradio as gr
load_dotenv()

# def pretty_print_message(message, indent=False):
#     pretty_message = message.pretty_repr(html=True)
#     if not indent:
#         print(pretty_message)
#         return

#     indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
#     print(indented)


# def pretty_print_messages(update, last_message=False):
#     is_subgraph = False
#     if isinstance(update, tuple):
#         ns, update = update
#         # skip parent graph updates in the printouts
#         if len(ns) == 0:
#             return

#         graph_id = ns[-1].split(":")[0]
#         print(f"Update from subgraph {graph_id}:")
#         print("\n")
#         is_subgraph = True

#     for node_name, node_update in update.items():
#         update_label = f"Update from node {node_name}:"
#         if is_subgraph:
#             update_label = "\t" + update_label

#         print(update_label)
#         print("\n")

#         messages = convert_to_messages(node_update["messages"])
#         if last_message:
#             messages = messages[-1:]

#         for m in messages:
#             pretty_print_message(m, indent=is_subgraph)
#         print("\n")

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
    
    # user_query = input("Frag etwas zum Markt (z.B. 'Was gibt es Neues bei NVIDIA?'): ")
    
    
    
    # result = coordinator.invoke({
    #     "messages": [
    #         {"role": "user", "content": user_query}
    #      ]
    #  })
    

    # for chunk in coordinator_agent.stream(
    # {
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": user_query,
    #         }
    #     ]
    # },
    # ):
    #     pretty_print_messages(chunk, last_message=True)
 
        #final_message_history = chunk["supervisor"]["messages"]

# web agent aufrufen
# print("\n\n🔍 Web-Agent:\n")
# for chunk in web_agent.stream(
#     {"messages": [{"role": "user", "content": user_query}]}
# ):
#     pretty_print_messages(chunk)



# web agent function
#result = handle_company_news(user_query)

# Выводим результат
#print("\n💬 Ответ от функции handle_company_news:\n")
#print(result)



    #print("\n📊 Ablaufplan (Mermaid-Graf):")
    #print(coordinator.get_graph().draw_mermaid()) 
    # print("\n💬 Antwort vom Koordinator:\n")
    # print(result)
    
    # if isinstance(result, dict) and "messages" in result:
    #     final_message = result["messages"][-1]
    #     print("\n💬 Finale Antwort:\n")
    #     print(final_message.content)
    # else:
    #     print("❌ Keine verständliche Antwort erhalten.")

    
   
    