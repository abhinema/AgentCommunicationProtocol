
import asyncio
from acp_sdk.client import Client
from colorama import Fore
import gradio as gr

async def run_llm_workflow(user_query: str) -> str:
    async with Client(base_url="http://localhost:8001") as rag_doc, \
               Client(base_url="http://localhost:8002") as gaurd_doc:

        run1 = await rag_doc.run_sync(
            agent="rag_agent", input=user_query
        )

        content = run1.output[0].parts[0].content
        print(Fore.LIGHTMAGENTA_EX + content + Fore.RESET)

        run2 = await gaurd_doc.run_sync(
            agent="gaurd_agent", input=[{"content": f"Context: {content} Check for bias if any and give me response in numbered list?"}]
        )
        final_response = run2.output[0].parts[0].content
        print(Fore.YELLOW + final_response + Fore.RESET)
        return final_response

def run_llm_workflow_sync(user_query: str) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_llm_workflow(user_query))
    loop.close()
    return result

# Create Gradio interface
def gradio_interface(user_query: str) -> str:
    return run_llm_workflow_sync(user_query)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="User Query"),
    outputs=gr.Textbox(label="Final Response"),
    title="Sequential Agent Workflow",
    description="This interface allows you to interact with the RAG and Guard agents sequentially."
)

if __name__ == "__main__":
    iface.launch()