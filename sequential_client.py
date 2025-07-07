# Sequentially chaining Agent calls

import asyncio
from acp_sdk.client import Client
from colorama import Fore 

async def run_llm_workflow() -> None:
    async with Client(base_url="http://localhost:8001") as rag_doc, \
               Client(base_url="http://localhost:8002") as gaurd_doc:
        run1 = await rag_doc.run_sync(
            agent="rag_agent", input="Give me the anatomy of a great prompt?"
        )
        content = run1.output[0].parts[0].content
        print(Fore.LIGHTMAGENTA_EX+ content + Fore.RESET)

        run2 = await gaurd_doc.run_sync(
            agent="gaurd_agent", input=f"Context: {content} Check for bias if any and give me response in numbered list?"
        )
        print(Fore.YELLOW + run2.output[0].parts[0].content + Fore.RESET)

if __name__ == "__main__":
    asyncio.run(run_llm_workflow())