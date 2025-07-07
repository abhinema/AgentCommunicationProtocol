from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool
import nest_asyncio

nest_asyncio.apply()

server = Server()

llm = LLM(
    model= "ollama/mistral-nemo:latest", # This is the model name you pulled in Ollama
    base_url= "http://192.168.0.120:11434",
    # You can add other parameters if needed, e.g., temperature=0.7
)

# --- CORRECTED RAG_CONFIG ---
rag_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "mistral-nemo:latest", # Or 'deepseek-r1:latest' if that's precise
            "base_url": "http://192.168.0.120:11434", # Moved base_url here
        }
    },
    "embedding_model": {
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large",
            "base_url": "http://192.168.0.120:11434", # Moved base_url here
        }
    }
}
# --- END CORRECTED RAG_CONFIG ---

rag_tool = RagTool(config=rag_config,
                   chunk_size=1200,
                   chunk_overlap=200,
                  )

rag_tool.add("./docs/learnlm_prompt_guide.pdf", data_type="pdf_file")


@server.agent()
async def rag_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is an agent for RAG, it uses a RAG pattern to find answers based on policy documentation."

    rag_agent = Agent(
        role="RAG Assistant",
        goal="Provide context based on user query",
        backstory="You are an expert to provide RAG based context for user queries",
        verbose=True,
        allow_delegation=False,
        llm=llm, # Using the 'llm' defined at the top
        tools=[rag_tool],
        max_retry_limit=5
    )

    task1 = Task(
        description=input[0].parts[0].content,
        expected_output = "A comprehensive response as to the users question",
        agent=rag_agent
    )
    crew = Crew(agents=[rag_agent], tasks=[task1], verbose=True)
    
    task_output = await crew.kickoff_async()
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    server.run(port=8001)
