from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
#from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool

from crewai import Crew, Task, Agent, LLM

import logging 
from dotenv import load_dotenv

load_dotenv() 

server = Server()

llm = LLM(
    model= "ollama/mistral-nemo:latest", # This is the model name you pulled in Ollama
    base_url= "http://192.168.0.120:11434",
    # You can add other parameters if needed, e.g., temperature=0.7
)


@server.agent()
async def gaurd_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a gaurd Agent which supports unbiased output."
    '''
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])
    '''
    gaurd_agent = Agent(
        role="Gaurdrails Assistant",
        goal="To provide secure output",
        backstory="You are an expert to provide secure output without bias, and output should be brief and in numbered list.",
        verbose=True,
        allow_delegation=False,
        llm=llm, # Using the 'llm' defined at the top
        max_retry_limit=5
    )

    task1 = Task(
        description=input[0].parts[0].content,
        expected_output = "provide secure output without bias, and output should be brief and in numbered list.",
        agent=gaurd_agent
    )
    crew = Crew(agents=[gaurd_agent], tasks=[task1], verbose=True)
    
    task_output = await crew.kickoff_async()
    yield Message(parts=[MessagePart(content=str(task_output))])




if __name__ == "__main__":
    server.run(port=8002)
