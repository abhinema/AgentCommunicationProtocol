# Sequentially chaining Agent calls

import asyncio
import nest_asyncio

from acp_sdk.client import Client

from crewai import Crew, Task, Agent, LLM
from fastacp import AgentCollection, ACPCallingAgent
from smolagents import LiteLLMModel

from colorama import Fore 
'''
model = LLM(
    model= "ollama/llama3.2:latest", # This is the model name you pulled in Ollama
    base_url= "http://192.168.0.120:11434",
    # You can add other parameters if needed, e.g., temperature=0.7
)
'''

model = LiteLLMModel(
    model_id="ollama_chat/mistral-nemo:latest",
    api_base="http://192.168.0.120:11434",
    #num_ctx=8192,
)

'''
model = LiteLLMModel(
    model_id="openai/gpt-4"
)
'''

async def run_llm_workflow() -> None:
    async with Client(base_url="http://localhost:8001") as rag_doc, \
               Client(base_url="http://localhost:8002") as gaurd_doc:

        agent_collection = await AgentCollection.from_acp(rag_doc, gaurd_doc)  
        acp_agents = {agent.name: {'agent':agent, 'client':client} for client, agent in agent_collection.agents}
        print(acp_agents) 
        # passing the agents as tools to ACPCallingAgent
        acpagent = ACPCallingAgent(acp_agents=acp_agents, model=model)
        print("acp agent")
        # running the agent with a user query
        result = await acpagent.run("Give me the anatomy of a great prompt?")
        print(Fore.YELLOW + f"Final result: {result}" + Fore.RESET)

if __name__ == "__main__":
    asyncio.run(run_llm_workflow())
