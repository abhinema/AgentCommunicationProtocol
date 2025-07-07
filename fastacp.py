
from typing import List, Dict, Callable, Optional, Union, Any, AsyncGenerator
import importlib.resources
import yaml
import json
from dataclasses import dataclass
from enum import Enum
from colorama import Fore
from acp_sdk.client import Client
from acp_sdk.models import (
    Message,
    MessagePart,
)
import requests

class Agent:
    """Representation of an ACP Agent."""
    
    def __init__(self, name: str, description: str, capabilities: List[str]):
        self.name = name
        self.description = description
        self.capabilities = capabilities
    
    def __str__(self):
        return f"Agent(name='{self.name}', description='{self.description}')"

class AgentCollection:
    """
    A collection of agents available on ACP servers.
    Allows users to discover available agents on ACP servers.
    """
    
    def __init__(self):
        self.agents = []
    
    @classmethod
    async def from_acp(cls, *servers) -> 'AgentCollection':
        """
        Creates an AgentCollection by fetching agents from the provided ACP servers.
        
        Args:
        *servers: ACP server client instances to fetch agents from
        
        Returns:
        AgentCollection: Collection containing all discovered agents
        """
        collection = cls()
        
        for server in servers:
            async for agent in server.agents():
                collection.agents.append((server, agent))
        
        return collection
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Find an agent by name in the collection.
        
        Args:
        name: Name of the agent to find
        
        Returns:
        Agent or None: The found agent or None if not found
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def __iter__(self):
        """Allows iteration over all agents in the collection."""
        return iter(self.agents)

class ACPCallingAgent:
    def __init__(self, acp_agents: Dict[str, Dict[str, Any]], model: Any, prompt_templates: Optional[Dict[str, str]] = None, planning_interval: int = 1):
        self.acp_agents = acp_agents
        self.model = model
        self.prompt_templates = prompt_templates or {}
        self.planning_interval = planning_interval

    def _generate_system_prompt(self) -> str:
        system_prompt = "You are an ACP calling agent. You can call the following agents:\n"
        for agent_name, agent_info in self.acp_agents.items():
            system_prompt += f"- {agent_name}: {agent_info['agent'].description}\n"
        return system_prompt

    async def _call_acp_agent(self, agent_name: str, message: str) -> str:
        agent_info = self.acp_agents[agent_name]
        client = agent_info['client']
        agent = agent_info['agent']
        
        # Use requests to send a POST request
        response = requests.post(
            f"{client.base_url}/agents/{agent.name}",
            json={"messages": [{"content": message}]},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()["content"]

    async def _step(self, user_query: str, step_number: int) -> str:
        system_prompt = self._generate_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        
        for i in range(self.planning_interval):
            response = await self.model(messages)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_response = await self._call_acp_agent(tool_call["name"], tool_call["arguments"]["prompt"])
                    messages.append({"role": "assistant", "content": tool_response})
            else:
                return response.content
        
        return "I wasn't able to complete this task within the maximum number of steps."

    async def run(self, user_query: str) -> str:
        for step_number in range(1, 11):
            print(f"[INFO] Step {step_number}/10")
            try:
                result = await self._step(user_query, step_number)
                print(f"[INFO] Step {step_number}/10 completed successfully")
                return result
            except Exception as e:
                print(f"[ERROR] Error in step {step_number}: {str(e)}")
        
        return "I wasn't able to complete this task within the maximum number of steps."