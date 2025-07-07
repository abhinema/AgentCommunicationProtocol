# Local RAG
from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

import warnings
warnings.filterwarnings('ignore')

# This LLM configuration for the Agent is likely correct for CrewAI's LLM class.
# The error is not coming from here, but from the RagTool config.
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
        description='How to write good Prompt for gemini like a pro?',
        expected_output = "A brief response as to the users question in numbered list.",
        agent=rag_agent
)

crew = Crew(agents=[rag_agent], tasks=[task1], verbose=True)
task_output = crew.kickoff()
print(task_output)