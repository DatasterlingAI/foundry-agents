"""Create (or update) the Meridian Knowledge Copilot agent in Foundry."""
import os
import pathlib

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition, AzureAISearchTool, AzureAISearchToolResource,
    AISearchIndexResource, AzureAISearchQueryType,
)
from azure.identity import DefaultAzureCredential

AGENT_NAME = os.environ.get("AGENT_NAME", "meridian-knowledge-copilot")
INDEX_NAME = os.environ.get("INDEX_NAME", "meridian-docs")
here = pathlib.Path(__file__).parent

project = AIProjectClient(endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                          credential=DefaultAzureCredential())

connection_id = project.connections.get(os.environ["SEARCH_CONNECTION"]).id

search_tool = AzureAISearchTool(
    azure_ai_search=AzureAISearchToolResource(indexes=[
        AISearchIndexResource(
            project_connection_id=connection_id,
            index_name=INDEX_NAME,
            # keyword + vector + semantic re-ranking: best quality for policy text
            query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
            top_k=5,
        )
    ])
)

agent = project.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model=os.environ["FOUNDRY_MODEL"],
        instructions=(here / "instructions.md").read_text(encoding="utf-8"),
        tools=[search_tool],
    ),
)
print(f"Deployed {agent.name} version {agent.version}")
