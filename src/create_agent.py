"""
Create and register the Sentiment Analysis agent in Azure AI Foundry.

Run after deploying infrastructure:
  azd provision --environment sentiment-analysis-mcp

Then:
  python src/create_agent.py
"""

from __future__ import annotations

import json
import os
import sys

# Load .env from project root
# .env values OVERRIDE existing env vars to avoid stale terminal sessions
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8-sig") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"')

from azure.ai.agents.models import FabricTool, MCPToolResource, McpTool, ToolSet
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# ─── Configuration ───────────────────────────────────────────────────────────

AI_SERVICES_ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
FOUNDRY_PROJECT = os.environ.get("FOUNDRY_PROJECT_NAME", "sentiment-analysis")
GPT_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT_NAME", "gpt-5.4-mini")
TOOL_MODE = os.environ.get("LANGUAGE_TOOL_MODE", "sdk").lower()

LANGUAGE_MCP_URL = (
    f"{AI_SERVICES_ENDPOINT.rstrip('/')}/language/mcp?api-version=2025-11-15-preview"
)

FABRIC_CONNECTION_NAME = os.environ.get("FABRIC_CONNECTION_NAME", "")


# ─── Authoritative instructions ─────────────────────────────────────────────

def _load_agent_instructions() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "agent.txt")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


AGENT_SYSTEM_PROMPT = _load_agent_instructions()


# ─── Agent builders ─────────────────────────────────────────────────────────

def _build_sdk_agent(client: AIProjectClient) -> object:
    """Create agent with Language SDK function tools and optional Fabric Data Agent."""
    from language_tools import TOOL_DEFINITIONS  # local import to avoid SDK dep at top

    tools = TOOL_DEFINITIONS.copy()

    if FABRIC_CONNECTION_NAME:
        conn_id = client.connections.get(FABRIC_CONNECTION_NAME).id
        fabric = FabricTool(connection_id=conn_id)
        tools.extend(fabric.definitions)
        print(f"✓ Fabric Data Agent added (connection: {FABRIC_CONNECTION_NAME})")

    return client.agents.create_agent(
        model=GPT_DEPLOYMENT,
        name="sentiment-analysis-agent",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=tools,
    )


def _build_mcp_agent(client: AIProjectClient) -> object:
    """Create agent with Language MCP server."""
    toolset = ToolSet()
    toolset.add(McpTool(server_label="azure_language", server_url=LANGUAGE_MCP_URL))

    resources = toolset.resources
    resources["mcp"] = [
        MCPToolResource(
            server_label="azure_language",
            headers={},
            require_approval="never",
        )
    ]

    return client.agents.create_agent(
        model=GPT_DEPLOYMENT,
        name="sentiment-analysis-agent",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=toolset.definitions,
        tool_resources=resources,
    )


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    endpoint = f"{AI_SERVICES_ENDPOINT.rstrip('/')}/api/projects/{FOUNDRY_PROJECT}"
    print(f"Connecting to Foundry project: {endpoint}")
    print(f"Tool mode:                     {TOOL_MODE.upper()}")
    if TOOL_MODE == "mcp":
        print(f"Language MCP server:           {LANGUAGE_MCP_URL}")

    credential = DefaultAzureCredential()
    client = AIProjectClient(endpoint=endpoint, credential=credential)

    if FABRIC_CONNECTION_NAME:
        print(f"Fabric connection:             {FABRIC_CONNECTION_NAME}")

    if TOOL_MODE == "mcp":
        agent = _build_mcp_agent(client)
    else:
        sys.path.insert(0, os.path.dirname(__file__))
        agent = _build_sdk_agent(client)

    print("\n✅ Agent created successfully!")
    print(f"   Agent ID:   {agent.id}")
    print(f"   Agent Name: {agent.name}")
    print(f"   Model:      {agent.model}")
    print(f"   Tool mode:  {TOOL_MODE}")

    config = {
        "agent_id": agent.id,
        "agent_name": agent.name,
        "model": agent.model,
        "endpoint": endpoint,
        "tool_mode": TOOL_MODE,
    }
    with open("agent_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("\n📄 Agent config saved to agent_config.json")


if __name__ == "__main__":
    main()
