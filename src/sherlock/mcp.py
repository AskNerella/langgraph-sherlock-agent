import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool

from sherlock.config import Settings


@dataclass
class Toolsets:
    mule: list[BaseTool]
    github: list[BaseTool]
    gitbook: list[BaseTool]
    linear: list[BaseTool]


def _client(url: str, token: str = "") -> Client:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return Client(StreamableHttpTransport(url=url, headers=headers))


def _only(tools: list[BaseTool], names: set[str]) -> list[BaseTool]:
    selected = [tool for tool in tools if tool.name in names]
    for tool in selected:
        schema = tool.args_schema
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        tool.args_schema = _openai_schema(schema)
    return selected or tools


def _openai_schema(value):
    """Remove JSON Schema keywords unsupported by OpenAI function tools."""
    if isinstance(value, dict):
        return {
            key: _openai_schema(child)
            for key, child in value.items()
            if key not in {"$schema", "propertyNames"}
        }
    if isinstance(value, list):
        return [_openai_schema(child) for child in value]
    return value


@asynccontextmanager
async def connect_toolsets(settings: Settings) -> AsyncIterator[Toolsets]:
    """Connect once so parallel tool calls reuse live MCP sessions."""
    if not settings.gitbook_mcp_url:
        raise ValueError("GITBOOK_MCP_URL is required")

    clients = {
        "mule": _client(str(settings.mule_mcp_url), settings.mule_mcp_token),
        "github": _client(str(settings.github_mcp_url), settings.github_token),
        "gitbook": _client(str(settings.gitbook_mcp_url), settings.gitbook_token),
        "linear": _client(str(settings.linear_mcp_url), settings.linear_api_key),
    }
    async with AsyncExitStack() as stack:
        adapters = {
            name: await stack.enter_async_context(MCPAdapter(client))
            for name, client in clients.items()
        }
        discovered = await asyncio.gather(
            *(adapter.list_tools(cache_mode="use") for adapter in adapters.values())
        )
        tools = dict(zip(adapters, discovered, strict=True))
        yield Toolsets(
            mule=_only(tools["mule"], {"get_mule_assets"}),
            github=_only(
                tools["github"],
                {"search_repositories", "search_code", "get_file_contents"},
            ),
            gitbook=_only(
                tools["gitbook"],
                {
                    "invoke_operation",
                    "search",
                    "describe_operation",
                    "list_sites",
                    "get_site_structure",
                    "get_page",
                },
            ),
            linear=_only(
                tools["linear"],
                {
                    "list_teams",
                    "list_issues",
                    "save_issue",
                },
            ),
        )
