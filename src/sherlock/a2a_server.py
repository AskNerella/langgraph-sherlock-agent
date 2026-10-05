import asyncio
import contextlib
from typing import Any

import uvicorn
from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Part,
    TaskState,
)
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from sherlock.config import Settings
from sherlock.graph import build_graph, create_model
from sherlock.mcp import connect_toolsets


def build_agent_card(base_url: str) -> AgentCard:
    endpoint = f"{base_url.rstrip('/')}/a2a/jsonrpc"
    return AgentCard(
        name="Sherlock",
        description="Recon and intelligence agent for Mule, GitHub, GitBook, and Linear.",
        provider=AgentProvider(organization="KATE", url=base_url),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="integration_asset_discovery",
                name="Integration Asset Discovery",
                description="Discover and document an integration app and its APIs.",
                tags=["mule", "github", "gitbook", "linear", "recon"],
                examples=[
                    "Identify assets for salesforce-system-api deployed in Sandbox"
                ],
                input_modes=["text"],
                output_modes=["text", "task-status"],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=endpoint,
            )
        ],
    )


def _status_text(node: str, update: dict[str, Any]) -> str | None:
    if node == "validate":
        return "Request validated" if update.get("valid") else update.get("error")
    if node == "mule_scan" and (result := update.get("mule")):
        return f"Mule scan {result.status}: {len(result.apps)} apps, {len(result.apis)} APIs"
    if node == "github_scan" and (result := update.get("github")):
        return f"GitHub scan {result.status}: {len(result.repos)} repositories"
    if node == "gitbook_scan" and (result := update.get("gitbook")):
        return f"GitBook scan {result.status}: {len(result.docs)} documents"
    if node == "reconcile":
        return f"Recon complete: {len(update.get('apps', []))} applications"
    if node == "process_app" and update.get("app_runs"):
        run = update["app_runs"][-1]
        return f"Completed analysis and Linear finalization for {run.app.name}"
    return None


class SherlockAgentExecutor(AgentExecutor):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.running: dict[str, asyncio.Task] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message or not context.task_id or not context.context_id:
            return

        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        self.running[context.task_id] = asyncio.current_task()
        await updater.start_work(
            message=updater.new_agent_message(parts=[Part(text="Starting Sherlock recon")])
        )

        try:
            final = ""
            async with connect_toolsets(self.settings) as tools:
                graph = build_graph(create_model(self.settings), tools, self.settings)
                async for event in graph.astream(
                    {"request": context.get_user_input(), "app_runs": []},
                    config={"max_concurrency": self.settings.sherlock_max_concurrency},
                    stream_mode="updates",
                ):
                    for node, update in event.items():
                        if text := _status_text(node, update):
                            await updater.update_status(
                                TaskState.TASK_STATE_WORKING,
                                message=updater.new_agent_message(parts=[Part(text=text)]),
                            )
                        if node == "summarize":
                            final = update.get("final", "")

            if not final:
                raise RuntimeError("Sherlock completed without a result")
            await updater.add_artifact(
                parts=[Part(text=final)], name="sherlock-catalog", last_chunk=True
            )
            await updater.complete(
                message=updater.new_agent_message(parts=[Part(text="Sherlock complete")])
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await updater.failed(
                message=updater.new_agent_message(parts=[Part(text=f"Sherlock failed: {exc}")])
            )
        finally:
            self.running.pop(context.task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.task_id and (task := self.running.get(context.task_id)):
            task.cancel()
        updater = TaskUpdater(
            event_queue, context.task_id or "", context.context_id or ""
        )
        await updater.cancel()


async def _health(_):
    return JSONResponse({"status": "ok", "protocol": "A2A 1.0"})


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or Settings()
    card = build_agent_card(settings.a2a_public_url)
    handler = DefaultRequestHandler(
        agent_executor=SherlockAgentExecutor(settings),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [Route("/health", _health)]
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/a2a/jsonrpc"))
    return Starlette(routes=routes)


def main() -> None:
    settings = Settings()
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(create_app(settings), host=settings.a2a_host, port=settings.a2a_port)


if __name__ == "__main__":
    main()
