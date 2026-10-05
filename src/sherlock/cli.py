import argparse
import asyncio

from sherlock.config import Settings
from sherlock.graph import build_graph, create_model
from sherlock.mcp import connect_toolsets


async def _run(request: str) -> None:
    settings = Settings()
    model = create_model(settings)
    async with connect_toolsets(settings) as tools:
        graph = build_graph(model, tools, settings)
        result = await graph.ainvoke(
            {"request": request, "app_runs": []},
            config={"max_concurrency": settings.sherlock_max_concurrency},
        )
    print(result.get("final") or result.get("error", "Sherlock did not return a result"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sherlock LangGraph agent")
    parser.add_argument("request", help="Integration discovery request")
    args = parser.parse_args()
    asyncio.run(_run(args.request))


if __name__ == "__main__":
    main()
