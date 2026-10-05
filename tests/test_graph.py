import asyncio
import time

from langgraph.graph import END
from langgraph.types import Send

from sherlock import graph as graph_module
from sherlock.graph import _catalog, _dispatch_apps, _validate
from sherlock.mcp import Toolsets
from sherlock.models import (
    App,
    Endpoint,
    EndpointAnalysis,
    GithubRecon,
    GitbookRecon,
    Intelligence,
    LinearResult,
    MuleDiscovery,
)


def test_validation_is_local_and_fast():
    assert _validate({"request": "scan the payments API"})["valid"] is True
    assert _validate({"request": "write a poem"})["valid"] is False


def test_dispatch_fans_out_once_per_app():
    state = {
        "request": "scan apps",
        "apps": [App(name="orders"), App(name="payments")],
        "mule": MuleDiscovery(),
        "github": GithubRecon(),
        "gitbook": GitbookRecon(),
    }
    sends = _dispatch_apps(state)
    assert isinstance(sends, list)
    assert all(isinstance(item, Send) for item in sends)
    assert [item.arg["app"].name for item in sends] == ["orders", "payments"]


def test_empty_discovery_routes_to_summary():
    assert _dispatch_apps({"request": "scan apps", "apps": []}) == "summarize"
    assert _catalog([]) == "SHERLOCK: No applications were discovered."


def test_graph_compiles_with_three_way_recon_fanout(monkeypatch):
    monkeypatch.setattr(graph_module, "_agent", lambda *args, **kwargs: object())
    graph = graph_module.build_graph(object(), Toolsets([], [], [], []))
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert {
        ("recon_start", "mule_scan"),
        ("recon_start", "github_scan"),
        ("recon_start", "gitbook_scan"),
    } <= edges


def test_recon_and_per_app_work_really_overlap(monkeypatch):
    starts: dict[str, list[float]] = {}
    linear_calls: list[str] = []

    class FakeAgent:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, _):
            starts.setdefault(self.schema.__name__, []).append(time.monotonic())
            await asyncio.sleep(0.05)
            values = {
                MuleDiscovery: MuleDiscovery(
                    apps=[App(name="orders"), App(name="payments")],
                    endpoints=[
                        Endpoint(method="GET", path="/customers"),
                        Endpoint(method="POST", path="/customers"),
                    ],
                ),
                GithubRecon: GithubRecon(),
                GitbookRecon: GitbookRecon(),
                Intelligence: Intelligence(
                    app_name="app", full_report_markdown="# report"
                ),
                EndpointAnalysis: EndpointAnalysis(
                    endpoint=Endpoint(method="GET", path="/customers"),
                    report_markdown="## Endpoint: GET /customers",
                ),
                LinearResult: LinearResult(app_name="app"),
            }
            return {"structured_response": values[self.schema]}

    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, args):
            linear_calls.append(self.name)
            if self.name == "list_issues":
                return {
                    "issues": [
                        {
                            "id": "ASK-52",
                            "title": args["query"],
                            "url": "https://linear.example/ASK-52",
                            "team": "AskMeAI",
                        }
                    ]
                }
            if self.name == "get_issue":
                return {"documents": [{"id": "doc-1", "title": "existing report"}]}
            if self.name == "list_comments":
                return {"comments": []}
            return {}

    monkeypatch.setattr(
        graph_module,
        "_agent",
        lambda model, tools, schema, prompt: FakeAgent(schema),
    )
    linear = [
        FakeTool(name)
        for name in (
            "list_issues",
            "get_issue",
            "save_document",
            "list_comments",
            "save_comment",
        )
    ]
    graph = graph_module.build_graph(object(), Toolsets([], [], [], linear))
    result = asyncio.run(graph.ainvoke({"request": "scan orders API", "app_runs": []}))

    recon_starts = starts["MuleDiscovery"] + starts["GithubRecon"] + starts["GitbookRecon"]
    assert max(recon_starts) - min(recon_starts) < 0.03
    assert len(starts["EndpointAnalysis"]) == 4
    assert max(starts["EndpointAnalysis"]) - min(starts["EndpointAnalysis"]) < 0.03
    assert len(result["app_runs"]) == 2
    assert "save_issue" not in linear_calls
    assert linear_calls.count("save_document") == 2
    assert linear_calls.count("save_comment") == 2
