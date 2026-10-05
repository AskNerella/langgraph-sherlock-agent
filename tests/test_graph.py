import asyncio
import time

from langgraph.graph import END
from langgraph.types import Send

from sherlock import graph as graph_module
from sherlock.graph import _build_findings, _catalog, _dispatch_apps, _validate
from sherlock.mcp import Toolsets
from sherlock.models import (
    Api,
    App,
    Document,
    ExternalApiCall,
    GithubRecon,
    GitbookRecon,
    MuleDiscovery,
    Repo,
)


def test_validation_is_local_and_fast():
    assert _validate({"request": "identify the payments API"})["valid"] is True
    assert _validate({"request": "write a poem"})["valid"] is False


def test_dispatch_fans_out_once_per_app():
    state = {
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
    assert _dispatch_apps({"apps": []}) == "summarize"
    assert _catalog([]) == "SHERLOCK: No applications were discovered."


def test_linear_report_contains_recon_fields_and_external_calls():
    app = App(
        name="salesforce-system-api",
        environment="Sandbox",
        status="RUNNING",
        deployment_url="https://salesforce.example",
        mule_runtime="4.11.6-7e-java17",
        runtime_engine="Mule 4",
    )
    mule = MuleDiscovery(
        apps=[app],
        apis=[
            Api(
                name="Salesforce System API",
                version="1.0.0",
                asset_id="salesforce-system-api",
                status="Active",
                instance_id="20901302",
                active_contracts=2,
            )
        ],
    )
    github = GithubRecon(
        repos=[
            Repo(
                name=app.name,
                url="https://github.example/repo",
                mule_files=3,
                dataweave_files=2,
                external_api_calls=[
                    ExternalApiCall(
                        name="customer-system-api",
                        api_type="system-api",
                        source_file="src/main/resources/config/sandbox.yaml",
                        evidence="customer-system-api.host",
                    )
                ],
            )
        ]
    )
    gitbook = GitbookRecon(
        docs=[Document(title="Salesforce System API", url="https://docs.example/page")]
    )

    report = _build_findings(app, mule, github, gitbook).report_markdown
    assert report.startswith("# Salesforce System API — Asset Findings")
    assert "**Active contracts:** 2" in report
    assert "customer-system-api" in report
    assert "src/main/resources/config/sandbox.yaml" in report
    assert "## Scope" in report


def test_graph_compiles_with_three_way_recon_fanout(monkeypatch):
    monkeypatch.setattr(graph_module, "_agent", lambda *args, **kwargs: object())
    graph = graph_module.build_graph(object(), Toolsets([], [], [], []))
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert {
        ("recon_start", "mule_scan"),
        ("recon_start", "github_scan"),
        ("recon_start", "gitbook_scan"),
    } <= edges


def test_recon_overlaps_and_linear_receives_full_issue_body(monkeypatch):
    starts: dict[str, float] = {}
    saved: list[dict] = []

    class FakeAgent:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, _):
            starts[self.schema.__name__] = time.monotonic()
            await asyncio.sleep(0.05)
            values = {
                MuleDiscovery: MuleDiscovery(
                    apps=[App(name="orders-system-api", environment="Sandbox")]
                ),
                GithubRecon: GithubRecon(),
                GitbookRecon: GitbookRecon(),
            }
            return {"structured_response": values[self.schema]}

    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, args):
            if self.name == "list_issues":
                return {
                    "issues": [
                        {
                            "id": "issue-id",
                            "identifier": "ASK-52",
                            "title": args["query"],
                            "url": "https://linear.example/ASK-52",
                        }
                    ]
                }
            saved.append(args)
            return {
                "id": "issue-id",
                "identifier": "ASK-52",
                "url": "https://linear.example/ASK-52",
            }

    monkeypatch.setattr(
        graph_module,
        "_agent",
        lambda model, tools, schema, prompt: FakeAgent(schema),
    )
    linear = [FakeTool("list_issues"), FakeTool("save_issue")]
    graph = graph_module.build_graph(object(), Toolsets([], [], [], linear))
    result = asyncio.run(
        graph.ainvoke({"request": "identify orders API", "app_runs": []})
    )

    assert max(starts.values()) - min(starts.values()) < 0.03
    assert len(result["app_runs"]) == 1
    assert len(saved) == 1
    assert saved[0]["id"] == "issue-id"
    assert "# Orders System API — Asset Findings" in saved[0]["description"]
    assert "## Anypoint" in saved[0]["description"]
    assert "## GitHub" in saved[0]["description"]
    assert "## GitBooks" in saved[0]["description"]
