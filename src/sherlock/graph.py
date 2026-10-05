import asyncio
import json
import operator
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from sherlock.config import Settings
from sherlock.mcp import Toolsets, connect_toolsets
from sherlock.models import (
    App,
    AppRun,
    Endpoint,
    EndpointAnalysis,
    GithubRecon,
    GitbookRecon,
    Intelligence,
    LinearResult,
    MuleDiscovery,
)


class SherlockState(TypedDict, total=False):
    request: str
    valid: bool
    error: str
    mule: MuleDiscovery
    github: GithubRecon
    gitbook: GitbookRecon
    apps: list[App]
    app_runs: Annotated[list[AppRun], operator.add]
    final: str


class AppState(TypedDict, total=False):
    request: str
    app: App
    mule: MuleDiscovery
    github: GithubRecon
    gitbook: GitbookRecon
    endpoint: Endpoint
    intelligence: Intelligence
    endpoint_reports: Annotated[list[EndpointAnalysis], operator.add]
    linear: LinearResult
    app_runs: list[AppRun]


class AppOutput(TypedDict):
    app_runs: list[AppRun]


def _agent(model: BaseChatModel, tools: list[BaseTool], schema: type, prompt: str):
    return create_agent(model=model, tools=tools, system_prompt=prompt, response_format=schema)


def create_model(settings: Settings) -> BaseChatModel:
    return init_chat_model(
        settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key.get_secret_value(),
        use_responses_api=True,
        reasoning_effort=settings.openai_reasoning_effort,
        verbosity=settings.openai_verbosity,
    )


def _tool(tools: list[BaseTool], name: str) -> BaseTool:
    try:
        return next(tool for tool in tools if tool.name == name)
    except StopIteration as exc:
        raise RuntimeError(f"MCP tool not available: {name}") from exc


def _tool_json(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text":
                return json.loads(block["text"])
    raise ValueError(f"Unexpected MCP result: {type(value).__name__}")


async def _call(tools: list[BaseTool], name: str, args: dict):
    return _tool_json(await _tool(tools, name).ainvoke(args))


async def _invoke(agent, message: str):
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]})
    return result["structured_response"]


def _validate(state: SherlockState) -> dict:
    request = state.get("request", "").strip()
    terms = ("discover", "scan", "document", "analyse", "analyze", "api", "mule", "anypoint", "integration", "repo")
    valid = bool(request) and any(term in request.lower() for term in terms)
    return {"valid": valid, "error": "" if valid else "Not an integration discovery task"}


def _after_validation(state: SherlockState) -> Literal["mule_scan", "__end__"]:
    return "mule_scan" if state["valid"] else END


def _dispatch_apps(state: SherlockState) -> list[Send] | str:
    apps = state["apps"]
    if not apps:
        return "summarize"
    return [
        Send(
            "process_app",
            {
                "request": state["request"],
                "app": app,
                "mule": state["mule"],
                "github": state["github"],
                "gitbook": state["gitbook"],
            },
        )
        for app in apps
    ]


def _catalog(runs: list[AppRun]) -> str:
    if not runs:
        return "SHERLOCK: No applications were discovered."
    lines = ["# SHERLOCK Asset Catalog", ""]
    for run in runs:
        intel, linear = run.intelligence, run.linear
        lines.extend(
            [
                f"## {run.app.name}",
                f"- Recon: GitHub {run.github.status}; GitBook {run.gitbook.status}"
                + (f" — {run.gitbook.error}" if run.gitbook.error else ""),
                f"- Endpoints: {len(intel.endpoints)}; connectors: {len(intel.connectors)}; DataWeave: {intel.total_dataweave_transforms}",
                f"- Linear: {linear.status} {linear.issue_url or linear.issue_id}"
                + (f" — {linear.error}" if linear.error else ""),
                "",
            ]
        )
    return "\n".join(lines)


def build_graph(model: BaseChatModel, tools: Toolsets, settings: Settings | None = None):
    settings = settings or Settings()
    mule_agent = _agent(
        model,
        tools.mule,
        MuleDiscovery,
        """You are Sherlock's Anypoint discovery specialist. Call get_mule_assets using the asset name in the request. If the request is broad or the first result is empty, search api, system, process, and experience. Normalize all findings into the response schema. A tool error becomes status=failed and empty lists; never invent assets.""",
    )
    github_agent = _agent(
        model,
        tools.github,
        GithubRecon,
        f"""You are Sherlock's GitHub specialist. Search only org:{settings.github_org} for the supplied app. Find matching repositories and read pom.xml only during recon. Return repository URLs and Maven dependencies. Tool failures are recorded and do not stop the workflow.""",
    )
    gitbook_agent = _agent(
        model,
        tools.gitbook,
        GitbookRecon,
        """You are Sherlock's GitBook specialist. Find documentation for the supplied app. First call invoke_operation with operationId=listOrganizationsForAuthenticatedUser and params={} to obtain the organization ID. Then search organization content. If empty, list sites for that organization, inspect site structures, and fetch matching pages. Never call an organization-scoped tool before obtaining the organization ID. Return only supported findings. Tool failures are recorded and do not stop the workflow.""",
    )
    endpoint_agent = _agent(
        model,
        tools.github,
        EndpointAnalysis,
        """Analyze exactly one integration endpoint. Use the supplied Exchange evidence first and GitHub tools only for files needed to verify this endpoint's implementation. Return a concise Markdown section (maximum 450 words) headed '## Endpoint: <METHOD> <path>' with Purpose, Request, Response, Downstream Calls, DataWeave, Error Handling, and Security subsections. State unknown when evidence is absent; never invent details.""",
    )
    fallback_analysis_agent = _agent(
        model,
        tools.github,
        Intelligence,
        """You are Sherlock's integration intelligence analyst. Use Exchange/API evidence first, then GitHub tools to browse the matched repository from its root and inspect implementation files. Produce a detailed Markdown report with Overview and a separate section for every endpoint. Every endpoint section must cover Purpose, Request, Response, Downstream Calls, DataWeave, Error Handling, and Security. Include Dependencies, Configuration, and GitBook Docs sections. Never claim details unsupported by the supplied evidence; label unknowns explicitly.""",
    )
    async def mule_scan(state: SherlockState) -> dict:
        try:
            result = await _invoke(mule_agent, state["request"])
        except Exception as exc:
            result = MuleDiscovery(status="failed", error=str(exc))
        return {"mule": result}

    async def github_scan(state: SherlockState) -> dict:
        try:
            result = await _invoke(
                github_agent,
                f"Find the app named in this request and its repository: {state['request']}",
            )
        except Exception as exc:
            result = GithubRecon(status="failed", error=str(exc))
        return {"github": result}

    async def gitbook_scan(state: SherlockState) -> dict:
        try:
            result = await _invoke(
                gitbook_agent,
                f"Find documentation for the app named in this request: {state['request']}",
            )
        except Exception as exc:
            result = GitbookRecon(status="failed", error=str(exc))
        return {"gitbook": result}

    def reconcile(state: SherlockState) -> dict:
        """Prefer authoritative Mule names, with independent-source fallbacks."""
        apps = state["mule"].apps
        if not apps:
            apps = [App(name=repo.name) for repo in state["github"].repos]
        if not apps:
            apps = [App(name=doc.title) for doc in state["gitbook"].docs]
        unique = {app.name.casefold(): app for app in apps if app.name.strip()}
        return {"apps": list(unique.values())}

    def dispatch_endpoint_analysis(state: AppState) -> list[Send] | str:
        if not state["mule"].endpoints:
            return "analyze_fallback"
        shared = {
            "request": state["request"],
            "app": state["app"],
            "mule": state["mule"],
            "github": state["github"],
            "gitbook": state["gitbook"],
        }
        return [
            Send("analyze_endpoint", {**shared, "endpoint": endpoint})
            for endpoint in state["mule"].endpoints
        ]

    async def analyze_endpoint(state: AppState) -> dict:
        endpoint = state["endpoint"]
        matching_apis = [
            api.model_dump()
            for api in state["mule"].apis
            if not endpoint.asset_id or api.asset_id == endpoint.asset_id
        ]
        payload = json.dumps(
            {
                "request": state["request"],
                "app": state["app"].model_dump(),
                "endpoint": endpoint.model_dump(),
                "apis": matching_apis,
                "github": state["github"].model_dump(),
                "gitbook": state["gitbook"].model_dump(),
            },
            default=str,
        )
        error = ""
        for attempt in range(2):
            try:
                return {"endpoint_reports": [await _invoke(endpoint_agent, payload)]}
            except Exception as exc:
                error = f"attempt {attempt + 1}: {exc}"
                if attempt == 0:
                    await asyncio.sleep(1)
        return {
            "endpoint_reports": [
                EndpointAnalysis(
                    endpoint=endpoint,
                    report_markdown=(
                        f"## Endpoint: {endpoint.method} {endpoint.path}\n\n"
                        f"Analysis failed: {error}"
                    ),
                    status="failed",
                    error=error,
                )
            ]
        }

    async def analyze_fallback(state: AppState) -> dict:
        payload = json.dumps(
            {
                "request": state["request"],
                "app": state["app"].model_dump(),
                "mule": state["mule"].model_dump(),
                "github": state["github"].model_dump(),
                "gitbook": state["gitbook"].model_dump(),
            },
            default=str,
        )
        try:
            return {"intelligence": await _invoke(fallback_analysis_agent, payload)}
        except Exception as exc:
            return {"intelligence": Intelligence(
                app_name=state["app"].name,
                full_report_markdown=f"# {state['app'].name}\n\nAnalysis failed: {exc}",
                status="failed",
                error=str(exc),
            )}

    def assemble_analysis(state: AppState) -> dict:
        order = {
            (endpoint.method, endpoint.path): index
            for index, endpoint in enumerate(state["mule"].endpoints)
        }
        reports = sorted(
            state.get("endpoint_reports", []),
            key=lambda report: order.get(
                (report.endpoint.method, report.endpoint.path), len(order)
            ),
        )
        dependencies = [
            dependency
            for repo in state["github"].repos
            for dependency in repo.pom_dependencies
        ]
        docs = state["gitbook"].docs
        markdown = [
            f"# SHERLOCK Intelligence Report: {state['app'].name}",
            "",
            "## Overview",
            f"- Environment: {state['app'].environment or 'unknown'}",
            f"- Runtime status: {state['app'].status or 'unknown'}",
            f"- GitHub: {state['github'].repos[0].url if state['github'].repos else 'not found'}",
            "",
            *(report.report_markdown for report in reports),
            "",
            "## Dependencies",
            *(f"- `{item}`" for item in dependencies),
            *( ["- No Maven dependencies found."] if not dependencies else [] ),
            "",
            "## Configuration",
            "See the endpoint evidence above; values not present in source evidence are marked unknown.",
            "",
            "## GitBook Docs",
            *(f"- [{doc.title}]({doc.url})" for doc in docs),
            *( ["- No matching GitBook documentation found."] if not docs else [] ),
        ]
        failed = [report.error for report in reports if report.status == "failed"]
        return {
            "intelligence": Intelligence(
                app_name=state["app"].name,
                github_repo=state["github"].repos[0].url if state["github"].repos else "",
                endpoints=[report.endpoint for report in reports],
                connectors=sorted({item for report in reports for item in report.connectors}),
                outbound_calls=sorted({item for report in reports for item in report.outbound_calls}),
                full_report_markdown="\n".join(markdown),
                total_dataweave_transforms=sum(
                    report.dataweave_transforms for report in reports
                ),
                status="failed" if reports and len(failed) == len(reports) else "success",
                error="; ".join(failed),
            )
        }

    async def write_linear(state: AppState) -> dict:
        app_name = state["app"].name
        title = f"[KATE] Document: {app_name}"
        issue_id = ""
        issue_url = ""
        team_name = ""
        try:
            matches = await _call(
                tools.linear, "list_issues", {"query": title, "limit": 10}
            )
            issue = next(
                (item for item in matches.get("issues", []) if item.get("title") == title),
                None,
            )
            if issue is None:
                teams = await _call(tools.linear, "list_teams", {"limit": 10})
                team = teams.get("teams", [])[0]
                team_name = team.get("name", team.get("id", ""))
                issue = await _call(
                    tools.linear,
                    "save_issue",
                    {
                        "team": team_name,
                        "title": title,
                        "description": f"Sherlock intelligence report for {app_name}.",
                        "priority": 3,
                    },
                )

            issue_id = issue.get("identifier") or issue.get("id", "")
            issue_url = issue.get("url", "")
            issue_team = issue.get("team")
            if not team_name:
                team_name = (
                    issue_team.get("name", "")
                    if isinstance(issue_team, dict)
                    else issue_team or ""
                )
            details = await _call(tools.linear, "get_issue", {"id": issue_id})
            doc_title = f"SHERLOCK Intelligence Report: {app_name}"
            documents = details.get("documents", [])
            document_args = {
                "title": doc_title,
                "content": state["intelligence"].full_report_markdown,
                "issue": issue_id,
            }
            if documents:
                document_args["id"] = documents[0]["id"]
                document_args.pop("issue")
            await _call(tools.linear, "save_document", document_args)

            comments = await _call(
                tools.linear, "list_comments", {"issueId": issue_id, "limit": 50}
            )
            existing_comment = next(
                (
                    comment
                    for comment in comments.get("comments", [])
                    if comment.get("body", "").startswith("SHERLOCK:")
                ),
                None,
            )
            intel = state["intelligence"]
            comment_args = {
                "issueId": issue_id,
                "body": (
                    f"SHERLOCK: 🕵️ **{app_name}**\n"
                    f"Endpoints: {len(intel.endpoints)} | "
                    f"Connectors: {len(intel.connectors)} | "
                    f"DataWeave: {intel.total_dataweave_transforms}\n"
                    "Full intelligence report attached."
                ),
            }
            if existing_comment:
                comment_args["id"] = existing_comment["id"]
            await _call(tools.linear, "save_comment", comment_args)
            result = LinearResult(
                app_name=app_name,
                issue_id=issue_id,
                issue_url=issue_url,
                team_name=team_name,
            )
        except Exception as exc:
            result = LinearResult(
                app_name=app_name,
                issue_id=issue_id,
                issue_url=issue_url,
                team_name=team_name,
                status="failed",
                error=str(exc),
            )
        return {"linear": result}

    def collect(state: AppState) -> dict:
        return {
            "app_runs": [
                AppRun(
                    app=state["app"],
                    github=state["github"],
                    gitbook=state["gitbook"],
                    intelligence=state["intelligence"],
                    linear=state["linear"],
                )
            ]
        }

    app_builder = StateGraph(AppState, output_schema=AppOutput)
    app_builder.add_node("analysis_start", lambda _: {})
    app_builder.add_node("analyze_endpoint", analyze_endpoint)
    app_builder.add_node("assemble_analysis", assemble_analysis)
    app_builder.add_node("analyze_fallback", analyze_fallback)
    app_builder.add_node("write_linear", write_linear)
    app_builder.add_node("collect", collect)
    app_builder.add_edge(START, "analysis_start")
    app_builder.add_conditional_edges("analysis_start", dispatch_endpoint_analysis)
    app_builder.add_edge("analyze_endpoint", "assemble_analysis")
    app_builder.add_edge("assemble_analysis", "write_linear")
    app_builder.add_edge("analyze_fallback", "write_linear")
    app_builder.add_edge("write_linear", "collect")
    app_builder.add_edge("collect", END)
    app_graph = app_builder.compile()

    def summarize(state: SherlockState) -> dict:
        final = _catalog(state.get("app_runs", []))
        if not state.get("app_runs"):
            failures = [
                f"{name}: {result.error}"
                for name in ("mule", "github", "gitbook")
                if (result := state.get(name)) and result.status == "failed"
            ]
            if failures:
                final += "\n\nFailures:\n- " + "\n- ".join(failures)
        return {"final": final}

    def recon_start(_: SherlockState) -> dict:
        return {}

    builder = StateGraph(SherlockState)
    builder.add_node("validate", _validate)
    builder.add_node("recon_start", recon_start)
    builder.add_node("mule_scan", mule_scan)
    builder.add_node("github_scan", github_scan)
    builder.add_node("gitbook_scan", gitbook_scan)
    builder.add_node("reconcile", reconcile)
    builder.add_node("process_app", app_graph)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges(
        "validate", _after_validation, {"mule_scan": "recon_start", END: END}
    )
    builder.add_edge("recon_start", "mule_scan")
    builder.add_edge("recon_start", "github_scan")
    builder.add_edge("recon_start", "gitbook_scan")
    builder.add_edge(["mule_scan", "github_scan", "gitbook_scan"], "reconcile")
    builder.add_conditional_edges("reconcile", _dispatch_apps)
    builder.add_edge("process_app", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


async def make_graph():
    """LangGraph deployment factory. CLI keeps MCP sessions open for lower latency."""
    settings = Settings()
    model = create_model(settings)
    async with connect_toolsets(settings) as tools:
        return build_graph(model, tools, settings)
