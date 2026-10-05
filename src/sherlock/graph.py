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
    Api,
    App,
    AppRun,
    GithubRecon,
    GitbookRecon,
    LinearResult,
    MuleDiscovery,
    ReconFindings,
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
    app: App
    mule: MuleDiscovery
    github: GithubRecon
    gitbook: GitbookRecon
    findings: ReconFindings
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
    terms = (
        "discover",
        "scan",
        "document",
        "identify",
        "api",
        "mule",
        "anypoint",
        "integration",
        "asset",
    )
    valid = bool(request) and any(term in request.lower() for term in terms)
    return {"valid": valid, "error": "" if valid else "Not an integration discovery task"}


def _after_validation(state: SherlockState) -> Literal["mule_scan", "__end__"]:
    return "mule_scan" if state["valid"] else END


def _dispatch_apps(state: SherlockState) -> list[Send] | str:
    if not state["apps"]:
        return "summarize"
    return [
        Send(
            "process_app",
            {
                "app": app,
                "mule": state["mule"],
                "github": state["github"],
                "gitbook": state["gitbook"],
            },
        )
        for app in state["apps"]
    ]


def _display_name(name: str) -> str:
    words = name.replace("_", "-").split("-")
    return " ".join("API" if word.casefold() == "api" else word.capitalize() for word in words)


def _matching_api(app: App, apis: list[Api]) -> Api | None:
    needle = app.name.casefold()
    return next(
        (
            api
            for api in apis
            if needle in api.name.casefold()
            or needle == api.asset_id.casefold()
            or (api.asset_id and api.asset_id.casefold() in needle)
        ),
        apis[0] if len(apis) == 1 else None,
    )


def _build_findings(
    app: App, mule: MuleDiscovery, github: GithubRecon, gitbook: GitbookRecon
) -> ReconFindings:
    api = _matching_api(app, mule.apis)
    repos = github.repos
    calls = [call for repo in repos for call in repo.external_api_calls]
    lines = [
        f"# {_display_name(app.name)} — Asset Findings",
        "",
        "## Anypoint",
        "",
        f"- **Application:** `{app.name}`",
        f"- **Environment:** {app.environment or 'Not found'}",
        f"- **Status:** {app.status or 'Not found'}",
        f"- **Deployment URL:** {app.deployment_url or 'Not found'}",
        f"- **Mule runtime:** {app.mule_runtime or 'Not found'}",
        f"- **API:** {api.name if api else 'Not found'}",
        f"- **Version:** {api.version if api and api.version else 'Not found'}",
        f"- **Asset ID:** `{api.asset_id}`" if api and api.asset_id else "- **Asset ID:** Not found",
        f"- **API status:** {api.status if api and api.status else 'Not found'}",
        f"- **API instance ID:** {api.instance_id if api and api.instance_id else 'Not found'}",
        f"- **Active contracts:** {api.active_contracts if api and api.active_contracts is not None else 'Not found'}",
        f"- **Runtime engine:** {app.runtime_engine or 'Not found'}",
        "",
        "## GitHub",
        "",
    ]
    if repos:
        lines.extend(
            [
                *(f"- **Repository:** [{repo.name}]({repo.url})" for repo in repos),
                f"- **Repositories:** {len(repos)}",
                f"- **Mule files:** {sum(repo.mule_files for repo in repos)}",
                f"- **DataWeave files:** {sum(repo.dataweave_files for repo in repos)}",
                "",
                "### External API calls",
                "",
                *(
                    f"- **{call.name}** ({call.api_type}) — `{call.source_file}`"
                    + (f": {call.evidence}" if call.evidence else "")
                    for call in calls
                ),
                *(
                    ["No external System API or Process API references were found in `pom.xml` or config YAML files."]
                    if not calls
                    else []
                ),
            ]
        )
    else:
        lines.extend(
            [
                "No matching GitHub repository was discovered for this app.",
                "",
                "- **Repositories:** 0",
                "- **Mule files:** 0",
                "- **DataWeave files:** 0",
            ]
        )

    lines.extend(["", "## GitBooks", ""])
    if gitbook.docs:
        for doc in gitbook.docs:
            lines.extend(
                [
                    f"- **Title:** {doc.title}",
                    f"- **URL:** {doc.url or 'Not found'}",
                    f"- **Space ID:** `{doc.space_id}`" if doc.space_id else "- **Space ID:** Not found",
                    f"- **Page ID:** `{doc.page_id}`" if doc.page_id else "- **Page ID:** Not found",
                    f"- **Summary:** {doc.summary or 'Not found'}",
                ]
            )
    else:
        lines.append("No matching GitBook documentation was discovered for this app.")

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This task documents the discovered Anypoint application and related API, GitHub, "
            f"and GitBooks findings for the {app.environment or 'requested'} environment.",
        ]
    )
    return ReconFindings(
        app_name=app.name,
        report_markdown="\n".join(lines),
        external_api_calls=calls,
    )


def _catalog(runs: list[AppRun]) -> str:
    if not runs:
        return "SHERLOCK: No applications were discovered."
    lines = ["# SHERLOCK Asset Catalog", ""]
    for run in runs:
        lines.extend(
            [
                f"## {run.app.name}",
                f"- Recon: GitHub {run.github.status}; GitBook {run.gitbook.status}",
                f"- External API calls: {len(run.findings.external_api_calls)}",
                f"- Linear: {run.linear.status} {run.linear.issue_url or run.linear.issue_id}"
                + (f" — {run.linear.error}" if run.linear.error else ""),
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
        """You are Sherlock's Anypoint recon specialist. Call get_mule_assets for the application named in the request. Return the application, environment, deployment status and URL, Mule runtime and engine, plus its API name, version, asset ID, status, instance ID, and active contract count. Do not analyze endpoints or implementation code. A tool error becomes status=failed with empty lists; never invent values.""",
    )
    github_agent = _agent(
        model,
        tools.github,
        GithubRecon,
        f"""You are Sherlock's GitHub recon specialist. Search only org:{settings.github_org} for a repository matching the application name. If no repository matches, return success with an empty repos list. If one matches, inspect only pom.xml and YAML/YML files inside config directories. Do not read Mule XML, DataWeave, Java, or other implementation source. Identify only external calls whose names or configured targets clearly reference a system-api or process-api. Record the source file and short evidence. Repository tree metadata may be used to count Mule XML and DataWeave files, but do not analyze them. Keep tool calls to the minimum needed and never invent findings.""",
    )
    gitbook_agent = _agent(
        model,
        tools.gitbook,
        GitbookRecon,
        """You are Sherlock's GitBook recon specialist. First call invoke_operation with operationId=listOrganizationsForAuthenticatedUser and params={} to obtain the organization ID. Search for documentation matching the application name and return title, URL, space ID, page ID, and a short summary. Do not perform implementation analysis. Tool errors are recorded and do not stop the workflow.""",
    )

    async def mule_scan(state: SherlockState) -> dict:
        try:
            result = await _invoke(mule_agent, state["request"])
        except Exception as exc:
            result = MuleDiscovery(status="failed", error=str(exc))
        return {"mule": result}

    async def github_scan(state: SherlockState) -> dict:
        try:
            result = await _invoke(github_agent, state["request"])
        except Exception as exc:
            result = GithubRecon(status="failed", error=str(exc))
        return {"github": result}

    async def gitbook_scan(state: SherlockState) -> dict:
        try:
            result = await _invoke(gitbook_agent, state["request"])
        except Exception as exc:
            result = GitbookRecon(status="failed", error=str(exc))
        return {"gitbook": result}

    def reconcile(state: SherlockState) -> dict:
        apps = state["mule"].apps
        if not apps:
            apps = [App(name=repo.name) for repo in state["github"].repos]
        if not apps:
            apps = [App(name=doc.title) for doc in state["gitbook"].docs]
        unique = {app.name.casefold(): app for app in apps if app.name.strip()}
        return {"apps": list(unique.values())}

    def assemble_findings(state: AppState) -> dict:
        return {
            "findings": _build_findings(
                state["app"], state["mule"], state["github"], state["gitbook"]
            )
        }

    async def write_linear(state: AppState) -> dict:
        app_name = state["app"].name
        title = f"[KATE] Document: {app_name}"
        issue_id = issue_url = team_name = ""
        try:
            matches = await _call(tools.linear, "list_issues", {"query": title, "limit": 10})
            issue = next(
                (item for item in matches.get("issues", []) if item.get("title") == title),
                None,
            )
            if issue:
                issue = await _call(
                    tools.linear,
                    "save_issue",
                    {
                        "id": issue.get("id") or issue.get("identifier"),
                        "description": state["findings"].report_markdown,
                    },
                )
            else:
                teams = await _call(tools.linear, "list_teams", {"limit": 10})
                team = teams.get("teams", [])[0]
                team_name = team.get("name", team.get("id", ""))
                issue = await _call(
                    tools.linear,
                    "save_issue",
                    {
                        "team": team_name,
                        "title": title,
                        "description": state["findings"].report_markdown,
                        "priority": 3,
                    },
                )
            issue_id = issue.get("identifier") or issue.get("id", "")
            issue_url = issue.get("url", "")
            issue_team = issue.get("team")
            if not team_name:
                team_name = issue_team.get("name", "") if isinstance(issue_team, dict) else issue_team or ""
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
                    findings=state["findings"],
                    linear=state["linear"],
                )
            ]
        }

    app_builder = StateGraph(AppState, output_schema=AppOutput)
    app_builder.add_node("assemble_findings", assemble_findings)
    app_builder.add_node("write_linear", write_linear)
    app_builder.add_node("collect", collect)
    app_builder.add_edge(START, "assemble_findings")
    app_builder.add_edge("assemble_findings", "write_linear")
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

    builder = StateGraph(SherlockState)
    builder.add_node("validate", _validate)
    builder.add_node("recon_start", lambda _: {})
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
    settings = Settings()
    async with connect_toolsets(settings) as tools:
        return build_graph(create_model(settings), tools, settings)
