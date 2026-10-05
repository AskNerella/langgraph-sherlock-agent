from typing import Literal

from pydantic import BaseModel, Field


class App(BaseModel):
    name: str
    environment: str = ""
    status: str = ""


class Api(BaseModel):
    name: str
    version: str = ""
    asset_id: str = ""
    spec: str = ""


class Endpoint(BaseModel):
    method: str
    path: str
    asset_id: str = ""
    source: str = ""


class MuleDiscovery(BaseModel):
    apps: list[App] = Field(default_factory=list)
    apis: list[Api] = Field(default_factory=list)
    exchange_assets: list[dict] = Field(default_factory=list)
    endpoints: list[Endpoint] = Field(default_factory=list)
    status: Literal["success", "failed"] = "success"
    error: str = ""


class Repo(BaseModel):
    name: str
    url: str = ""
    owner: str = "AskNerella"
    pom_dependencies: list[dict] = Field(default_factory=list)


class GithubRecon(BaseModel):
    repos: list[Repo] = Field(default_factory=list)
    status: Literal["success", "failed"] = "success"
    error: str = ""


class Document(BaseModel):
    title: str
    url: str = ""
    space_id: str = ""
    page_id: str = ""
    summary: str = ""


class GitbookRecon(BaseModel):
    docs: list[Document] = Field(default_factory=list)
    status: Literal["success", "failed"] = "success"
    error: str = ""


class Intelligence(BaseModel):
    app_name: str
    github_repo: str = ""
    runtime_version: str = ""
    endpoints: list[Endpoint] = Field(default_factory=list)
    connectors: list[str] = Field(default_factory=list)
    outbound_calls: list[str] = Field(default_factory=list)
    exchange_pages: list[str] = Field(default_factory=list)
    full_report_markdown: str
    total_dataweave_transforms: int = 0
    status: Literal["success", "failed"] = "success"
    error: str = ""


class EndpointAnalysis(BaseModel):
    endpoint: Endpoint
    report_markdown: str
    connectors: list[str] = Field(default_factory=list)
    outbound_calls: list[str] = Field(default_factory=list)
    dataweave_transforms: int = 0
    status: Literal["success", "failed"] = "success"
    error: str = ""


class LinearResult(BaseModel):
    app_name: str
    issue_id: str = ""
    issue_url: str = ""
    team_name: str = ""
    status: Literal["success", "failed"] = "success"
    error: str = ""


class AppRun(BaseModel):
    app: App
    github: GithubRecon
    gitbook: GitbookRecon
    intelligence: Intelligence
    linear: LinearResult
