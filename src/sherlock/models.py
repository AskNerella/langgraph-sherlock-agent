from typing import Literal

from pydantic import BaseModel, Field


class App(BaseModel):
    name: str
    environment: str = ""
    status: str = ""
    deployment_url: str = ""
    mule_runtime: str = ""
    runtime_engine: str = ""


class Api(BaseModel):
    name: str
    version: str = ""
    asset_id: str = ""
    spec: str = ""
    status: str = ""
    instance_id: str = ""
    active_contracts: int | None = None


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


class ExternalApiCall(BaseModel):
    name: str
    api_type: Literal["system-api", "process-api"]
    source_file: str
    evidence: str = ""


class Repo(BaseModel):
    name: str
    url: str = ""
    owner: str = "AskNerella"
    mule_files: int = 0
    dataweave_files: int = 0
    inspected_files: list[str] = Field(default_factory=list)
    external_api_calls: list[ExternalApiCall] = Field(default_factory=list)


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


class ReconFindings(BaseModel):
    app_name: str
    report_markdown: str
    external_api_calls: list[ExternalApiCall] = Field(default_factory=list)


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
    findings: ReconFindings
    linear: LinearResult
