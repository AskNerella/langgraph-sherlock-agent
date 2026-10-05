from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    openai_model: str = "gpt-5.6-luna"
    openai_reasoning_effort: str = "low"
    openai_verbosity: str = "low"
    mule_mcp_url: HttpUrl = HttpUrl(
        "https://mule-mcp-a4ie7e.rxfr6l.usa-e2.cloudhub.io/sherlock/mcp"
    )
    mule_mcp_token: str = ""
    github_mcp_url: HttpUrl = HttpUrl("https://api.githubcopilot.com/mcp/")
    github_token: str = ""
    github_org: str = "AskNerella"
    gitbook_mcp_url: HttpUrl | None = None
    gitbook_token: str = ""
    linear_mcp_url: HttpUrl = HttpUrl("https://mcp.linear.app/mcp")
    linear_api_key: str = ""
    sherlock_max_concurrency: int = 16
    a2a_host: str = "0.0.0.0"
    a2a_port: int = 8000
    a2a_public_url: str = "http://localhost:8000"
