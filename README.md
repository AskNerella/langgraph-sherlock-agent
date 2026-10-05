# Sherlock on LangGraph

This replaces the sequential `sherlock.agent` pipeline with a recon-only fan-out/fan-in LangGraph:

```text
                           ┌→ Mule ───┐
validate → recon fan-out ──┼→ GitHub ─┼→ reconcile → Send(app 1..n)
                           └→ GitBook ┘                 └→ format findings → Linear
                                               all app branches join → catalog
```

Mule, GitHub, and GitBook search independently from the app name and execute concurrently. Mule is not a prerequisite for GitHub. Sherlock does not analyze endpoints or implementation code. GitHub recon reads only `pom.xml` and YAML files under config directories to identify external System API or Process API calls. The Linear issue description receives the complete asset findings directly.

## Configure

```bash
cp .env.example .env
uv sync --extra dev
```

Set `OPENAI_API_KEY`, `GITHUB_TOKEN`, `GITBOOK_MCP_URL`, and the credentials required by your GitBook/Linear servers. The supplied Mule URL and the official GitHub and Linear remote MCP URLs are already defaults. Tokens are sent as bearer credentials.

## Run

```bash
uv run sherlock "Discover and document the orders integration API"
```

For LangGraph Studio/API:

```bash
uv run langgraph dev
```

Run checks with `uv run pytest`.

## A2A 1.0 server

Set `A2A_PUBLIC_URL` to the externally reachable base URL, then start the server:

```bash
uv run sherlock-a2a
```

- Agent card: `GET /.well-known/agent-card.json`
- JSON-RPC endpoint: `POST /a2a/jsonrpc`
- Streaming: A2A `SendStreamingMessage` over SSE
- Health: `GET /health`

The stream emits the initial A2A `Task`, live status updates as each LangGraph node finishes, the final catalog artifact, and a terminal completed or failed status.
