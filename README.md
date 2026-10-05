# Sherlock on LangGraph

This replaces the sequential `sherlock.agent` pipeline with a real fan-out/fan-in LangGraph:

```text
                           ┌→ Mule ───┐
validate → recon fan-out ──┼→ GitHub ─┼→ reconcile → Send(app 1..n)
                           └→ GitBook ┘                 └→ analysis → Linear
                                               all app branches join → catalog
```

Mule, GitHub, and GitBook all search independently from the app name in the request and execute concurrently. Mule is not a prerequisite for GitHub. Their results fan in once, then all reconciled apps execute concurrently up to `SHERLOCK_MAX_CONCURRENCY`. Linear waits only for that app's analysis. One failed source is recorded without cancelling other work.

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

## What the timing test proves

The old graph's minimum wall time is approximately the sum of all five stages. This graph changes the critical path to roughly `max(Mule, GitHub, GitBook) + max(per-app analysis + Linear)`. It should reduce 1m37s when recon and multiple app scans are significant, but the exact gain must be measured against the live MCPs; LLM latency, MCP throttling, and a single large analysis can still dominate.
