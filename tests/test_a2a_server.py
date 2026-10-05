from starlette.testclient import TestClient

from sherlock.a2a_server import _status_text, create_app
from sherlock.config import Settings
from sherlock.models import App, MuleDiscovery


def test_agent_card_exposes_a2a_1_0_streaming_endpoint():
    settings = Settings(openai_api_key="test", _env_file=None)
    client = TestClient(create_app(settings))

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"] == [
        {
            "url": "http://localhost:8000/a2a/jsonrpc",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ]
    assert client.get("/health").json()["protocol"] == "A2A 1.0"


def test_langgraph_updates_become_a2a_progress_messages():
    result = MuleDiscovery(apps=[App(name="orders")], apis=[])
    assert _status_text("mule_scan", {"mule": result}) == (
        "Mule scan success: 1 apps, 0 APIs"
    )
