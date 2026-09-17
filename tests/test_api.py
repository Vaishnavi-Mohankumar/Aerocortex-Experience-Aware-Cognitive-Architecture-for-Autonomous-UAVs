import pytest
from fastapi.testclient import TestClient
from api.telemetry_api import app
from models import UAVTelemetry

@pytest.fixture
def client():
    return TestClient(app)

def test_api_root(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert data["offline_mode"] is True

def test_api_telemetry_endpoint(client):
    t = UAVTelemetry(mission_id="API_TEST_001")
    res = client.post("/telemetry", json=t.model_dump())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "action" in data
    assert "is_approved" in data

def test_api_status_endpoint(client):
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "working_memory" in data

def test_api_memory_endpoint(client):
    res = client.get("/memory")
    assert res.status_code == 200
    data = res.json()
    assert "episodic_experiences_count" in data
    assert "semantic_rules_count" in data
    assert "knowledge_graph" in data

def test_api_simulate_endpoint(client):
    res = client.post("/simulate", json={"scenario": "GPS_INTERFERENCE", "steps": 1})
    assert res.status_code == 200
    data = res.json()
    assert "result" in data
    assert data["result"]["scenario"] == "GPS_INTERFERENCE"

def test_api_reset_endpoint(client):
    res = client.post("/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "SUCCESS"
