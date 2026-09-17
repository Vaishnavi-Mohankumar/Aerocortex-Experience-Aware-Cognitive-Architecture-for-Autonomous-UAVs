import pytest
from models import UAVTelemetry, SituationReport, EpisodicExperience
from memory.working_memory import WorkingMemory
from memory.vector_store import VectorStore
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph
from agents.memory_agent import MemoryAgent

def test_working_memory():
    wm = WorkingMemory()
    t = UAVTelemetry(battery_level=80.0, altitude=60.0)
    wm.update_telemetry(t)
    
    snap = wm.get_snapshot()
    assert snap["latest_telemetry"]["battery_level"] == 80.0
    assert snap["battery_status"]["level"] == 80.0
    assert snap["history_length"] == 1
    
    wm.clear()
    assert wm.get_snapshot()["latest_telemetry"] is None

def test_vector_store_offline():
    vs = VectorStore(collection_name="test_vector_store")
    vs.reset()
    vs.add_documents(
        ids=["doc_1", "doc_2"],
        documents=["GPS failure during cruise in high wind", "Battery depleted emergency land"],
        metadatas=[{"tag": "gps"}, {"tag": "battery"}]
    )
    assert vs.count() == 2
    res = vs.query("GPS loss interference", n_results=1)
    assert len(res["ids"][0]) == 1
    assert res["ids"][0][0] == "doc_1"

def test_semantic_memory_matching():
    sm = SemanticMemory()
    rules = sm.match_rules("GPS_INTERFERENCE")
    assert len(rules) >= 1
    assert rules[0].trigger == "GPS_INTERFERENCE"
    assert "SWITCH_TO_VIO" in rules[0].action

def test_knowledge_graph_traversal():
    kg = KnowledgeGraph()
    paths = kg.query_action_relevance("GPS_INTERFERENCE")
    assert len(paths) >= 1
    assert any("SWITCH_TO_VIO" in p["action"] for p in paths)

def test_memory_agent_hybrid_scoring():
    mem_agent = MemoryAgent()
    t = UAVTelemetry(altitude=120.0, wind_speed=7.0)
    sit = SituationReport(
        anomaly_detected=True,
        failure_type="GPS_INTERFERENCE",
        severity="HIGH",
        mission_phase="CRUISE"
    )
    context = mem_agent.retrieve_context(t, sit)
    assert context.retrieved_experiences
    assert context.top_recommended_action is not None
    top_exp = context.retrieved_experiences[0]
    # Check hybrid score calculation bounds [0.0, 1.0]
    assert 0.0 <= top_exp.final_score <= 1.0
    assert 0.0 <= top_exp.vector_similarity <= 1.0
    assert 0.0 <= top_exp.graph_relevance <= 1.0
