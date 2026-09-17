import os
import time
import pytest
from aerocortex.models import TelemetryData, EpisodeRecord, ContextSnapshot, AnomalyReport
from aerocortex.memory.working_memory import WorkingMemory
from aerocortex.memory.episodic_memory import EpisodicMemory
from aerocortex.memory.semantic_memory import SemanticMemory
from aerocortex.memory.knowledge_graph import KnowledgeGraph

def test_working_memory():
    wm = WorkingMemory(window_size=5)
    t1 = TelemetryData(timestamp=100.0, battery_pct=95.0, z=40.0)
    t2 = TelemetryData(timestamp=101.0, battery_pct=94.0, z=42.0)
    wm.update(t1)
    wm.update(t2)
    
    assert wm.get_latest().battery_pct == 94.0
    drain = wm.calculate_battery_drain_rate()
    assert drain > 0.5 # ~1% per second
    alt_rate = wm.calculate_altitude_rate()
    assert alt_rate == pytest.approx(2.0, 0.1)

def test_episodic_memory(tmp_path):
    persist_dir = str(tmp_path / "test_chroma")
    em = EpisodicMemory(persist_dir=persist_dir)
    
    ep = EpisodeRecord(
        episode_id="EP-TEST-01",
        mission_id="M-TEST",
        timestamp=time.time(),
        anomaly_type="GPS_LOSS",
        context_text="High wind 12m/s GPS signal loss test",
        action_taken="ALTITUDE_HOLD_DESCENT",
        outcome="SUCCESS",
        time_to_stabilize_s=3.5,
        battery_remaining_pct=60.0
    )
    em.add_episode(ep)
    
    results = em.retrieve_similar("UAV GPS loss high wind", n_results=1)
    assert len(results) > 0
    assert results[0]["action_taken"] == "ALTITUDE_HOLD_DESCENT"
    assert results[0]["outcome"] == "SUCCESS"

def test_semantic_memory(tmp_path):
    rule_file = str(tmp_path / "rules.json")
    sm = SemanticMemory(file_path=rule_file)
    
    assert len(sm.rules) >= 5
    
    telemetry = TelemetryData(wind_speed_ms=12.0, battery_pct=85.0, gps_valid=False)
    anomaly = AnomalyReport(anomaly_type="GPS_LOSS", severity=0.8)
    ctx = ContextSnapshot(telemetry=telemetry, anomaly=anomaly)
    
    matches = sm.match_rules(ctx)
    assert len(matches) > 0
    assert matches[0].recommended_action in ["ALTITUDE_HOLD_DESCENT", "SWITCH_INERTIAL_DEAD_RECKONING"]
    
    # Test confidence update
    r_id = matches[0].rule_id
    initial_conf = sm.rules[r_id].confidence
    sm.update_rule_outcome(r_id, success=True)
    assert sm.rules[r_id].success_count > 0

def test_knowledge_graph(tmp_path):
    kg_file = str(tmp_path / "kg.json")
    kg = KnowledgeGraph(file_path=kg_file)
    
    assert kg.graph.number_of_nodes() > 10
    assert kg.graph.number_of_edges() > 5
    
    recs = kg.query_recommendation("GPS_LOSS", ["HIGH_WIND"])
    assert len(recs) > 0
    assert any(r["action"] == "ALTITUDE_HOLD_DESCENT" for r in recs)
    
    # Test adding experience
    kg.add_or_update_experience("GPS_LOSS", "HIGH_WIND", "ALTITUDE_HOLD_DESCENT", "SUCCESS")
    recs_after = kg.query_recommendation("GPS_LOSS", ["HIGH_WIND"])
    assert len(recs_after) > 0
