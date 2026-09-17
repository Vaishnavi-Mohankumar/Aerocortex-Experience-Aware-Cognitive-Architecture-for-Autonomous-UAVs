import pytest
from aerocortex.models import TelemetryData, RecoveryPlan
from aerocortex.memory.working_memory import WorkingMemory
from aerocortex.agents.situation_agent import SituationAgent
from aerocortex.agents.memory_agent import MemoryAgent
from aerocortex.agents.planner_agent import PlannerAgent
from aerocortex.agents.safety_agent import SafetyAgent
from aerocortex.agents.learning_agent import LearningAgent
from aerocortex.agents.orchestrator import CognitiveOrchestrator

def test_situation_agent_anomaly_detection():
    wm = WorkingMemory()
    sa = SituationAgent(wm)
    
    # Nominal telemetry
    nom_tele = TelemetryData(battery_pct=90.0, gps_valid=True, gps_satellites=12, wind_speed_ms=4.0)
    ctx1 = sa.assess_situation(nom_tele)
    assert ctx1.anomaly.anomaly_type == "NONE"
    assert ctx1.anomaly.requires_cognitive_escalation is False

    # GPS Loss
    gps_tele = TelemetryData(battery_pct=90.0, gps_valid=False, gps_satellites=2, wind_speed_ms=4.0)
    ctx2 = sa.assess_situation(gps_tele)
    assert ctx2.anomaly.anomaly_type == "GPS_LOSS"
    assert ctx2.anomaly.requires_cognitive_escalation is True

    # Fast Reactive Emergency (Critical Battery <= 10%)
    bat_tele = TelemetryData(battery_pct=8.0, gps_valid=True)
    ctx3 = sa.assess_situation(bat_tele)
    assert ctx3.anomaly.reactive_action == "CONTROLLED_EMERGENCY_LAND"

def test_safety_agent_overrides():
    safety = SafetyAgent()
    
    # Test case 1: Planner suggests CONTINUE_MISSION when GPS is lost -> MUST OVERRIDE
    ctx_gps = SituationAgent(WorkingMemory()).assess_situation(TelemetryData(gps_valid=False, gps_satellites=2))
    plan_unsafe = RecoveryPlan(proposed_action="CONTINUE_MISSION", planner_confidence=0.9)
    verdict1 = safety.validate_plan(ctx_gps, plan_unsafe)
    assert verdict1.is_overridden is True
    assert verdict1.final_action == "SWITCH_INERTIAL_DEAD_RECKONING"

    # Test case 2: Low confidence AI action (< 0.70) -> MUST OVERRIDE WITH RTH
    ctx_nom = SituationAgent(WorkingMemory()).assess_situation(TelemetryData())
    plan_low_conf = RecoveryPlan(proposed_action="CONTINUE_MISSION", planner_confidence=0.55)
    verdict2 = safety.validate_plan(ctx_nom, plan_low_conf)
    assert verdict2.is_overridden is True
    assert verdict2.final_action == "RETURN_TO_HOME"

def test_orchestrator_end_to_end_trace():
    orchestrator = CognitiveOrchestrator()
    
    # Step 1: Nominal
    tele_nom = TelemetryData()
    trace_nom = orchestrator.process_telemetry(tele_nom, mission_id="TEST-01")
    assert trace_nom.final_action_executed == "CONTINUE_MISSION"
    assert trace_nom.execution_time_total_ms >= 0.0
    
    # Step 2: Fault injected - GPS loss in high wind
    tele_fault = TelemetryData(gps_valid=False, gps_satellites=3, wind_speed_ms=13.5)
    trace_fault = orchestrator.process_telemetry(tele_fault, mission_id="TEST-01")
    assert trace_fault.situation_report.anomaly_type in ["GPS_LOSS", "COMPOUND_FAILURE"]
    assert trace_fault.final_action_executed in ["ALTITUDE_HOLD_DESCENT", "SWITCH_INERTIAL_DEAD_RECKONING"]
    assert trace_fault.safety_verdict.is_approved is True
    
    # Step 3: Fast reactive critical battery
    tele_crit = TelemetryData(battery_pct=7.5)
    trace_crit = orchestrator.process_telemetry(tele_crit, mission_id="TEST-01")
    assert trace_crit.reactive_triggered is True
    assert trace_crit.final_action_executed == "CONTROLLED_EMERGENCY_LAND"
