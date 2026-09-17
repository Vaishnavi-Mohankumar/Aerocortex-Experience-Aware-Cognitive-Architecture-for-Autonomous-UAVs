from typing import Dict, Any, List, Optional
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aerocortex.config import settings
from aerocortex.models import TelemetryData, AgentExecutionTrace
from aerocortex.agents.orchestrator import CognitiveOrchestrator
from aerocortex.simulator.uav_dynamics import UAVDigitalTwin
from aerocortex.simulator.matlab_bridge import matlab_bridge

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AeroCortex Cognitive Memory & Multi-Agent UAV Recovery Gateway"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = CognitiveOrchestrator()
simulator = UAVDigitalTwin(mission_id="MISSION-ALPHA-01")

class FaultInjectionRequest(BaseModel):
    fault_type: str # gps_loss, gps_spoof, battery_sag, comms_dropout, severe_wind, all_clear
    enable: bool = True

class MissionCompleteRequest(BaseModel):
    mission_id: str
    anomaly_type: str
    actions_taken: List[str]
    final_status: str
    battery_remaining: float
    context_description: str
    duration_s: float

@app.get("/api/v1/health")
def health():
    return {
        "status": "HEALTHY",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "edge_platform": "Local-Edge-Architecture",
        "ollama_url": settings.OLLAMA_BASE_URL,
        "memory_status": {
            "episodic_count": orchestrator.episodic_memory.collection.count(),
            "semantic_rules_count": len(orchestrator.semantic_memory.rules),
            "kg_nodes": orchestrator.knowledge_graph.graph.number_of_nodes(),
            "kg_edges": orchestrator.knowledge_graph.graph.number_of_edges()
        }
    }

@app.post("/api/v1/telemetry")
def ingest_telemetry(data: Dict[str, Any]):
    """
    Primary endpoint for telemetry streams (Python UAV simulator or MATLAB/Simulink).
    Returns recovery action and safety verification.
    """
    # Parse whether packet is MATLAB format or standard
    if "battery" in data or "gps_sats" in data:
        telemetry = matlab_bridge.process_matlab_packet(data)
    else:
        telemetry = TelemetryData(**data)

    trace = orchestrator.process_telemetry(telemetry, mission_id=simulator.mission_id)
    
    # Return formatted actuation command
    return {
        "action": trace.final_action_executed,
        "is_reactive": trace.reactive_triggered,
        "anomaly_detected": trace.situation_report.anomaly_type if trace.situation_report else "NONE",
        "severity": trace.situation_report.severity if trace.situation_report else 0.0,
        "confidence": trace.recovery_plan.planner_confidence if trace.recovery_plan else 1.0,
        "safety_overridden": trace.safety_verdict.is_overridden if trace.safety_verdict else False,
        "override_reason": trace.safety_verdict.override_reason if trace.safety_verdict else None,
        "execution_time_ms": trace.execution_time_total_ms
    }

@app.post("/api/v1/simulator/step")
def step_simulator():
    """
    Executes one physical simulation step on the UAV Digital Twin
    and passes the resulting telemetry through the cognitive agent loop.
    """
    # 1. Step simulation
    telemetry = simulator.step(dt=settings.SIMULATION_DT)
    
    # 2. Process via Cognitive Multi-Agent Pipeline
    trace = orchestrator.process_telemetry(telemetry, mission_id=simulator.mission_id)
    
    # 3. Apply safety-validated action back to the UAV Digital Twin
    simulator.set_recovery_action(trace.final_action_executed)
    
    return {
        "telemetry": telemetry.model_dump(),
        "trace": trace.model_dump(),
        "mission_status": simulator.mission_status
    }

@app.post("/api/v1/simulator/reset")
def reset_simulator(mission_id: str = "MISSION-ALPHA-01"):
    simulator.mission_id = mission_id
    simulator.reset()
    orchestrator.reset_for_new_mission(mission_id)
    return {"status": "RESET", "mission_id": mission_id}

@app.post("/api/v1/fault_injection")
def inject_fault(req: FaultInjectionRequest):
    success = simulator.inject_failure(req.fault_type, req.enable)
    if not success:
        raise HTTPException(status_code=400, detail=f"Unknown fault type: {req.fault_type}")
    return {
        "status": "APPLIED",
        "fault_type": req.fault_type,
        "enabled": req.enable,
        "active_faults": simulator.injected_failures
    }

@app.get("/api/v1/memory/overview")
def get_memory_overview():
    return {
        "working_memory": {
            "current_state": orchestrator.working_memory.current_state.model_dump() if orchestrator.working_memory.current_state else None,
            "active_anomaly": orchestrator.working_memory.active_anomaly.model_dump(),
            "battery_drain_rate": orchestrator.working_memory.calculate_battery_drain_rate()
        },
        "episodic_memory": {
            "total_episodes": orchestrator.episodic_memory.collection.count(),
            "episodes": orchestrator.episodic_memory.get_all_episodes()
        },
        "semantic_memory": {
            "total_rules": len(orchestrator.semantic_memory.rules),
            "rules": orchestrator.semantic_memory.get_all_rules()
        },
        "knowledge_graph": orchestrator.knowledge_graph.get_elements_for_visualization()
    }

@app.post("/api/v1/mission/complete")
def complete_mission(req: MissionCompleteRequest):
    learning_result = orchestrator.trigger_post_mission_learning(
        mission_id=req.mission_id,
        anomaly_type=req.anomaly_type,
        actions_taken=req.actions_taken,
        final_status=req.final_status,
        battery_remaining=req.battery_remaining,
        context_desc=req.context_description,
        duration_s=req.duration_s
    )
    return {
        "status": "EXPERIENCE_CONSOLIDATED",
        "learning_result": learning_result
    }

@app.get("/api/v1/traces")
def get_traces():
    return [t.model_dump() for t in orchestrator.trace_history[-25:]]
