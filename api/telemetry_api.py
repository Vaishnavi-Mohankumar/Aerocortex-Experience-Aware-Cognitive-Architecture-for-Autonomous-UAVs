import time
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import UAVTelemetry
from orchestration.graph import AeroCortexGraph
from simulation.mission_simulator import MissionSimulator
from simulation.failure_scenarios import FailureScenarioInjector
from config import config

app = FastAPI(
    title="AeroCortex Telemetry API",
    description="Offline Edge Cognitive Memory & Flight Recovery Gateway",
    version=config.system.version
)

# Global singleton pipeline
graph = AeroCortexGraph()
simulator = MissionSimulator(graph=graph)
mission_log: List[Dict[str, Any]] = []

class SimulateRequest(BaseModel):
    scenario: str = "GPS_INTERFERENCE"
    steps: int = 1
    inject_step: int = 1

@app.get("/")
def root():
    return {
        "system": config.system.app_name,
        "version": config.system.version,
        "status": "ONLINE",
        "offline_mode": config.system.offline_mode
    }

@app.post("/telemetry")
def receive_telemetry(telemetry: UAVTelemetry):
    """
    Ingests live UAV telemetry (from MATLAB or flight controller),
    routes through LangGraph cognitive multi-agent pipeline,
    and returns safety-validated recovery action.
    """
    try:
        result = graph.run(telemetry)
        record = {
            "timestamp": time.time(),
            "mission_id": telemetry.mission_id,
            "situation": result.get("situation").model_dump() if result.get("situation") else None,
            "final_plan": result.get("final_plan").model_dump() if result.get("final_plan") else None,
            "safety_verdict": result.get("safety_verdict").model_dump() if result.get("safety_verdict") else None,
            "execution_status": result.get("execution_status", "UNKNOWN")
        }
        mission_log.append(record)
        return {
            "status": "SUCCESS",
            "action": record["final_plan"]["action"] if record["final_plan"] else "CONTINUE_MISSION",
            "is_approved": record["safety_verdict"]["approved"] if record["safety_verdict"] else True,
            "execution_status": record["execution_status"],
            "plan": record["final_plan"],
            "safety": record["safety_verdict"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_system_status():
    """
    Returns current active flight status, working memory snapshot, and anomaly state.
    """
    snapshot = graph.working_memory.get_snapshot()
    return {
        "status": "HEALTHY",
        "working_memory": snapshot,
        "kg_summary": graph.memory_agent.knowledge_graph.get_summary()
    }

@app.get("/memory")
def get_memory_state():
    """
    Inspects episodic experiences count, semantic rules, and knowledge graph structure.
    """
    rules = [r.model_dump() for r in graph.memory_agent.semantic_memory.get_all_rules()]
    kg_summary = graph.memory_agent.knowledge_graph.get_summary()
    chroma_count = graph.memory_agent.episodic_memory.vector_store.count()
    
    return {
        "episodic_experiences_count": chroma_count,
        "semantic_rules_count": len(rules),
        "semantic_rules": rules,
        "knowledge_graph": kg_summary
    }

@app.get("/missions")
def get_missions_history():
    """
    Returns historical telemetry events and recovery outcomes processed by the API.
    """
    return {
        "total_events": len(mission_log),
        "history": mission_log[-50:] # latest 50 events
    }

@app.post("/simulate")
def trigger_simulation(req: SimulateRequest):
    """
    Executes a simulated mission with specified failure scenario.
    """
    if req.scenario not in FailureScenarioInjector.list_available_scenarios():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Available: {FailureScenarioInjector.list_available_scenarios()}"
        )
    
    if req.steps == 1:
        step_res = simulator.run_step(scenario=req.scenario)
        return {"result": step_res}
    else:
        results = simulator.run_full_mission(
            total_steps=req.steps,
            inject_step=req.inject_step,
            scenario=req.scenario
        )
        return {"total_steps": len(results), "results": results}

@app.post("/reset")
def reset_state():
    """
    Resets working memory, simulation state, and runtime telemetry buffers.
    """
    graph.working_memory.clear()
    simulator.reset()
    mission_log.clear()
    return {"status": "SUCCESS", "message": "Working memory and simulation reset successfully"}
