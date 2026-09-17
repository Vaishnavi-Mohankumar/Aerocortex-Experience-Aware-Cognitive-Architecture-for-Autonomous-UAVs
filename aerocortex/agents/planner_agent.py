import time
import json
import httpx
from typing import Dict, Any, Optional
from aerocortex.config import settings
from aerocortex.models import ContextSnapshot, MemoryEvidence, RecoveryPlan

class PlannerAgent:
    """
    3. Planner Agent:
    Reasons over current mission state and retrieved persistent memories to generate
    an optimal step-by-step recovery strategy using Gemma 3 via Ollama or edge deterministic reasoning.
    """
    def __init__(self, ollama_url: Optional[str] = None, model_name: Optional[str] = None):
        self.ollama_url = ollama_url or settings.OLLAMA_BASE_URL
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS

    def plan_recovery(self, context: ContextSnapshot, memory: MemoryEvidence) -> RecoveryPlan:
        start_t = time.time()
        
        # Nominal condition
        if context.anomaly.anomaly_type == "NONE":
            return RecoveryPlan(
                proposed_action="CONTINUE_MISSION",
                rationale="Telemetry nominal. Proceeding along scheduled mission waypoints.",
                target_parameters={"target_speed": 8.5, "target_altitude": context.telemetry.z},
                planner_confidence=0.98,
                reasoning_source="Nominal-Waypoint-Follower",
                plan_latency_ms=round((time.time() - start_t) * 1000, 2)
            )

        # Attempt to reason with Gemma 3 via Ollama
        gemma_plan = self._call_gemma_ollama(context, memory)
        if gemma_plan:
            gemma_plan.plan_latency_ms = round((time.time() - start_t) * 1000, 2)
            return gemma_plan

        # Edge Deterministic Cognitive Fallback (Slide 17, 26)
        return self._deterministic_cognitive_plan(context, memory, start_t)

    def _call_gemma_ollama(self, context: ContextSnapshot, memory: MemoryEvidence) -> Optional[RecoveryPlan]:
        prompt = (
            f"You are the AeroCortex UAV Recovery Planner Agent on an edge Raspberry Pi 5.\n"
            f"Current UAV State:\n"
            f"- Anomaly: {context.anomaly.anomaly_type} (Severity: {context.anomaly.severity})\n"
            f"- Battery: {context.telemetry.battery_pct}%\n"
            f"- Altitude: {context.telemetry.z} m AGL\n"
            f"- Wind Speed: {context.telemetry.wind_speed_ms} m/s\n"
            f"- GPS Satellites: {context.telemetry.gps_satellites}, Valid: {context.telemetry.gps_valid}\n"
            f"- Comms Signal: {context.telemetry.comms_rssi_dbm} dBm\n\n"
            f"Retrieved Cognitive Memories:\n"
            f"- Composite Recommendation: {memory.composite_recommendation}\n"
            f"- Top Rule: {memory.matching_rules[0]['recommended_action'] if memory.matching_rules else 'None'}\n"
            f"- Top Episode: {memory.similar_episodes[0]['action_taken'] if memory.similar_episodes else 'None'}\n\n"
            f"Select the single best recovery action from:\n"
            f"[SWITCH_INERTIAL_DEAD_RECKONING, ALTITUDE_HOLD_DESCENT, RETURN_TO_HOME, CONTROLLED_EMERGENCY_LAND, POWER_CONSERVATIVE_LOITER]\n"
            f"Respond ONLY in valid JSON format:\n"
            f'{{"proposed_action": "<ACTION>", "rationale": "<BRIEF_EXPLANATION>", "confidence": 0.85, "target_parameters": {{"target_altitude": 25.0}}}}'
        )

        try:
            # Fast connect timeout (0.2s) ensures offline edge systems do not block waiting for Ollama
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=0.2)) as client:
                res = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "")
                    parsed = json.loads(response_text)
                    return RecoveryPlan(
                        proposed_action=parsed.get("proposed_action", memory.composite_recommendation or "RETURN_TO_HOME"),
                        rationale=parsed.get("rationale", "Gemma-3 inferred optimal recovery based on memory."),
                        target_parameters=parsed.get("target_parameters", {}),
                        planner_confidence=float(parsed.get("confidence", 0.85)),
                        reasoning_source=f"Gemma-3 ({self.model_name})"
                    )
        except Exception:
            # Ollama not reachable or timed out - seamlessly proceed to deterministic reasoner
            pass
        return None

    def _deterministic_cognitive_plan(self, context: ContextSnapshot, memory: MemoryEvidence, start_t: float) -> RecoveryPlan:
        telemetry = context.telemetry
        anomaly = context.anomaly.anomaly_type
        rec = memory.composite_recommendation or "RETURN_TO_HOME"
        conf = max(0.72, memory.memory_confidence)
        
        target_params = {}
        rationale = ""

        if rec == "SWITCH_INERTIAL_DEAD_RECKONING":
            target_params = {"target_speed": 4.0, "nav_mode": "IMU_INTEGRATION"}
            rationale = (
                f"Retrieved prior episodes showing successful recovery using inertial dead-reckoning "
                f"during {anomaly} under moderate wind conditions ({telemetry.wind_speed_ms:.1f} m/s)."
            )
        elif rec == "ALTITUDE_HOLD_DESCENT":
            target_params = {"target_altitude": 25.0, "descent_rate": 1.8}
            rationale = (
                f"Episodic and Semantic memory advise descending from {telemetry.z:.1f}m to low-drag altitude (25m) "
                f"to reduce exposure to high shear winds ({telemetry.wind_speed_ms:.1f} m/s)."
            )
        elif rec == "CONTROLLED_EMERGENCY_LAND":
            target_params = {"descent_rate": 1.2, "target_altitude": 0.0}
            rationale = (
                f"Knowledge Graph and Battery Rule mandate immediate emergency touchdown "
                f"due to critical battery level ({telemetry.battery_pct:.1f}%)."
            )
        elif rec == "RETURN_TO_HOME":
            target_params = {"cruise_speed": 7.0, "heading_target": "BASE"}
            rationale = (
                f"Safe autonomous return to origin activated for {anomaly}; "
                f"reserve battery ({telemetry.battery_pct:.1f}%) is sufficient for return corridor."
            )
        else:
            target_params = {"loiter_radius": 15.0}
            rationale = f"Holding station at reduced power consumption to reassess {anomaly}."

        return RecoveryPlan(
            proposed_action=rec,
            rationale=rationale,
            target_parameters=target_params,
            planner_confidence=conf,
            reasoning_source="Deterministic-Cognitive-Reasoner (Edge Fallback)",
            plan_latency_ms=round((time.time() - start_t) * 1000, 2)
        )
