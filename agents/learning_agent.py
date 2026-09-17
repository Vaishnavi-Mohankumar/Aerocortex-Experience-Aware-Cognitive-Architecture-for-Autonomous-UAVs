import time
from typing import Dict, Any, Optional
from models import (
    UAVTelemetry, SituationReport, RecoveryPlan, SafetyVerdict,
    EpisodicExperience, MissionOutcome
)
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph

class LearningAgent:
    """
    Learning Agent:
    Consolidates mission telemetry, failure events, recovery actions, and outcomes.
    Executes continual experience-aware learning by:
    1. Storing complete experiences in ChromaDB episodic memory.
    2. Reinforcing/updating semantic IF-THEN rules.
    3. Creating and weighting relationships in the Knowledge Graph.
    Enables future missions to retrieve and reuse this validated operational experience.
    """
    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None
    ):
        self.episodic_memory = episodic_memory or EpisodicMemory()
        self.semantic_memory = semantic_memory or SemanticMemory()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

    def process_mission_outcome(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        plan: RecoveryPlan,
        verdict: SafetyVerdict,
        success: bool = True,
        duration_s: float = 30.0,
        summary: str = ""
    ) -> Dict[str, Any]:
        # Only learn if an anomaly was faced and handled
        if not situation.anomaly_detected or situation.failure_type == "NONE":
            return {"status": "SKIPPED", "reason": "No anomaly detected in mission"}

        # Determine condition category
        condition_name = "MODERATE_WIND"
        if telemetry.wind_speed > 12.0:
            condition_name = "HIGH_WIND"
        elif telemetry.battery_level < 15.0:
            condition_name = "CRITICAL_BATTERY"
        elif telemetry.battery_level < 30.0:
            condition_name = "RESERVE_BATTERY"

        outcome_name = "MISSION_SUCCESS" if success else "MISSION_FAILURE"
        action_name = plan.action

        # 1. Commit Episode to ChromaDB
        experience = EpisodicExperience(
            mission_id=telemetry.mission_id,
            failure=situation.failure_type,
            context=f"{situation.description} at altitude {telemetry.altitude:.1f}m in {condition_name}",
            environmental_conditions={
                "altitude": telemetry.altitude,
                "velocity": telemetry.velocity,
                "battery_pct": telemetry.battery_level,
                "wind_speed": telemetry.wind_speed,
                "condition": condition_name
            },
            action=action_name,
            outcome=outcome_name,
            success=success,
            mission_duration=round(duration_s, 2),
            confidence=round(plan.confidence, 2),
            timestamp=time.time()
        )
        episode_id = self.episodic_memory.store_experience(experience)

        # 2. Update Semantic Memory IF-THEN Rules
        updated_rule = self.semantic_memory.reinforce_action(
            failure_type=situation.failure_type,
            action=action_name,
            success=success
        )

        # 3. Update Knowledge Graph Relationships (Neo4j / Embedded)
        self.knowledge_graph.add_mission_resolution(
            mission_id=telemetry.mission_id,
            failure=situation.failure_type,
            condition=condition_name,
            action=action_name,
            outcome=outcome_name,
            success=success
        )

        return {
            "status": "SUCCESS",
            "episode_id": episode_id,
            "failure_type": situation.failure_type,
            "action": action_name,
            "outcome": outcome_name,
            "rule_updated": updated_rule.rule_id if updated_rule else None,
            "summary": summary or f"Stored experience {episode_id} for {situation.failure_type} -> {action_name}"
        }
