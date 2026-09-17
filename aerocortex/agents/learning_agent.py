import time
import uuid
from typing import Dict, Any, List
from aerocortex.models import MissionOutcome, EpisodeRecord
from aerocortex.memory.episodic_memory import EpisodicMemory
from aerocortex.memory.semantic_memory import SemanticMemory
from aerocortex.memory.knowledge_graph import KnowledgeGraph

class LearningAgent:
    """
    5. Learning Agent:
    Evaluates mission outcomes, extracts new operational experiences,
    updates Episodic Memory in ChromaDB, refines Semantic Rules in JSON,
    and updates relational paths in the Knowledge Graph for continual learning.
    """
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory, kg: KnowledgeGraph):
        self.episodic = episodic
        self.semantic = semantic
        self.kg = kg

    def process_mission_completion(
        self,
        mission_id: str,
        anomaly_type: str,
        actions_taken: List[str],
        final_status: str,
        battery_remaining: float,
        context_description: str,
        recovery_duration_s: float
    ) -> Dict[str, Any]:
        """
        Post-mission continual learning consolidation loop.
        """
        outcome_status = "SUCCESS"
        if "CRASHED" in final_status or final_status == "FAILURE":
            outcome_status = "FAILURE"
        elif "ABORTED" in final_status:
            outcome_status = "ABORTED"

        primary_action = actions_taken[-1] if actions_taken else "CONTINUE_MISSION"
        episode_id = f"EP-{str(uuid.uuid4())[:8].upper()}"

        # 1. Update Episodic Memory (ChromaDB)
        new_episode = EpisodeRecord(
            episode_id=episode_id,
            mission_id=mission_id,
            timestamp=time.time(),
            anomaly_type=anomaly_type,
            context_text=f"{context_description} (Outcome: {outcome_status})",
            action_taken=primary_action,
            outcome=outcome_status,
            time_to_stabilize_s=round(recovery_duration_s, 2),
            battery_remaining_pct=round(battery_remaining, 1),
            metadata={
                "final_status": final_status,
                "actions_count": len(actions_taken)
            }
        )
        self.episodic.add_episode(new_episode)

        # 2. Update Semantic Memory
        updated_rules = []
        is_success = (outcome_status == "SUCCESS")
        
        # Check existing rules to reinforce or penalize
        for r_id, r in self.semantic.rules.items():
            if r.anomaly_trigger == anomaly_type and r.recommended_action == primary_action:
                self.semantic.update_rule_outcome(r_id, success=is_success)
                updated_rules.append(r_id)

        # If novel successful action, extract new candidate rule
        if not updated_rules and is_success and anomaly_type != "NONE":
            extracted = self.semantic.extract_rule_from_episode(new_episode)
            updated_rules.append(extracted.rule_id)

        # 3. Update Knowledge Graph
        context_tag = "ANOMALOUS_CONDITIONS"
        if "wind" in context_description.lower():
            context_tag = "HIGH_WIND"
        elif "battery" in context_description.lower():
            context_tag = "LOW_BATTERY"

        self.kg.add_or_update_experience(
            anomaly_type=anomaly_type,
            context_label=context_tag,
            action=primary_action,
            outcome=outcome_status
        )

        return {
            "episode_id": episode_id,
            "outcome": outcome_status,
            "actions_consolidated": actions_taken,
            "updated_rules": updated_rules,
            "knowledge_graph_reinforced": True,
            "timestamp": time.time()
        }
