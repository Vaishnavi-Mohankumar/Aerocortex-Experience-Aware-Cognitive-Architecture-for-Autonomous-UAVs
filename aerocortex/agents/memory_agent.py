import time
from typing import Dict, Any, List
from aerocortex.models import ContextSnapshot, MemoryEvidence
from aerocortex.memory.episodic_memory import EpisodicMemory
from aerocortex.memory.semantic_memory import SemanticMemory
from aerocortex.memory.knowledge_graph import KnowledgeGraph

class MemoryAgent:
    """
    2. Memory Agent:
    Performs hybrid retrieval across:
    - Episodic Memory (ChromaDB vector similarity search)
    - Semantic Memory (JSON IF-THEN rules)
    - Knowledge Graph (relational graph paths)
    Synthesizes and ranks contextual evidence for the Planner Agent.
    """
    def __init__(self, episodic_mem: EpisodicMemory, semantic_mem: SemanticMemory, knowledge_graph: KnowledgeGraph):
        self.episodic = episodic_mem
        self.semantic = semantic_mem
        self.kg = knowledge_graph

    def retrieve_memory(self, context: ContextSnapshot) -> MemoryEvidence:
        start_t = time.time()
        anomaly = context.anomaly.anomaly_type
        telemetry = context.telemetry
        
        if anomaly == "NONE":
            return MemoryEvidence(
                similar_episodes=[],
                matching_rules=[],
                kg_paths=[],
                composite_recommendation="CONTINUE_MISSION",
                memory_confidence=1.0,
                retrieval_latency_ms=round((time.time() - start_t) * 1000, 2)
            )

        # 1. Episodic Memory Retrieval (ChromaDB)
        query_text = (
            f"UAV {anomaly} failure at altitude {telemetry.z}m with wind {telemetry.wind_speed_ms}m/s "
            f"and battery {telemetry.battery_pct}%"
        )
        episodes = self.episodic.retrieve_similar(query_text, n_results=3)

        # 2. Semantic Rules Retrieval
        matched_rules = self.semantic.match_rules(context)
        rules_data = [r.model_dump() for r in matched_rules]

        # 3. Knowledge Graph Relational Paths
        context_keys = []
        if telemetry.wind_speed_ms > 10.0:
            context_keys.append("HIGH_WIND")
        else:
            context_keys.append("MODERATE_WIND")
            
        if telemetry.battery_pct < 20.0:
            context_keys.append("LOW_BATTERY")
        else:
            context_keys.append("MEDIUM_BATTERY")

        kg_recommendations = self.kg.query_recommendation(anomaly, context_keys)

        # 4. Synthesize Composite Recommendation & Confidence
        candidate_votes: Dict[str, float] = {}

        # Vote from rules (weight = 0.4)
        for r in matched_rules:
            act = r.recommended_action
            candidate_votes[act] = candidate_votes.get(act, 0.0) + (r.confidence * 0.45)

        # Vote from episodes (weight = 0.35)
        for ep in episodes:
            if ep["outcome"] == "SUCCESS":
                act = ep["action_taken"]
                sim = ep["similarity"]
                candidate_votes[act] = candidate_votes.get(act, 0.0) + (sim * 0.35)

        # Vote from Knowledge Graph (weight = 0.3)
        for kg in kg_recommendations:
            act = kg["action"]
            score = kg["score"]
            candidate_votes[act] = candidate_votes.get(act, 0.0) + (score * 0.3)

        if candidate_votes:
            best_action = max(candidate_votes, key=candidate_votes.get)
            confidence = min(0.98, candidate_votes[best_action] / 1.0)
        else:
            best_action = "RETURN_TO_HOME"
            confidence = 0.50

        latency = round((time.time() - start_t) * 1000, 2)
        return MemoryEvidence(
            similar_episodes=episodes,
            matching_rules=rules_data,
            kg_paths=kg_recommendations,
            composite_recommendation=best_action,
            memory_confidence=round(confidence, 2),
            retrieval_latency_ms=latency
        )
