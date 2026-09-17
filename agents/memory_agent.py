import time
from typing import List, Dict, Any, Optional
from config import config
from models import (
    UAVTelemetry, SituationReport, EpisodicExperience, RetrievedExperience,
    SemanticRule, HybridMemoryContext
)
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph

class MemoryAgent:
    """
    Memory Agent:
    Coordinates Hybrid Memory Retrieval across:
    1. ChromaDB Vector Store (Episodic Experience)
    2. Neo4j / Embedded Knowledge Graph (Relational Paths)
    3. Semantic Memory (Operational IF-THEN Rules)
    
    Computes composite relevance score:
    final_score = w_vector * vector_similarity + w_graph * graph_relevance
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
        
        self.w_vector = config.memory.hybrid_weights.vector_similarity
        self.w_graph = config.memory.hybrid_weights.graph_relevance
        self.top_k = config.memory.top_k_episodes

    def retrieve_context(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport
    ) -> HybridMemoryContext:
        start_t = time.time()
        
        if not situation.anomaly_detected or situation.failure_type == "NONE":
            return HybridMemoryContext(
                retrieved_experiences=[],
                semantic_rules=[],
                graph_paths=[],
                retrieval_latency_ms=0.0,
                top_recommended_action="CONTINUE_MISSION"
            )

        # 1. Episodic Retrieval from ChromaDB
        query_text = (
            f"Failure: {situation.failure_type}. "
            f"Altitude: {telemetry.altitude}m, Wind: {telemetry.wind_speed}m/s, "
            f"Battery: {telemetry.battery_level}%, State: {situation.mission_phase}"
        )
        raw_episodes = self.episodic_memory.retrieve_similar_experiences(query_text, top_k=self.top_k)

        # 2. Knowledge Graph Relational Query
        # Find condition keyword (e.g. HIGH_WIND, CRITICAL_BATTERY, MODERATE_WIND)
        cond_hint = "MODERATE_WIND"
        if telemetry.wind_speed > 12.0:
            cond_hint = "HIGH_WIND"
        elif telemetry.battery_level < 15.0:
            cond_hint = "CRITICAL_BATTERY"
        elif telemetry.battery_level < 30.0:
            cond_hint = "RESERVE_BATTERY"

        kg_candidates = self.knowledge_graph.query_action_relevance(
            situation.failure_type,
            condition=cond_hint
        )
        kg_score_map = {item["action"]: item["graph_relevance"] for item in kg_candidates}

        # 3. Hybrid Scoring & Ranking of Experiences
        ranked_experiences: List[RetrievedExperience] = []
        for ep_data in raw_episodes:
            exp: EpisodicExperience = ep_data["experience"]
            vec_sim = ep_data["vector_similarity"]
            # Lookup corresponding graph relevance for this action
            graph_rel = kg_score_map.get(exp.action, 0.5)
            
            final_score = round(self.w_vector * vec_sim + self.w_graph * graph_rel, 4)
            
            ranked_experiences.append(RetrievedExperience(
                experience=exp,
                vector_similarity=vec_sim,
                graph_relevance=graph_rel,
                final_score=final_score
            ))

        # Sort experiences by hybrid score descending
        ranked_experiences.sort(key=lambda x: x.final_score, reverse=True)

        # 4. Semantic Rules Matching
        matched_rules = self.semantic_memory.match_rules(situation.failure_type, telemetry)

        # Determine composite recommendation
        top_action = None
        if ranked_experiences and ranked_experiences[0].final_score > 0.65:
            top_action = ranked_experiences[0].experience.action
        elif matched_rules:
            top_action = matched_rules[0].action
        elif kg_candidates:
            top_action = kg_candidates[0]["action"]
        else:
            top_action = "RETURN_TO_HOME"

        latency = round((time.time() - start_t) * 1000, 2)

        return HybridMemoryContext(
            retrieved_experiences=ranked_experiences,
            semantic_rules=matched_rules,
            graph_paths=kg_candidates,
            retrieval_latency_ms=latency,
            top_recommended_action=top_action
        )
