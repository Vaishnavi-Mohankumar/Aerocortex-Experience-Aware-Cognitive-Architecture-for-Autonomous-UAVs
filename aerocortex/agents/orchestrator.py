import time
from typing import List, Optional
from aerocortex.models import (
    TelemetryData,
    ContextSnapshot,
    MemoryEvidence,
    RecoveryPlan,
    SafetyVerdict,
    AgentExecutionTrace
)
from aerocortex.memory.working_memory import WorkingMemory
from aerocortex.memory.episodic_memory import EpisodicMemory
from aerocortex.memory.semantic_memory import SemanticMemory
from aerocortex.memory.knowledge_graph import KnowledgeGraph
from aerocortex.agents.situation_agent import SituationAgent
from aerocortex.agents.memory_agent import MemoryAgent
from aerocortex.agents.planner_agent import PlannerAgent
from aerocortex.agents.safety_agent import SafetyAgent
from aerocortex.agents.learning_agent import LearningAgent

class CognitiveOrchestrator:
    """
    AeroCortex Cognitive Orchestrator:
    Coordinates real-time telemetry processing, reactive circuit breaker (<50ms),
    and multi-agent state flow: Situation -> Memory -> Planner -> Safety.
    """
    def __init__(self):
        # Memory Layers
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.knowledge_graph = KnowledgeGraph()
        
        # Specialized Agents
        self.situation_agent = SituationAgent(self.working_memory)
        self.memory_agent = MemoryAgent(self.episodic_memory, self.semantic_memory, self.knowledge_graph)
        self.planner_agent = PlannerAgent()
        self.safety_agent = SafetyAgent()
        self.learning_agent = LearningAgent(self.episodic_memory, self.semantic_memory, self.knowledge_graph)

        self.execution_step: int = 0
        self.trace_history: List[AgentExecutionTrace] = []

    def process_telemetry(self, telemetry: TelemetryData, mission_id: str = "MISSION-01") -> AgentExecutionTrace:
        start_t = time.perf_counter()
        self.execution_step += 1
        
        # 1. Situation Assessment & Reactive Pre-filter
        context: ContextSnapshot = self.situation_agent.assess_situation(telemetry, mission_id=mission_id)
        report = context.anomaly

        # 2. Fast Reactive Circuit Breaker (< 50ms response)
        if report.reactive_action:
            verdict = SafetyVerdict(
                is_approved=True,
                final_action=report.reactive_action,
                is_overridden=False,
                safety_score=1.0,
                applied_constraints=["CRITICAL_CIRCUIT_BREAKER_TRIGGERED"],
                validation_latency_ms=0.5
            )
            total_time_ms = round((time.perf_counter() - start_t) * 1000, 3)
            
            trace = AgentExecutionTrace(
                timestamp=time.time(),
                mission_id=mission_id,
                step=self.execution_step,
                reactive_triggered=True,
                situation_report=report,
                retrieved_memory=None,
                recovery_plan=None,
                safety_verdict=verdict,
                final_action_executed=report.reactive_action,
                execution_time_total_ms=total_time_ms
            )
            self._record_trace(trace)
            return trace

        # 3. Memory Retrieval
        memory_evidence: MemoryEvidence = self.memory_agent.retrieve_memory(context)

        # 4. Cognitive Recovery Planning (Gemma 3 or deterministic reasoner)
        recovery_plan: RecoveryPlan = self.planner_agent.plan_recovery(context, memory_evidence)

        # 5. Safety Validation & Operational Constraints Check
        safety_verdict: SafetyVerdict = self.safety_agent.validate_plan(context, recovery_plan)

        total_time_ms = round((time.perf_counter() - start_t) * 1000, 3)
        
        trace = AgentExecutionTrace(
            timestamp=time.time(),
            mission_id=mission_id,
            step=self.execution_step,
            reactive_triggered=False,
            situation_report=report,
            retrieved_memory=memory_evidence,
            recovery_plan=recovery_plan,
            safety_verdict=safety_verdict,
            final_action_executed=safety_verdict.final_action,
            execution_time_total_ms=total_time_ms
        )
        self._record_trace(trace)
        return trace

    def _record_trace(self, trace: AgentExecutionTrace):
        self.trace_history.append(trace)
        if len(self.trace_history) > 200:
            self.trace_history.pop(0)

    def trigger_post_mission_learning(
        self,
        mission_id: str,
        anomaly_type: str,
        actions_taken: List[str],
        final_status: str,
        battery_remaining: float,
        context_desc: str,
        duration_s: float
    ):
        return self.learning_agent.process_mission_completion(
            mission_id=mission_id,
            anomaly_type=anomaly_type,
            actions_taken=actions_taken,
            final_status=final_status,
            battery_remaining=battery_remaining,
            context_description=context_desc,
            recovery_duration_s=duration_s
        )

    def reset_for_new_mission(self, mission_id: str):
        self.working_memory.clear()
        self.working_memory.mission_id = mission_id
        self.execution_step = 0
