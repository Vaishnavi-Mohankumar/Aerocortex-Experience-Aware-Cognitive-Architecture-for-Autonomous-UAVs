from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class TelemetryData(BaseModel):
    timestamp: float = 0.0
    x: float = 0.0          # East (meters)
    y: float = 0.0          # North (meters)
    z: float = 45.0         # Altitude AGL (meters)
    vx: float = 0.0         # m/s
    vy: float = 8.0         # m/s forward speed
    vz: float = 0.0         # m/s vertical speed
    roll: float = 0.0       # degrees
    pitch: float = 2.0      # degrees
    yaw: float = 90.0       # degrees (heading North)
    battery_pct: float = 95.0
    gps_satellites: int = 12
    gps_hdop: float = 0.9
    gps_valid: bool = True
    comms_rssi_dbm: float = -62.0
    comms_connected: bool = True
    wind_speed_ms: float = 3.5
    wind_heading_deg: float = 45.0
    flight_mode: str = "MISSION_NAV"

class AnomalyReport(BaseModel):
    anomaly_type: str = "NONE" # GPS_LOSS, BATTERY_SAG, COMMS_DROPOUT, SEVERE_WIND_GUST, COMPOUND_FAILURE, NONE
    severity: float = 0.0       # 0.0 (nominal) to 1.0 (critical)
    description: str = "Nominal flight operating conditions"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    requires_cognitive_escalation: bool = False
    reactive_action: Optional[str] = None

class ContextSnapshot(BaseModel):
    mission_id: str = "M-001"
    flight_phase: str = "EN_ROUTE"
    telemetry: TelemetryData
    anomaly: AnomalyReport
    environment: Dict[str, Any] = Field(default_factory=dict)

class MemoryEvidence(BaseModel):
    similar_episodes: List[Dict[str, Any]] = Field(default_factory=list)
    matching_rules: List[Dict[str, Any]] = Field(default_factory=list)
    kg_paths: List[Dict[str, Any]] = Field(default_factory=list)
    composite_recommendation: Optional[str] = None
    memory_confidence: float = 0.0
    retrieval_latency_ms: float = 0.0

class RecoveryPlan(BaseModel):
    proposed_action: str = "CONTINUE_MISSION"
    rationale: str = "Maintain planned trajectory"
    target_parameters: Dict[str, Any] = Field(default_factory=dict)
    planner_confidence: float = 0.95
    reasoning_source: str = "Planner"
    plan_latency_ms: float = 0.0

class SafetyVerdict(BaseModel):
    is_approved: bool = True
    final_action: str = "CONTINUE_MISSION"
    is_overridden: bool = False
    override_reason: Optional[str] = None
    safety_score: float = 1.0
    applied_constraints: List[str] = Field(default_factory=list)
    validation_latency_ms: float = 0.0

class EpisodeRecord(BaseModel):
    episode_id: str
    mission_id: str
    timestamp: float
    anomaly_type: str
    context_text: str
    action_taken: str
    outcome: str # SUCCESS, FAILURE, ABORTED
    time_to_stabilize_s: float = 0.0
    battery_remaining_pct: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SemanticRule(BaseModel):
    rule_id: str
    anomaly_trigger: str
    context_criteria: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str
    confidence: float = 0.75
    success_count: int = 1
    failure_count: int = 0
    description: str = ""

class AgentExecutionTrace(BaseModel):
    timestamp: float
    mission_id: str
    step: int
    reactive_triggered: bool = False
    situation_report: Optional[AnomalyReport] = None
    retrieved_memory: Optional[MemoryEvidence] = None
    recovery_plan: Optional[RecoveryPlan] = None
    safety_verdict: Optional[SafetyVerdict] = None
    final_action_executed: str = "CONTINUE_MISSION"
    execution_time_total_ms: float = 0.0

class MissionOutcome(BaseModel):
    mission_id: str
    status: str # COMPLETED, SAFE_RECOVERY_LANDED, RETURNED_HOME, ABORTED, CRASHED
    duration_s: float
    battery_final_pct: float
    distance_traveled_m: float
    anomalies_encountered: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    safety_overrides_count: int = 0
    experience_updated: bool = False
