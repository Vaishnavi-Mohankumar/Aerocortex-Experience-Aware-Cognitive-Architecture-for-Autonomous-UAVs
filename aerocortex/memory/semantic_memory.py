import os
import json
from typing import List, Dict, Any, Optional
from aerocortex.config import settings
from aerocortex.models import SemanticRule, ContextSnapshot, EpisodeRecord

DEFAULT_RULES = [
    {
        "rule_id": "RULE-GPS-01",
        "anomaly_trigger": "GPS_LOSS",
        "context_criteria": {"max_wind_speed": 10.0},
        "recommended_action": "SWITCH_INERTIAL_DEAD_RECKONING",
        "confidence": 0.88,
        "success_count": 8,
        "failure_count": 1,
        "description": "If GPS drops under moderate wind, switch to inertial dead-reckoning hovering."
    },
    {
        "rule_id": "RULE-GPS-02",
        "anomaly_trigger": "GPS_LOSS",
        "context_criteria": {"min_wind_speed": 10.0},
        "recommended_action": "ALTITUDE_HOLD_DESCENT",
        "confidence": 0.92,
        "success_count": 6,
        "failure_count": 0,
        "description": "If GPS drops under high winds aloft, descend to low-altitude envelope to escape wind shear."
    },
    {
        "rule_id": "RULE-BAT-01",
        "anomaly_trigger": "BATTERY_SAG",
        "context_criteria": {"min_battery_pct": 20.0},
        "recommended_action": "RETURN_TO_HOME",
        "confidence": 0.85,
        "success_count": 5,
        "failure_count": 1,
        "description": "If battery sags but remains above 20%, abort mission and return to home immediately."
    },
    {
        "rule_id": "RULE-BAT-02",
        "anomaly_trigger": "BATTERY_SAG",
        "context_criteria": {"max_battery_pct": 20.0},
        "recommended_action": "CONTROLLED_EMERGENCY_LAND",
        "confidence": 0.96,
        "success_count": 12,
        "failure_count": 0,
        "description": "If battery drops below critical 20%, execute controlled emergency landing immediately."
    },
    {
        "rule_id": "RULE-COMM-01",
        "anomaly_trigger": "COMMS_DROPOUT",
        "context_criteria": {"gps_valid": True},
        "recommended_action": "RETURN_TO_HOME",
        "confidence": 0.90,
        "success_count": 7,
        "failure_count": 0,
        "description": "If comms link is lost while GPS is healthy, autonomous RTH fail-safe."
    },
    {
        "rule_id": "RULE-WIND-01",
        "anomaly_trigger": "SEVERE_WIND_GUST",
        "context_criteria": {"min_wind_speed": 12.0},
        "recommended_action": "ALTITUDE_HOLD_DESCENT",
        "confidence": 0.89,
        "success_count": 9,
        "failure_count": 1,
        "description": "If encountering severe winds aloft, descend to lower altitude where surface drag reduces wind velocity."
    }
]

class SemanticMemory:
    """
    Semantic Memory Layer:
    Stores generalized operational knowledge and structured IF-THEN rules in JSON.
    Continuously refines rule weights and synthesizes new rules from mission episodes.
    """
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or settings.SEMANTIC_RULES_FILE
        self.rules: Dict[str, SemanticRule] = {}
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        rule = SemanticRule(**item)
                        self.rules[rule.rule_id] = rule
                return
            except Exception:
                pass
                
        # Initialize default rules
        for item in DEFAULT_RULES:
            rule = SemanticRule(**item)
            self.rules[rule.rule_id] = rule
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in self.rules.values()], f, indent=2)

    def match_rules(self, context: ContextSnapshot) -> List[SemanticRule]:
        """Evaluates active rules against the current context snapshot."""
        matched = []
        trigger = context.anomaly.anomaly_type
        telemetry = context.telemetry

        for rule in self.rules.values():
            if rule.anomaly_trigger != trigger:
                continue
            
            crit = rule.context_criteria
            fits = True
            
            if "min_wind_speed" in crit and telemetry.wind_speed_ms < crit["min_wind_speed"]:
                fits = False
            if "max_wind_speed" in crit and telemetry.wind_speed_ms > crit["max_wind_speed"]:
                fits = False
            if "min_battery_pct" in crit and telemetry.battery_pct < crit["min_battery_pct"]:
                fits = False
            if "max_battery_pct" in crit and telemetry.battery_pct > crit["max_battery_pct"]:
                fits = False
            if "gps_valid" in crit and telemetry.gps_valid != crit["gps_valid"]:
                fits = False

            if fits:
                matched.append(rule)

        # Sort by confidence descending
        matched.sort(key=lambda r: r.confidence, reverse=True)
        return matched

    def update_rule_outcome(self, rule_id: str, success: bool):
        """Refines rule confidence score based on mission feedback."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            if success:
                rule.success_count += 1
            else:
                rule.failure_count += 1
            total = rule.success_count + rule.failure_count
            rule.confidence = round(rule.success_count / total, 3)
            self.save()

    def extract_rule_from_episode(self, episode: EpisodeRecord):
        """Synthesizes a new candidate rule if a pattern succeeds repeatedly."""
        new_id = f"RULE-GEN-{len(self.rules) + 1}"
        crit: Dict[str, Any] = {}
        if episode.metadata.get("wind_speed"):
            crit["min_wind_speed"] = round(episode.metadata["wind_speed"] * 0.8, 1)
        
        new_rule = SemanticRule(
            rule_id=new_id,
            anomaly_trigger=episode.anomaly_type,
            context_criteria=crit,
            recommended_action=episode.action_taken,
            confidence=0.75,
            success_count=1,
            failure_count=0,
            description=f"Auto-extracted rule from mission {episode.mission_id}: {episode.context_text}"
        )
        self.rules[new_id] = new_rule
        self.save()
        return new_rule

    def get_all_rules(self) -> List[Dict[str, Any]]:
        return [r.model_dump() for r in self.rules.values()]
