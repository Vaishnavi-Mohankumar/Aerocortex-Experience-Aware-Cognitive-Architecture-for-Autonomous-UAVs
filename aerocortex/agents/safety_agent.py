import time
from typing import List
from aerocortex.config import settings
from aerocortex.models import ContextSnapshot, RecoveryPlan, SafetyVerdict

class SafetyAgent:
    """
    4. Safety Agent:
    Validates AI-generated recovery plans against strict physical and operational flight constraints.
    Rejects or overrides unsafe, unfeasible, or low-confidence actions with validated fallback procedures.
    """
    def __init__(self):
        pass

    def validate_plan(self, context: ContextSnapshot, plan: RecoveryPlan) -> SafetyVerdict:
        start_t = time.time()
        telemetry = context.telemetry
        proposed = plan.proposed_action
        constraints_checked: List[str] = []
        is_overridden = False
        override_reason = None
        final_action = proposed
        safety_score = 1.0

        # Constraint 1: Critical Battery Floor (< 10%)
        constraints_checked.append("CRITICAL_BATTERY_RESERVE_CONSTRAINT")
        if telemetry.battery_pct <= settings.CRITICAL_BATTERY_LEVEL:
            if proposed != "CONTROLLED_EMERGENCY_LAND":
                is_overridden = True
                override_reason = (
                    f"SAFETY OVERRIDE: UAV battery at critical level ({telemetry.battery_pct:.1f}% <= "
                    f"{settings.CRITICAL_BATTERY_LEVEL}%). Proposed '{proposed}' rejected. Immediate emergency touchdown enforced."
                )
                final_action = "CONTROLLED_EMERGENCY_LAND"
                safety_score = 0.50

        # Constraint 2: Aerodynamic High-Wind Safety (> 14 m/s)
        constraints_checked.append("MAX_PERMISSIBLE_WIND_CONSTRAINT")
        if telemetry.wind_speed_ms > settings.MAX_PERMISSIBLE_WIND_SPEED:
            if proposed == "RETURN_TO_HOME" and telemetry.battery_pct < 30.0:
                is_overridden = True
                override_reason = (
                    f"SAFETY OVERRIDE: Flying RTH against severe winds ({telemetry.wind_speed_ms:.1f} m/s) "
                    f"with limited battery ({telemetry.battery_pct:.1f}%) risks mid-air depletion. Forcing low-altitude hold & descend."
                )
                final_action = "ALTITUDE_HOLD_DESCENT"
                safety_score = 0.65

        # Constraint 3: GPS Spoof / Dead-Reckoning Boundary Check
        constraints_checked.append("NAVIGATION_INTEGRITY_CONSTRAINT")
        if not telemetry.gps_valid and proposed == "CONTINUE_MISSION":
            is_overridden = True
            override_reason = "SAFETY OVERRIDE: Cannot continue nominal mission with invalid GPS. Switching to Inertial Dead-Reckoning."
            final_action = "SWITCH_INERTIAL_DEAD_RECKONING"
            safety_score = 0.40

        # Constraint 4: Low Confidence AI Action Check (< 0.70)
        constraints_checked.append("AI_CONFIDENCE_THRESHOLD_CONSTRAINT")
        if plan.planner_confidence < settings.CONFIDENCE_APPROVAL_THRESHOLD and not is_overridden:
            is_overridden = True
            override_reason = (
                f"SAFETY OVERRIDE: Planner confidence ({plan.planner_confidence:.2f}) below threshold "
                f"({settings.CONFIDENCE_APPROVAL_THRESHOLD:.2f}). Substituting with deterministic RTH fallback."
            )
            final_action = "RETURN_TO_HOME"
            safety_score = 0.70

        is_approved = not is_overridden
        latency = round((time.time() - start_t) * 1000, 2)
        
        return SafetyVerdict(
            is_approved=is_approved,
            final_action=final_action,
            is_overridden=is_overridden,
            override_reason=override_reason,
            safety_score=round(safety_score, 2),
            applied_constraints=constraints_checked,
            validation_latency_ms=latency
        )
