import time
from aerocortex.config import settings
from aerocortex.models import TelemetryData, AnomalyReport, ContextSnapshot
from aerocortex.memory.working_memory import WorkingMemory

class SituationAgent:
    """
    1. Situation Agent:
    Monitors real-time telemetry, runs pre-filters and anomaly detection,
    evaluates deterministic fast triggers (< 50ms), and builds the mission context snapshot.
    """
    def __init__(self, working_memory: WorkingMemory):
        self.wm = working_memory

    def assess_situation(self, telemetry: TelemetryData, mission_id: str = "MISSION-01") -> ContextSnapshot:
        self.wm.update(telemetry)
        
        anomalies_detected = []
        severity = 0.0
        reactive_action = None
        
        # 1. Critical Battery Check (Fast Reactive Circuit Breaker)
        drain_rate = self.wm.calculate_battery_drain_rate()
        if telemetry.battery_pct <= settings.CRITICAL_BATTERY_LEVEL:
            anomalies_detected.append("BATTERY_SAG")
            severity = max(severity, 0.98)
            reactive_action = "CONTROLLED_EMERGENCY_LAND" # Critical emergency: land immediately!
        elif telemetry.battery_pct <= settings.MIN_SAFE_BATTERY_LEVEL or drain_rate > 0.2:
            anomalies_detected.append("BATTERY_SAG")
            severity = max(severity, 0.75)

        # 2. GPS Integrity Check
        if not telemetry.gps_valid or telemetry.gps_satellites < settings.MIN_GPS_SATELLITES or telemetry.gps_hdop > settings.MAX_GPS_HDOP:
            anomalies_detected.append("GPS_LOSS")
            severity = max(severity, 0.82)

        # 3. Communications Health
        if not telemetry.comms_connected or telemetry.comms_rssi_dbm < settings.MIN_COMMS_SIGNAL_DBM:
            anomalies_detected.append("COMMS_DROPOUT")
            severity = max(severity, 0.65)

        # 4. Severe Environmental Wind / Gusts
        if telemetry.wind_speed_ms > settings.MAX_PERMISSIBLE_WIND_SPEED:
            anomalies_detected.append("SEVERE_WIND_GUST")
            severity = max(severity, 0.78)

        # Classify Primary Anomaly
        if len(anomalies_detected) > 1:
            primary_anomaly = "COMPOUND_FAILURE"
            description = f"Multiple simultaneous anomalies detected: {', '.join(anomalies_detected)}"
        elif len(anomalies_detected) == 1:
            primary_anomaly = anomalies_detected[0]
            description = f"Operational anomaly detected: {primary_anomaly} (Severity: {severity:.2f})"
        else:
            primary_anomaly = "NONE"
            description = "Nominal UAV flight conditions"
            severity = 0.0

        requires_escalation = (primary_anomaly != "NONE" and reactive_action is None)
        
        report = AnomalyReport(
            anomaly_type=primary_anomaly,
            severity=round(severity, 2),
            description=description,
            metrics={
                "anomalies": anomalies_detected,
                "battery_drain_rate": round(drain_rate, 3),
                "altitude_agl": telemetry.z,
                "wind_speed": telemetry.wind_speed_ms,
                "gps_sats": telemetry.gps_satellites
            },
            requires_cognitive_escalation=requires_escalation,
            reactive_action=reactive_action
        )
        
        self.wm.active_anomaly = report
        
        # Flight Phase determination
        phase = "MISSION_NAV"
        if telemetry.z <= 0.5:
            phase = "LANDED"
        elif primary_anomaly != "NONE":
            phase = "ANOMALY_HANDLING"

        context = ContextSnapshot(
            mission_id=mission_id,
            flight_phase=phase,
            telemetry=telemetry,
            anomaly=report,
            environment={
                "wind_speed": telemetry.wind_speed_ms,
                "altitude": telemetry.z,
                "comms_dbm": telemetry.comms_rssi_dbm
            }
        )
        return context
