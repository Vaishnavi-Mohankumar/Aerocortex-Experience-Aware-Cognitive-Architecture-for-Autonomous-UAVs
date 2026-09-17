"""
MATLAB & Simulink Interface Bridge for AeroCortex.
Allows external MATLAB/Simulink simulations to stream telemetry and receive cognitive recovery commands.
"""
from typing import Dict, Any
from aerocortex.models import TelemetryData, SafetyVerdict

class MatlabBridge:
    def __init__(self):
        self.connected_clients: int = 0
        self.last_telemetry: Dict[str, Any] = {}
        self.last_command: str = "CONTINUE_MISSION"

    def process_matlab_packet(self, packet: Dict[str, Any]) -> TelemetryData:
        """
        Parses MATLAB / Simulink UAV Toolbox standard state bus:
        [x, y, z, roll, pitch, yaw, vx, vy, vz, battery, gps_sats, hdop, rssi, wind]
        """
        return TelemetryData(
            timestamp=packet.get("timestamp", 0.0),
            x=packet.get("x", 0.0),
            y=packet.get("y", 0.0),
            z=packet.get("z", 40.0),
            vx=packet.get("vx", 0.0),
            vy=packet.get("vy", 0.0),
            vz=packet.get("vz", 0.0),
            roll=packet.get("roll", 0.0),
            pitch=packet.get("pitch", 0.0),
            yaw=packet.get("yaw", 0.0),
            battery_pct=packet.get("battery", 100.0),
            gps_satellites=int(packet.get("gps_sats", 12)),
            gps_hdop=packet.get("hdop", 1.0),
            gps_valid=packet.get("gps_valid", True),
            comms_rssi_dbm=packet.get("rssi", -65.0),
            comms_connected=packet.get("comms_ok", True),
            wind_speed_ms=packet.get("wind_speed", 3.0),
            wind_heading_deg=packet.get("wind_heading", 0.0),
            flight_mode=packet.get("mode", "NAV")
        )

    def format_action_for_simulink(self, verdict: SafetyVerdict) -> Dict[str, Any]:
        """
        Translates AeroCortex action string into numeric actuation codes for Simulink Stateflow:
        0: HOLD / CONTINUE
        1: INERTIAL_DEAD_RECKONING
        2: ALTITUDE_HOLD_DESCENT
        3: RETURN_TO_HOME
        4: CONTROLLED_EMERGENCY_LAND
        5: POWER_CONSERVATIVE_LOITER
        """
        action_map = {
            "CONTINUE_MISSION": 0,
            "SWITCH_INERTIAL_DEAD_RECKONING": 1,
            "ALTITUDE_HOLD_DESCENT": 2,
            "RETURN_TO_HOME": 3,
            "CONTROLLED_EMERGENCY_LAND": 4,
            "POWER_CONSERVATIVE_LOITER": 5
        }
        cmd_code = action_map.get(verdict.final_action, 0)
        return {
            "action_code": cmd_code,
            "action_name": verdict.final_action,
            "is_overridden": verdict.is_overridden,
            "safety_score": verdict.safety_score,
            "applied_constraints": verdict.applied_constraints
        }

matlab_bridge = MatlabBridge()
