import time
import math
import random
from typing import Dict, Any, Optional
from aerocortex.models import TelemetryData

class UAVDigitalTwin:
    """
    High-Fidelity UAV Digital Twin simulating physical dynamics,
    sensor degradation, environmental disturbances, and control responses.
    """
    def __init__(self, mission_id: str = "MISSION-ALPHA-01"):
        self.mission_id = mission_id
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.sim_time = 0.0
        
        # Position & Velocity (NED-like / Local Cartesian)
        self.x = 0.0          # East (meters)
        self.y = 0.0          # North (meters)
        self.z = 45.0         # Altitude AGL (meters)
        self.vx = 0.0         # m/s
        self.vy = 8.0         # m/s forward speed
        self.vz = 0.0         # m/s
        
        # Attitude (Degrees)
        self.roll = 0.0
        self.pitch = 2.0
        self.yaw = 90.0       # Heading North
        
        # Power & Subsystems
        self.battery_pct = 98.0
        self.battery_drain_rate = 0.05 # % per second baseline
        
        # Navigation & Comms Sensors
        self.gps_satellites = 14
        self.gps_hdop = 0.8
        self.gps_valid = True
        self.comms_rssi_dbm = -60.0
        self.comms_connected = True
        
        # Environmental Wind
        self.wind_speed_ms = 4.0
        self.wind_heading_deg = 45.0
        
        # Waypoints for mission
        self.waypoints = [
            (0.0, 100.0, 45.0),
            (200.0, 300.0, 50.0),
            (400.0, 500.0, 45.0),
            (500.0, 800.0, 40.0)
        ]
        self.current_wp_idx = 0
        self.flight_mode = "MISSION_NAV"
        
        # Active Failure Injections
        self.injected_failures = {
            "gps_loss": False,
            "gps_spoof": False,
            "battery_sag": False,
            "comms_dropout": False,
            "severe_wind": False
        }
        self.active_recovery_action: Optional[str] = None
        self.action_execution_time = 0.0
        self.mission_status = "RUNNING" # RUNNING, COMPLETED, SAFE_RECOVERY_LANDED, RETURNED_HOME, ABORTED

    def inject_failure(self, failure_type: str, enable: bool = True):
        ft = failure_type.lower()
        if ft in self.injected_failures:
            self.injected_failures[ft] = enable
            return True
        elif ft == "all_clear":
            for k in self.injected_failures:
                self.injected_failures[k] = False
            return True
        return False

    def set_recovery_action(self, action: str):
        self.active_recovery_action = action
        self.action_execution_time = 0.0
        self.flight_mode = action

    def step(self, dt: float = 0.1) -> TelemetryData:
        self.sim_time += dt
        self.action_execution_time += dt
        
        # 1. Simulate Environmental Wind and Turbulence
        if self.injected_failures["severe_wind"]:
            # Extreme gust: 15 to 18 m/s with rapid fluctuations
            target_wind = 16.5 + 2.0 * math.sin(self.sim_time * 1.5)
            self.wind_speed_ms += (target_wind - self.wind_speed_ms) * 0.1
        else:
            # Nominal breeze: 3.5 to 5.0 m/s
            nominal_wind = 4.0 + 0.8 * math.sin(self.sim_time * 0.2)
            self.wind_speed_ms += (nominal_wind - self.wind_speed_ms) * 0.05
            
        # 2. Simulate GPS Sensors
        if self.injected_failures["gps_loss"]:
            self.gps_satellites = 3
            self.gps_hdop = 4.8
            self.gps_valid = False
        elif self.injected_failures["gps_spoof"]:
            self.gps_satellites = 10
            self.gps_hdop = 3.2
            self.gps_valid = False # Spoofed signal detected via IMU mismatch
        else:
            self.gps_satellites = 12 + random.randint(0, 2)
            self.gps_hdop = 0.85 + random.uniform(-0.05, 0.05)
            self.gps_valid = True

        # 3. Simulate Communications Link
        if self.injected_failures["comms_dropout"]:
            self.comms_rssi_dbm = -94.0 + random.uniform(-3.0, 1.0)
            self.comms_connected = False
        else:
            self.comms_rssi_dbm = -63.0 + random.uniform(-2.0, 2.0)
            self.comms_connected = True

        # 4. Action Execution Dynamics
        action = self.active_recovery_action or "CONTINUE_MISSION"
        
        if action == "SWITCH_INERTIAL_DEAD_RECKONING":
            # Drone stabilizes heading, lowers speed to safe dead-reckoning envelope
            self.vy = max(3.0, self.vy - 1.5 * dt)
            self.vx = 0.0
            self.roll = 1.0 * math.sin(self.sim_time)
            self.pitch = 1.0
            # Drift is managed via IMU
            self.flight_mode = "INERTIAL_DEAD_RECKONING"

        elif action == "ALTITUDE_HOLD_DESCENT":
            # Descend to low altitude (e.g. 25m) to escape strong winds aloft
            target_alt = 25.0
            if self.z > target_alt:
                self.vz = -1.8
                self.z += self.vz * dt
            else:
                self.vz = 0.0
                self.z = target_alt
            self.vy = 5.0
            self.flight_mode = "ALT_HOLD_LOW_WIND"

        elif action == "RETURN_TO_HOME":
            # Head back towards (0,0)
            angle_to_home = math.atan2(-self.y, -self.x) * 180.0 / math.pi
            self.yaw = (self.yaw + (angle_to_home - self.yaw) * 0.1) % 360.0
            speed = 7.0
            self.vx = speed * math.cos(math.radians(self.yaw))
            self.vy = speed * math.sin(math.radians(self.yaw))
            self.flight_mode = "RTH"
            dist_home = math.hypot(self.x, self.y)
            if dist_home < 10.0:
                self.active_recovery_action = "CONTROLLED_EMERGENCY_LAND"

        elif action == "CONTROLLED_EMERGENCY_LAND":
            # Controlled safe descent
            self.vx *= 0.8
            self.vy *= 0.8
            self.vz = -1.2
            self.z = max(0.0, self.z + self.vz * dt)
            self.flight_mode = "EMERGENCY_LANDING"
            if self.z <= 0.2:
                self.z = 0.0
                self.vx = 0.0
                self.vy = 0.0
                self.vz = 0.0
                self.mission_status = "SAFE_RECOVERY_LANDED"

        elif action == "POWER_CONSERVATIVE_LOITER":
            # Hover in place at minimum power
            self.vx *= 0.85
            self.vy *= 0.85
            self.vz = 0.0
            self.flight_mode = "LOITER_CONSERVE"

        else: # CONTINUE_MISSION
            # Normal waypoint navigation
            if self.current_wp_idx < len(self.waypoints):
                tx, ty, tz = self.waypoints[self.current_wp_idx]
                dx = tx - self.x
                dy = ty - self.y
                dz = tz - self.z
                dist = math.hypot(dx, dy)
                if dist < 8.0:
                    self.current_wp_idx += 1
                else:
                    target_yaw = math.atan2(dy, dx) * 180.0 / math.pi
                    self.yaw += (target_yaw - self.yaw) * 0.08
                    speed = 8.5
                    self.vx = speed * math.cos(math.radians(self.yaw))
                    self.vy = speed * math.sin(math.radians(self.yaw))
                    self.vz = max(-1.5, min(1.5, dz * 0.3))
                    self.z += self.vz * dt
            else:
                self.mission_status = "COMPLETED"
                self.flight_mode = "MISSION_COMPLETE"

        # Update Position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # 5. Simulate Battery Drain
        effective_drain = self.battery_drain_rate
        if self.injected_failures["battery_sag"]:
            effective_drain *= 4.5 # Fast anomalous discharge
        if self.wind_speed_ms > 12.0:
            effective_drain *= 1.4 # Extra power against headwinds
        if abs(self.vz) > 1.0 and self.vz > 0:
            effective_drain *= 1.2 # Climbing costs more
            
        self.battery_pct = max(0.0, self.battery_pct - (effective_drain * dt))
        if self.battery_pct <= 0.5 and self.z > 0:
            self.mission_status = "CRASHED"
            self.z = 0.0

        # Create Telemetry
        return TelemetryData(
            timestamp=time.time(),
            x=round(self.x, 2),
            y=round(self.y, 2),
            z=round(self.z, 2),
            vx=round(self.vx, 2),
            vy=round(self.vy, 2),
            vz=round(self.vz, 2),
            roll=round(self.roll + random.uniform(-0.5, 0.5), 2),
            pitch=round(self.pitch + random.uniform(-0.5, 0.5), 2),
            yaw=round(self.yaw % 360.0, 2),
            battery_pct=round(self.battery_pct, 2),
            gps_satellites=self.gps_satellites,
            gps_hdop=round(self.gps_hdop, 2),
            gps_valid=self.gps_valid,
            comms_rssi_dbm=round(self.comms_rssi_dbm, 1),
            comms_connected=self.comms_connected,
            wind_speed_ms=round(self.wind_speed_ms, 2),
            wind_heading_deg=round(self.wind_heading_deg, 1),
            flight_mode=self.flight_mode
        )
