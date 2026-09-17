# MATLAB/Simulink Digital Twin Integration Guide

AeroCortex provides an open, decoupled HTTP REST gateway enabling seamless co-simulation with MATLAB/Simulink UAV digital twins.

---

## 1. Architecture Overview

```
+------------------------------------+
|  MATLAB / Simulink Digital Twin   |
|   (Aerodynamics / 6-DOF Motors)    |
+------------------------------------+
                 |
                 | HTTP POST /telemetry (JSON)
                 v
+------------------------------------+
|       AeroCortex Local API         |
| (Situation -> Memory -> Safety)    |
+------------------------------------+
                 |
                 | JSON Response: { action, is_approved, plan }
                 v
+------------------------------------+
|      Simulink Flight Autopilot     |
|   (Applies Trajectory / Mode)      |
+------------------------------------+
```

---

## 2. Telemetry JSON Specification

Your MATLAB client or Simulink MATLAB Function Block must serialize a JSON struct matching this schema to `POST http://127.0.0.1:8000/telemetry`:

```json
{
  "timestamp": 1726500000.0,
  "mission_id": "MATLAB_MISSION_001",
  "latitude": 11.0168,
  "longitude": 76.9558,
  "altitude": 120.0,
  "velocity": 12.5,
  "battery_level": 92.0,
  "battery_voltage": 16.2,
  "gps_status": "healthy",
  "gps_accuracy": 1.4,
  "imu_acceleration": [0.0, 0.0, 9.81],
  "imu_gyroscope": [0.0, 0.0, 0.0],
  "wind_speed": 6.8,
  "wind_direction": 180.0,
  "communication_status": "connected",
  "mission_state": "CRUISE",
  "waypoint": 1,
  "heading": 90.0,
  "payload_status": "nominal"
}
```

---

## 3. Response Format

The API returns an actuation response:

```json
{
  "status": "SUCCESS",
  "action": "SWITCH_TO_VIO_DEAD_RECKONING",
  "is_approved": true,
  "execution_status": "RECOVERED",
  "plan": {
    "action": "SWITCH_TO_VIO_DEAD_RECKONING",
    "reason": "GPS degraded; switching to VIO and inertial dead-reckoning",
    "steps": [
      "Disengage GPS-based position hold",
      "Engage VIO / Optical Flow positioning loop",
      "Maintain current altitude and evaluate drift rate"
    ],
    "confidence": 0.92,
    "risk_level": "LOW"
  },
  "safety": {
    "approved": true,
    "reason": "All deterministic safety criteria satisfied.",
    "fallback_action": "RETURN_TO_HOME"
  }
}
```

---

## 4. Running the MATLAB Client

1. Start AeroCortex API:
   ```bash
   python main.py --api
   ```
2. Open MATLAB and run:
   ```matlab
   run('matlab/aerocortex_telemetry_sender.m')
   ```
