from typing import List, Optional, Deque
from collections import deque
import time
from aerocortex.models import TelemetryData, AnomalyReport

class WorkingMemory:
    """
    Working Memory: Real-time mission state.
    Maintains the current flight state and a sliding window of recent telemetry frames
    for immediate rate-of-change and trend detection.
    """
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: Deque[TelemetryData] = deque(maxlen=window_size)
        self.current_state: Optional[TelemetryData] = None
        self.active_anomaly: AnomalyReport = AnomalyReport()
        self.active_command: str = "CONTINUE_MISSION"
        self.mission_id: str = "MISSION-001"
        self.start_time: float = time.time()

    def update(self, telemetry: TelemetryData):
        self.current_state = telemetry
        self.history.append(telemetry)

    def get_latest(self) -> Optional[TelemetryData]:
        return self.current_state

    def get_history(self) -> List[TelemetryData]:
        return list(self.history)

    def calculate_battery_drain_rate(self) -> float:
        """Calculates battery % loss per second over the sliding window."""
        if len(self.history) < 2:
            return 0.0
        first = self.history[0]
        last = self.history[-1]
        dt = max(0.001, last.timestamp - first.timestamp)
        drop = max(0.0, first.battery_pct - last.battery_pct)
        return drop / dt

    def calculate_altitude_rate(self) -> float:
        """Calculates vertical velocity trend m/s."""
        if len(self.history) < 2:
            return 0.0
        first = self.history[0]
        last = self.history[-1]
        dt = max(0.001, last.timestamp - first.timestamp)
        return (last.z - first.z) / dt

    def clear(self):
        self.history.clear()
        self.current_state = None
        self.active_anomaly = AnomalyReport()
        self.active_command = "CONTINUE_MISSION"
        self.start_time = time.time()
