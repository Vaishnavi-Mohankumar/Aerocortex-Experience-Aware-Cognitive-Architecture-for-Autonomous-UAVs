from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

class Settings(BaseSettings):
    PROJECT_NAME: str = "AeroCortex"
    VERSION: str = "1.0.0"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Storage Paths
    EPISODIC_DB_DIR: str = str(DATA_DIR / "episodic_chroma")
    SEMANTIC_RULES_FILE: str = str(DATA_DIR / "semantic_rules.json")
    KNOWLEDGE_GRAPH_FILE: str = str(DATA_DIR / "knowledge_graph.json")
    MISSION_LOGS_DIR: str = str(DATA_DIR / "mission_logs")
    
    # Safety Envelope Constraints (Raspberry Pi 5 / UAV Standards)
    CRITICAL_BATTERY_LEVEL: float = 10.0      # % -> Immediate emergency auto-land
    MIN_SAFE_BATTERY_LEVEL: float = 20.0      # % -> Trigger RTH / conservative mode
    MAX_PERMISSIBLE_WIND_SPEED: float = 14.0  # m/s -> Ground or loiter low altitude
    MIN_GPS_SATELLITES: int = 6               # < 6 satellites -> GPS Denied
    MAX_GPS_HDOP: float = 2.5                 # > 2.5 -> Poor precision / spoofing
    MIN_COMMS_SIGNAL_DBM: float = -85.0       # < -85 dBm -> Comms blackout
    MAX_DESCENT_RATE: float = 3.0             # m/s -> Maximum safe vertical descent
    MAX_TILT_ANGLE_DEG: float = 35.0          # deg -> Maximum safe roll/pitch
    MAX_OPERATIONAL_ALTITUDE: float = 120.0   # m AGL
    MIN_OPERATIONAL_ALTITUDE: float = 5.0     # m AGL
    CONFIDENCE_APPROVAL_THRESHOLD: float = 0.70 # < 0.70 -> Safety Agent fallback
    
    # LLM & Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:latest"
    LLM_TIMEOUT_SECONDS: float = 4.0
    USE_FALLBACK_REASONER_IF_OLLAMA_UNAVAILABLE: bool = True
    
    # Simulation Settings
    SIMULATION_DT: float = 0.1  # 10 Hz telemetry loop

settings = Settings()
