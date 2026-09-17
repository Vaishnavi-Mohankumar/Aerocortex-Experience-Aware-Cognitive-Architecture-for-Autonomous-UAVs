import os
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
CONFIG_PATH = CONFIG_DIR / "config.yaml"

class FlightEnvelopeConfig(BaseModel):
    min_battery_percent: float = 15.0
    critical_battery_percent: float = 10.0
    max_altitude_m: float = 150.0
    min_altitude_m: float = 5.0
    max_velocity_mps: float = 20.0
    max_wind_speed_mps: float = 12.0
    max_gps_hdop: float = 3.0
    min_comms_rssi_dbm: float = -85.0
    geofence: Dict[str, float] = Field(default_factory=lambda: {
        "min_lat": 10.9000, "max_lat": 11.1500,
        "min_lon": 76.8500, "max_lon": 77.1000
    })

class HybridWeightsConfig(BaseModel):
    vector_similarity: float = 0.6
    graph_relevance: float = 0.4

class MemoryConfig(BaseModel):
    hybrid_weights: HybridWeightsConfig = Field(default_factory=HybridWeightsConfig)
    top_k_episodes: int = 3
    chroma_db_dir: str = "data/knowledge/chroma"
    chroma_collection: str = "aerocortex_episodes"
    semantic_rules_path: str = "data/knowledge/semantic_rules.json"
    knowledge_graph_path: str = "data/knowledge/knowledge_graph.json"

class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "aerocortex_password"
    enabled: bool = False

class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "gemma3:latest"
    timeout_seconds: float = 5.0
    connect_timeout_seconds: float = 0.3
    min_confidence_threshold: float = 0.70

class SystemConfig(BaseModel):
    app_name: str = "AeroCortex Cognitive Edge Architecture"
    version: str = "1.0.0"
    log_level: str = "INFO"
    offline_mode: bool = True
    device_target: str = "edge_uav"

class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    flight_envelope: FlightEnvelopeConfig = Field(default_factory=FlightEnvelopeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501

def load_config(config_file: Optional[Path] = None) -> AppConfig:
    target_file = config_file or CONFIG_PATH
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Flatten api/dashboard ports if nested
            api_data = data.get("api", {})
            dash_data = data.get("dashboard", {})
            return AppConfig(
                system=SystemConfig(**data.get("system", {})),
                flight_envelope=FlightEnvelopeConfig(**data.get("flight_envelope", {})),
                memory=MemoryConfig(**data.get("memory", {})),
                neo4j=Neo4jConfig(**data.get("neo4j", {})),
                llm=LLMConfig(**data.get("llm", {})),
                api_host=api_data.get("host", "0.0.0.0"),
                api_port=api_data.get("port", 8000),
                dashboard_port=dash_data.get("port", 8501),
            )
        except Exception:
            pass
    return AppConfig()

config = load_config()
