import os
import time
import math
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from aerocortex.config import settings
from aerocortex.models import EpisodeRecord

class DeterministicOfflineEmbedding(EmbeddingFunction):
    """
    Lightweight, deterministic offline embedding function for edge devices (e.g. Raspberry Pi 5).
    Produces normalized 64-dimensional vector embeddings without external network dependencies.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "deterministic_offline_embedding"

    def get_config(self) -> dict:
        return {}

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        dim = 64
        for text in input:
            tokens = text.lower().replace(",", " ").replace(":", " ").replace("-", " ").split()
            vec = [0.0] * dim
            for i, token in enumerate(tokens):
                # Hash token to dimensions
                h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
                idx1 = h % dim
                idx2 = (h >> 6) % dim
                weight = 1.0 / (math.log(i + 2))
                vec[idx1] += weight
                vec[idx2] += 0.5 * weight
            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

class EpisodicMemory:
    """
    Episodic Memory Layer:
    Persists past mission experiences and retrieves semantically similar scenarios
    using ChromaDB vector similarity search.
    """
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.EPISODIC_DB_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = DeterministicOfflineEmbedding()
        self.collection = self.client.get_or_create_collection(
            name="uav_mission_episodes",
            embedding_function=self.embedding_fn,
            metadata={"description": "Historical UAV anomaly recovery episodes"}
        )
        self._ensure_seeded()

    def _ensure_seeded(self):
        """Seeds initial historical mission episodes if database is empty."""
        if self.collection.count() == 0:
            seed_episodes = [
                EpisodeRecord(
                    episode_id="EP-HIST-01",
                    mission_id="MISSION-LEGACY-04",
                    timestamp=time.time() - 86400 * 5,
                    anomaly_type="GPS_LOSS",
                    context_text="GPS loss in urban corridor with crosswind 8 m/s, altitude 45m",
                    action_taken="SWITCH_INERTIAL_DEAD_RECKONING",
                    outcome="SUCCESS",
                    time_to_stabilize_s=4.2,
                    battery_remaining_pct=64.0,
                    metadata={"wind_speed": 8.0, "altitude": 45.0, "risk": "medium"}
                ),
                EpisodeRecord(
                    episode_id="EP-HIST-02",
                    mission_id="MISSION-LEGACY-09",
                    timestamp=time.time() - 86400 * 4,
                    anomaly_type="GPS_LOSS",
                    context_text="Severe GPS loss and high wind 14 m/s at 50m altitude",
                    action_taken="ALTITUDE_HOLD_DESCENT",
                    outcome="SUCCESS",
                    time_to_stabilize_s=6.1,
                    battery_remaining_pct=48.0,
                    metadata={"wind_speed": 14.0, "altitude": 50.0, "risk": "high"}
                ),
                EpisodeRecord(
                    episode_id="EP-HIST-03",
                    mission_id="MISSION-LEGACY-12",
                    timestamp=time.time() - 86400 * 3,
                    anomaly_type="BATTERY_SAG",
                    context_text="Rapid battery voltage drop at 18% remaining, 400m from base",
                    action_taken="CONTROLLED_EMERGENCY_LAND",
                    outcome="SUCCESS",
                    time_to_stabilize_s=5.0,
                    battery_remaining_pct=14.0,
                    metadata={"wind_speed": 3.0, "altitude": 30.0, "risk": "critical"}
                ),
                EpisodeRecord(
                    episode_id="EP-HIST-04",
                    mission_id="MISSION-LEGACY-15",
                    timestamp=time.time() - 86400 * 2,
                    anomaly_type="COMMS_DROPOUT",
                    context_text="Telemetry comms signal loss for 12 seconds, GPS normal",
                    action_taken="RETURN_TO_HOME",
                    outcome="SUCCESS",
                    time_to_stabilize_s=3.0,
                    battery_remaining_pct=72.0,
                    metadata={"wind_speed": 4.5, "altitude": 40.0, "risk": "low"}
                ),
                EpisodeRecord(
                    episode_id="EP-HIST-05",
                    mission_id="MISSION-LEGACY-18",
                    timestamp=time.time() - 86400 * 1,
                    anomaly_type="SEVERE_WIND_GUST",
                    context_text="Sudden wind gust 16 m/s at high altitude 60m with high motor power draw",
                    action_taken="ALTITUDE_HOLD_DESCENT",
                    outcome="SUCCESS",
                    time_to_stabilize_s=5.5,
                    battery_remaining_pct=55.0,
                    metadata={"wind_speed": 16.0, "altitude": 60.0, "risk": "high"}
                )
            ]
            for ep in seed_episodes:
                self.add_episode(ep)

    def add_episode(self, episode: EpisodeRecord):
        """Stores a completed mission episode into ChromaDB."""
        doc = (
            f"Anomaly: {episode.anomaly_type}. Context: {episode.context_text}. "
            f"Action: {episode.action_taken}. Outcome: {episode.outcome}."
        )
        meta = {
            "mission_id": episode.mission_id,
            "anomaly_type": episode.anomaly_type,
            "action_taken": episode.action_taken,
            "outcome": episode.outcome,
            "time_to_stabilize_s": float(episode.time_to_stabilize_s),
            "battery_remaining_pct": float(episode.battery_remaining_pct),
            "timestamp": float(episode.timestamp)
        }
        self.collection.upsert(
            ids=[episode.episode_id],
            documents=[doc],
            metadatas=[meta]
        )

    def retrieve_similar(self, query_context: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k most similar past episodes."""
        if self.collection.count() == 0:
            return []
            
        count = min(n_results, self.collection.count())
        results = self.collection.query(
            query_texts=[query_context],
            n_results=count
        )
        
        episodes = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                ep_id = results["ids"][0][i]
                doc = results["documents"][0][i] if "documents" in results else ""
                meta = results["metadatas"][0][i] if "metadatas" in results else {}
                dist = results["distances"][0][i] if "distances" in results and results["distances"] else 0.5
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                episodes.append({
                    "episode_id": ep_id,
                    "document": doc,
                    "action_taken": meta.get("action_taken", "UNKNOWN"),
                    "outcome": meta.get("outcome", "UNKNOWN"),
                    "similarity": round(similarity, 3),
                    "metadata": meta
                })
        return episodes

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Returns all stored episodes for dashboard inspection."""
        data = self.collection.get()
        episodes = []
        if data and "ids" in data:
            for i in range(len(data["ids"])):
                episodes.append({
                    "id": data["ids"][i],
                    "document": data["documents"][i] if "documents" in data else "",
                    "metadata": data["metadatas"][i] if "metadatas" in data else {}
                })
        return episodes
