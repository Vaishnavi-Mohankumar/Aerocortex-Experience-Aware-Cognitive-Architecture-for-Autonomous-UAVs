import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import networkx as nx
from config import config, PROJECT_ROOT

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

class KnowledgeGraph:
    """
    Dual-engine Knowledge Graph:
    1. Primary: Neo4j client using official bolt protocol when available.
    2. Resilient Edge: Embedded NetworkX DiGraph backed by persistent JSON storage.
    
    Nodes: Mission, Failure, Condition, Sensor, RecoveryAction, Outcome, Environment
    Relationships: CAUSES, OCCURRED_DURING, TRIGGERED, RECOVERED_BY, RESULTED_IN, SIMILAR_TO
    """
    def __init__(self, file_path: Optional[str] = None):
        if file_path:
            self.file_path = Path(file_path)
        else:
            self.file_path = PROJECT_ROOT / config.memory.knowledge_graph_path
            
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.nx_graph = nx.DiGraph()
        self.neo4j_driver = None
        self.use_neo4j = False

        # Check Neo4j connectivity
        if NEO4J_AVAILABLE and config.neo4j.enabled:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    config.neo4j.uri,
                    auth=(config.neo4j.user, config.neo4j.password),
                    connection_timeout=0.5
                )
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                self.use_neo4j = True
            except Exception:
                self.use_neo4j = False

        # Load or initialize embedded graph
        self._load_or_seed_graph()

    def _load_or_seed_graph(self) -> None:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.nx_graph = nx.node_link_graph(data, edges="links")
                    return
            except Exception:
                pass
        self._seed_default_graph()
        self.save()

    def _seed_default_graph(self) -> None:
        self.nx_graph.clear()
        
        # 1. Failures
        failures = [
            ("failure:GPS_INTERFERENCE", {"type": "Failure", "name": "GPS_INTERFERENCE"}),
            ("failure:GPS_LOSS", {"type": "Failure", "name": "GPS_LOSS"}),
            ("failure:BATTERY_DEGRADATION", {"type": "Failure", "name": "BATTERY_DEGRADATION"}),
            ("failure:LOW_BATTERY", {"type": "Failure", "name": "LOW_BATTERY"}),
            ("failure:STRONG_WIND", {"type": "Failure", "name": "STRONG_WIND"}),
            ("failure:COMMUNICATION_LOSS", {"type": "Failure", "name": "COMMUNICATION_LOSS"}),
            ("failure:SENSOR_ANOMALY", {"type": "Failure", "name": "SENSOR_ANOMALY"}),
            ("failure:COMBINED_FAILURE", {"type": "Failure", "name": "COMBINED_FAILURE"}),
        ]
        
        # 2. Conditions / Environments
        conditions = [
            ("condition:MODERATE_WIND", {"type": "Condition", "name": "MODERATE_WIND"}),
            ("condition:HIGH_WIND", {"type": "Condition", "name": "HIGH_WIND"}),
            ("condition:CRITICAL_BATTERY", {"type": "Condition", "name": "CRITICAL_BATTERY"}),
            ("condition:RESERVE_BATTERY", {"type": "Condition", "name": "RESERVE_BATTERY"}),
            ("condition:HIGH_ALTITUDE", {"type": "Condition", "name": "HIGH_ALTITUDE"}),
        ]
        
        # 3. Recovery Actions
        actions = [
            ("action:SWITCH_TO_VIO_DEAD_RECKONING", {"type": "RecoveryAction", "name": "SWITCH_TO_VIO_DEAD_RECKONING"}),
            ("action:INERTIAL_DEAD_RECKONING_SAFE_RTH", {"type": "RecoveryAction", "name": "INERTIAL_DEAD_RECKONING_SAFE_RTH"}),
            ("action:POWER_CONSERVATIVE_RTH", {"type": "RecoveryAction", "name": "POWER_CONSERVATIVE_RTH"}),
            ("action:CONTROLLED_EMERGENCY_LAND", {"type": "RecoveryAction", "name": "CONTROLLED_EMERGENCY_LAND"}),
            ("action:REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT", {"type": "RecoveryAction", "name": "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"}),
            ("action:AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH", {"type": "RecoveryAction", "name": "AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH"}),
            ("action:SWITCH_SECONDARY_IMU_STABILIZE", {"type": "RecoveryAction", "name": "SWITCH_SECONDARY_IMU_STABILIZE"}),
        ]
        
        # 4. Outcomes
        outcomes = [
            ("outcome:MISSION_SUCCESS", {"type": "Outcome", "name": "MISSION_SUCCESS", "weight": 1.0}),
            ("outcome:MISSION_ABORTED_SAFE", {"type": "Outcome", "name": "MISSION_ABORTED_SAFE", "weight": 0.8}),
            ("outcome:MISSION_FAILURE", {"type": "Outcome", "name": "MISSION_FAILURE", "weight": 0.0}),
        ]
        
        for n, attrs in failures + conditions + actions + outcomes:
            self.nx_graph.add_node(n, **attrs)

        # Connect paths: Failure -> Condition -> RecoveryAction -> Outcome
        edges = [
            # GPS Interference
            ("failure:GPS_INTERFERENCE", "condition:MODERATE_WIND", "OCCURRED_DURING", 0.8),
            ("condition:MODERATE_WIND", "action:SWITCH_TO_VIO_DEAD_RECKONING", "TRIGGERED", 0.9),
            ("action:SWITCH_TO_VIO_DEAD_RECKONING", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.95),

            ("failure:GPS_INTERFERENCE", "condition:HIGH_WIND", "OCCURRED_DURING", 0.7),
            ("condition:HIGH_WIND", "action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "TRIGGERED", 0.85),
            ("action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.9),

            # GPS Loss
            ("failure:GPS_LOSS", "action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "RECOVERED_BY", 0.92),
            ("action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.95),

            # Battery Degradation
            ("failure:BATTERY_DEGRADATION", "condition:RESERVE_BATTERY", "OCCURRED_DURING", 0.9),
            ("condition:RESERVE_BATTERY", "action:POWER_CONSERVATIVE_RTH", "TRIGGERED", 0.92),
            ("action:POWER_CONSERVATIVE_RTH", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.93),

            # Low Battery
            ("failure:LOW_BATTERY", "condition:CRITICAL_BATTERY", "OCCURRED_DURING", 0.95),
            ("condition:CRITICAL_BATTERY", "action:CONTROLLED_EMERGENCY_LAND", "TRIGGERED", 0.98),
            ("action:CONTROLLED_EMERGENCY_LAND", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.96),

            # Strong Wind
            ("failure:STRONG_WIND", "condition:HIGH_WIND", "OCCURRED_DURING", 0.95),
            ("condition:HIGH_WIND", "action:REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT", "TRIGGERED", 0.9),
            ("action:REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.92),

            # Comms Loss
            ("failure:COMMUNICATION_LOSS", "action:AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH", "RECOVERED_BY", 0.91),
            ("action:AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.90),

            # Combined Failure
            ("failure:COMBINED_FAILURE", "action:CONTROLLED_EMERGENCY_LAND", "RECOVERED_BY", 0.95),
            ("action:CONTROLLED_EMERGENCY_LAND", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.95),
        ]

        for u, v, rel, w in edges:
            self.nx_graph.add_edge(u, v, relation=rel, weight=w)

    def save(self) -> None:
        try:
            data = nx.node_link_data(self.nx_graph, edges="links")
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def query_action_relevance(self, failure_type: str, condition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Traverses paths from the failure node to find candidate recovery actions
        and their associated outcome weights.
        Returns a list of candidate actions ranked by graph relevance score.
        """
        target_f = f"failure:{failure_type}"
        if not self.nx_graph.has_node(target_f):
            return []

        candidates = {}
        
        # 1. Direct failure -> action edges (RECOVERED_BY)
        for succ in self.nx_graph.successors(target_f):
            if succ.startswith("action:"):
                action_name = self.nx_graph.nodes[succ].get("name", succ.replace("action:", ""))
                weight = self.nx_graph[target_f][succ].get("weight", 0.7)
                outcome_bonus = self._get_action_outcome_weight(succ)
                score = round(weight * outcome_bonus, 4)
                candidates[action_name] = max(candidates.get(action_name, 0.0), score)

            # 2. Multi-hop failure -> condition -> action
            elif succ.startswith("condition:"):
                cond_weight = self.nx_graph[target_f][succ].get("weight", 0.6)
                for cond_succ in self.nx_graph.successors(succ):
                    if cond_succ.startswith("action:"):
                        action_name = self.nx_graph.nodes[cond_succ].get("name", cond_succ.replace("action:", ""))
                        act_weight = self.nx_graph[succ][cond_succ].get("weight", 0.8)
                        outcome_bonus = self._get_action_outcome_weight(cond_succ)
                        score = round(cond_weight * act_weight * outcome_bonus, 4)
                        candidates[action_name] = max(candidates.get(action_name, 0.0), score)

        # Convert to sorted list
        results = [
            {"action": act, "graph_relevance": score}
            for act, score in candidates.items()
        ]
        results.sort(key=lambda x: x["graph_relevance"], reverse=True)
        return results

    def _get_action_outcome_weight(self, action_node: str) -> float:
        for succ in self.nx_graph.successors(action_node):
            if succ.startswith("outcome:"):
                return self.nx_graph.nodes[succ].get("weight", 0.85)
        return 0.75

    def add_mission_resolution(
        self,
        mission_id: str,
        failure: str,
        condition: str,
        action: str,
        outcome: str,
        success: bool
    ) -> None:
        """
        Adds mission experience relationships to the knowledge graph.
        Mission -> OCCURRED_DURING -> Failure
        Failure -> TRIGGERED -> Action
        Action -> RESULTED_IN -> Outcome
        """
        m_node = f"mission:{mission_id}"
        f_node = f"failure:{failure}"
        c_node = f"condition:{condition}"
        a_node = f"action:{action}"
        o_node = f"outcome:{outcome}"

        self.nx_graph.add_node(m_node, type="Mission", name=mission_id)
        self.nx_graph.add_node(f_node, type="Failure", name=failure)
        self.nx_graph.add_node(c_node, type="Condition", name=condition)
        self.nx_graph.add_node(a_node, type="RecoveryAction", name=action)
        self.nx_graph.add_node(o_node, type="Outcome", name=outcome, weight=1.0 if success else 0.2)

        self.nx_graph.add_edge(m_node, f_node, relation="ENCOUNTERED", weight=1.0)
        self.nx_graph.add_edge(f_node, c_node, relation="OCCURRED_DURING", weight=0.9)
        self.nx_graph.add_edge(c_node, a_node, relation="TRIGGERED", weight=0.95 if success else 0.4)
        self.nx_graph.add_edge(a_node, o_node, relation="RESULTED_IN", weight=0.95 if success else 0.3)
        self.save()

        # Mirror to Neo4j if active
        if self.use_neo4j and self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run(
                        """
                        MERGE (m:Mission {name: $mission_id})
                        MERGE (f:Failure {name: $failure})
                        MERGE (c:Condition {name: $condition})
                        MERGE (a:RecoveryAction {name: $action})
                        MERGE (o:Outcome {name: $outcome})
                        MERGE (m)-[:ENCOUNTERED]->(f)
                        MERGE (f)-[:OCCURRED_DURING]->(c)
                        MERGE (c)-[:TRIGGERED {weight: $weight}]->(a)
                        MERGE (a)-[:RESULTED_IN]->(o)
                        """,
                        mission_id=mission_id,
                        failure=failure,
                        condition=condition,
                        action=action,
                        outcome=outcome,
                        weight=0.95 if success else 0.3
                    )
            except Exception:
                pass

    def get_summary(self) -> Dict[str, Any]:
        return {
            "engine": "Neo4j" if self.use_neo4j else "NetworkX Embedded",
            "nodes_count": self.nx_graph.number_of_nodes(),
            "edges_count": self.nx_graph.number_of_edges(),
            "neo4j_connected": self.use_neo4j
        }
