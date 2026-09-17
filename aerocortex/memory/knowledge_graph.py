import os
import json
from typing import List, Dict, Any, Optional
import networkx as nx
from aerocortex.config import settings

class KnowledgeGraph:
    """
    Knowledge Graph:
    Connects Anomalies, Environmental Contexts, Recovery Actions, and Mission Outcomes.
    Enables multi-hop relational reasoning beyond pure embedding similarity.
    """
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or settings.KNOWLEDGE_GRAPH_FILE
        self.graph = nx.DiGraph()
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data, edges="links")
                    return
            except Exception:
                pass
        self._seed_default_graph()
        self.save()

    def _seed_default_graph(self):
        self.graph.clear()
        
        # 1. Anomalies
        anomalies = [
            ("anomaly:GPS_LOSS", {"type": "anomaly", "label": "GPS Loss", "color": "#E53935"}),
            ("anomaly:BATTERY_SAG", {"type": "anomaly", "label": "Battery Sag", "color": "#E53935"}),
            ("anomaly:COMMS_DROPOUT", {"type": "anomaly", "label": "Comms Loss", "color": "#E53935"}),
            ("anomaly:SEVERE_WIND_GUST", {"type": "anomaly", "label": "Severe Wind Gust", "color": "#E53935"})
        ]
        
        # 2. Contexts
        contexts = [
            ("context:HIGH_WIND", {"type": "context", "label": "High Wind (>10m/s)", "color": "#FB8C00"}),
            ("context:MODERATE_WIND", {"type": "context", "label": "Moderate Wind", "color": "#FB8C00"}),
            ("context:LOW_BATTERY", {"type": "context", "label": "Critical Battery (<20%)", "color": "#FB8C00"}),
            ("context:MEDIUM_BATTERY", {"type": "context", "label": "Reserve Battery (>20%)", "color": "#FB8C00"}),
            ("context:URBAN_CORRIDOR", {"type": "context", "label": "Urban Canyon", "color": "#FB8C00"})
        ]
        
        # 3. Actions
        actions = [
            ("action:SWITCH_INERTIAL_DEAD_RECKONING", {"type": "action", "label": "Inertial Dead-Reckoning", "color": "#1E88E5"}),
            ("action:ALTITUDE_HOLD_DESCENT", {"type": "action", "label": "Altitude Hold & Descend", "color": "#1E88E5"}),
            ("action:RETURN_TO_HOME", {"type": "action", "label": "Return to Home (RTH)", "color": "#1E88E5"}),
            ("action:CONTROLLED_EMERGENCY_LAND", {"type": "action", "label": "Emergency Landing", "color": "#1E88E5"}),
            ("action:POWER_CONSERVATIVE_LOITER", {"type": "action", "label": "Loiter & Conserve", "color": "#1E88E5"})
        ]
        
        # 4. Outcomes
        outcomes = [
            ("outcome:SUCCESS", {"type": "outcome", "label": "Safe Recovery / Success", "color": "#43A047"}),
            ("outcome:ABORTED", {"type": "outcome", "label": "Mission Aborted Safely", "color": "#FDD835"}),
            ("outcome:FAILURE", {"type": "outcome", "label": "UAV Failure / Crash", "color": "#D32F2F"})
        ]
        
        for n, attrs in anomalies + contexts + actions + outcomes:
            self.graph.add_node(n, **attrs)

        # Edges (relation, weight)
        relationships = [
            # GPS Loss scenarios
            ("anomaly:GPS_LOSS", "context:MODERATE_WIND", {"relation": "OCCURRED_IN", "weight": 0.8}),
            ("context:MODERATE_WIND", "action:SWITCH_INERTIAL_DEAD_RECKONING", {"relation": "MITIGATED_BY", "weight": 0.9}),
            ("action:SWITCH_INERTIAL_DEAD_RECKONING", "outcome:SUCCESS", {"relation": "RESULTED_IN", "weight": 0.95}),

            ("anomaly:GPS_LOSS", "context:HIGH_WIND", {"relation": "OCCURRED_IN", "weight": 0.7}),
            ("context:HIGH_WIND", "action:ALTITUDE_HOLD_DESCENT", {"relation": "MITIGATED_BY", "weight": 0.85}),
            ("action:ALTITUDE_HOLD_DESCENT", "outcome:SUCCESS", {"relation": "RESULTED_IN", "weight": 0.9}),

            # Battery Sag scenarios
            ("anomaly:BATTERY_SAG", "context:LOW_BATTERY", {"relation": "OCCURRED_IN", "weight": 0.9}),
            ("context:LOW_BATTERY", "action:CONTROLLED_EMERGENCY_LAND", {"relation": "MITIGATED_BY", "weight": 0.95}),
            ("action:CONTROLLED_EMERGENCY_LAND", "outcome:SUCCESS", {"relation": "RESULTED_IN", "weight": 0.98}),

            ("anomaly:BATTERY_SAG", "context:MEDIUM_BATTERY", {"relation": "OCCURRED_IN", "weight": 0.6}),
            ("context:MEDIUM_BATTERY", "action:RETURN_TO_HOME", {"relation": "MITIGATED_BY", "weight": 0.8}),
            ("action:RETURN_TO_HOME", "outcome:SUCCESS", {"relation": "RESULTED_IN", "weight": 0.85}),

            # Comms Lost
            ("anomaly:COMMS_DROPOUT", "context:URBAN_CORRIDOR", {"relation": "OCCURRED_IN", "weight": 0.75}),
            ("context:URBAN_CORRIDOR", "action:RETURN_TO_HOME", {"relation": "MITIGATED_BY", "weight": 0.85}),

            # Severe Wind
            ("anomaly:SEVERE_WIND_GUST", "context:HIGH_WIND", {"relation": "OCCURRED_IN", "weight": 0.95}),
            ("context:HIGH_WIND", "action:POWER_CONSERVATIVE_LOITER", {"relation": "MITIGATED_BY", "weight": 0.7}),
            ("action:POWER_CONSERVATIVE_LOITER", "outcome:ABORTED", {"relation": "RESULTED_IN", "weight": 0.8})
        ]
        
        for u, v, attrs in relationships:
            self.graph.add_edge(u, v, **attrs)

    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        data = nx.node_link_data(self.graph, edges="links")
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def query_recommendation(self, anomaly_type: str, context_keys: List[str]) -> List[Dict[str, Any]]:
        """
        Traverses paths from Anomaly -> Context -> Action -> Outcome
        Returns candidate actions ranked by path weights and success probability.
        """
        anomaly_node = f"anomaly:{anomaly_type}"
        if anomaly_node not in self.graph:
            return []

        candidates = []
        for _, context_node, edge_data1 in self.graph.out_edges(anomaly_node, data=True):
            w1 = edge_data1.get("weight", 0.5)
            # Find actions branching from this context
            for _, action_node, edge_data2 in self.graph.out_edges(context_node, data=True):
                if not action_node.startswith("action:"):
                    continue
                w2 = edge_data2.get("weight", 0.5)
                # Find outcomes
                for _, outcome_node, edge_data3 in self.graph.out_edges(action_node, data=True):
                    w3 = edge_data3.get("weight", 0.5)
                    action_name = action_node.replace("action:", "")
                    outcome_name = outcome_node.replace("outcome:", "")
                    score = w1 * w2 * (w3 if outcome_name == "SUCCESS" else w3 * 0.4)
                    
                    candidates.append({
                        "anomaly": anomaly_type,
                        "context": self.graph.nodes[context_node].get("label", context_node),
                        "action": action_name,
                        "outcome": outcome_name,
                        "score": round(score, 3),
                        "path": [anomaly_node, context_node, action_node, outcome_node]
                    })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def add_or_update_experience(self, anomaly_type: str, context_label: str, action: str, outcome: str):
        """
        Expands or reinforces the Knowledge Graph after a mission.
        """
        a_node = f"anomaly:{anomaly_type}"
        c_node = f"context:{context_label.upper().replace(' ', '_')}"
        act_node = f"action:{action}"
        o_node = f"outcome:{outcome}"

        if not self.graph.has_node(a_node):
            self.graph.add_node(a_node, type="anomaly", label=anomaly_type, color="#E53935")
        if not self.graph.has_node(c_node):
            self.graph.add_node(c_node, type="context", label=context_label, color="#FB8C00")
        if not self.graph.has_node(act_node):
            self.graph.add_node(act_node, type="action", label=action, color="#1E88E5")
        if not self.graph.has_node(o_node):
            self.graph.add_node(o_node, type="outcome", label=outcome, color="#43A047")

        # Reinforce edges
        for u, v in [(a_node, c_node), (c_node, act_node), (act_node, o_node)]:
            if self.graph.has_edge(u, v):
                current_w = self.graph[u][v].get("weight", 0.5)
                new_w = min(1.0, current_w + (0.05 if outcome == "SUCCESS" else -0.05))
                self.graph[u][v]["weight"] = round(new_w, 3)
            else:
                self.graph.add_edge(u, v, relation="LEARNED_LINK", weight=0.6)

        self.save()

    def get_elements_for_visualization(self) -> Dict[str, Any]:
        """Formats graph data for interactive Streamlit / Plotly / Pyvis visualizers."""
        nodes = []
        for n, d in self.graph.nodes(data=True):
            nodes.append({
                "id": n,
                "label": d.get("label", n),
                "type": d.get("type", "node"),
                "color": d.get("color", "#9E9E9E")
            })
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": d.get("relation", ""),
                "weight": d.get("weight", 1.0)
            })
        return {"nodes": nodes, "edges": edges}
