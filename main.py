# main.py
import os
import json
import glob
import torch
from flask import Flask, render_template, jsonify, request
from torch_geometric.data import Data
from copy import deepcopy
import numpy as np
import random

from src.model.model import RiskGCN

app = Flask(__name__, template_folder="templates", static_folder="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gnn_model.pt")
META_PATH = os.path.join(MODEL_DIR, "meta.json")
SNAPSHOT_DIR = os.path.join("data", "snapshots")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("Model files not found. Run the data generator and training scripts first.")

with open(META_PATH, "r") as f:
    meta = json.load(f)
IN_CHANNELS = meta["in_channels"]

model = RiskGCN(in_channels=IN_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def list_timesteps():
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.startswith("G") and f.endswith(".json")]
    timesteps = sorted(int(f[1:-5]) for f in files)
    return timesteps

def load_snapshot(timestep: int):
    fname = os.path.join(SNAPSHOT_DIR, f"G{timestep}.json")
    if not os.path.exists(fname):
        return None, None, None
    with open(fname, "r") as fp:
        snap = json.load(fp)
    nodes = snap["nodes"]
    edges = snap["edges"]
    # Build tensors in the node order provided in the file
    node_names = [n["id"] for n in nodes]
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    x_list = []
    y_list = []
    for node in nodes:
        feats = node["features"]
        x_list.append([
            feats["liquidity_ratio"],
            feats["capital_adequacy"],
            feats["leverage"],
            feats["base_risk"],
            feats["stress_t"],
        ])
        y_list.append(node["label_next_high_risk"])
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    edge_index_list = [[], []]
    edge_weight_list = []
    for e in edges:
        u = name_to_idx[e["source"]]
        v = name_to_idx[e["target"]]
        w = e["exposure"]
        edge_index_list[0].extend([u, v])
        edge_index_list[1].extend([v, u])
        edge_weight_list.extend([w, w])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    # pass back original nodes/edges (with names)
    return data, nodes, edges

@app.route("/")
def index():
    timesteps = list_timesteps()
    max_t = max(timesteps) if timesteps else 0
    return render_template("index.html", max_timestep=max_t)

@app.route("/predict", methods=["GET"])
def predict():
    t = int(request.args.get("t", 1))
    data, nodes, edges = load_snapshot(t)
    if data is None:
        return jsonify({"error": "Invalid timestep"}), 400
    data = data.to(DEVICE)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
    results = []
    for node, p in zip(nodes, probs):
        results.append({
            "id": node["id"],
            "features": node["features"],
            "label_next_high_risk": node["label_next_high_risk"],
            "predicted_prob_high_risk": float(p),
            # convenience fields used by frontend:
            "stress": float(node["features"]["stress_t"]),
            "trueLabel": int(node["label_next_high_risk"]),
            "prob": float(p)
        })
    return jsonify({
        "timestep": t,
        "nodes": results,
        "edges": edges
    })

@app.route("/stress_test", methods=["POST"])
def stress_test():
    data_in = request.json
    liquidity_factor = float(data_in.get("liquidity_factor", 1.0))
    exposure_factor = float(data_in.get("exposure_factor", 1.0))
    macro_override = float(data_in.get("macro", 0.0))
    t = int(data_in.get("timestep", 1))
    graph, nodes, edges = load_snapshot(t)
    if graph is None:
        return jsonify({"error": "Invalid timestep"}), 400
    x = graph.x.clone().to(DEVICE)
    edge_index = graph.edge_index.clone().to(DEVICE)
    # apply liquidity shock (feature index 0)
    x[:, 0] = x[:, 0] * liquidity_factor
    # apply macro override to stress feature (index 4)
    x[:, 4] = x[:, 4] + macro_override
    # modify edge weights according to exposure_factor (we must recreate edge weights matching edge_index ordering)
    num_edges = graph.edge_index.shape[1]  # includes both directions
    # original edges list (edges param) uses names; produce new exposures in same paired order
    new_edge_weight = []
    # edges list contains undirected edges once; edge_index contains both directions; we will create weights by doubling exposure
    for e in edges:
        new_w = float(e["exposure"]) * exposure_factor
        new_edge_weight.extend([new_w, new_w])
    new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x.to(DEVICE), edge_index, new_edge_weight)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
    # Build results with same fields as /predict (so frontend drawGraph works)
    results = []
    for node, p in zip(nodes, probs):
        results.append({
            "id": node["id"],
            "features": node["features"],
            "label_next_high_risk": node["label_next_high_risk"],
            "predicted_prob_high_risk": float(p),
            "stress": float(node["features"]["stress_t"] + macro_override),
            "trueLabel": int(node["label_next_high_risk"]),
            "prob": float(p)
        })
    # edges with updated exposures to send back
    modified_edges = []
    for e in edges:
        modified_edges.append({
            "source": e["source"],
            "target": e["target"],
            "exposure": float(e["exposure"]) * exposure_factor
        })
    return jsonify({"nodes": results, "edges": modified_edges})
\



# Add to main.py (below other endpoints)
from copy import deepcopy
import math
import random

def _simulate_contagion_from_snapshot(nodes_list, edges_list, seed_node_name,
                                      scenario_type="liquidity_drain",
                                      magnitude=0.4, steps=5,
                                      macro_override=0.0):
    """
    Run a multi-step contagion starting from one snapshot (nodes_list, edges_list).
    - nodes_list: list of node dicts (id, features, label_next_high_risk)
    - edges_list: list of edges dicts (source, target, exposure)
    - seed_node_name: the institution name to shock
    - scenario_type: one of ["liquidity_drain", "credit_default", "exposure_spike", "payment_freeze", "nbfc_collapse", "mutualfund_run"]
    - magnitude: 0..1 (relative intensity)
    - steps: number of propagation steps to simulate
    - macro_override: optional constant added to stress
    Returns: list_of_snapshots where each snapshot is {"nodes": [...], "edges": [...]}
    """

    # Build adjacency and exposures (name -> neighbors)
    names = [n["id"] for n in nodes_list]
    name_to_idx = {name: i for i, name in enumerate(names)}
    N = len(names)

    # Node state (working copy): features and current stress
    state_feats = []
    for n in nodes_list:
        f = n["features"]
        state_feats.append({
            "liquidity_ratio": float(f["liquidity_ratio"]),
            "capital_adequacy": float(f["capital_adequacy"]),
            "leverage": float(f["leverage"]),
            "base_risk": float(f["base_risk"]),
            "stress": float(f["stress_t"])
        })

    # Build neighbor lists and exposure matrix (symmetric)
    neighbors = {name: [] for name in names}
    exposure_map = {}
    for e in edges_list:
        s = e["source"]; t = e["target"]; w = float(e["exposure"])
        neighbors[s].append((t, w))
        neighbors[t].append((s, w))
        exposure_map[(s,t)] = w
        exposure_map[(t,s)] = w

    # Scenario-specific initial shock (only to seed node)
    def apply_seed_shock(state, seed, scenario, mag):
        """Mutate state (list) - apply direct shock to seed"""
        idx = name_to_idx[seed]
        if scenario == "liquidity_drain":
            # reduce liquidity drastically -> increases vulnerability
            state[idx]["liquidity_ratio"] = max(0.01, state[idx]["liquidity_ratio"] * (1.0 - mag*0.9))
            # bump stress
            state[idx]["stress"] = min(1.0, state[idx]["stress"] + mag * 0.6)
        elif scenario == "credit_default":
            # simulate default by setting stress -> 1.0 and increase exposures effectively
            state[idx]["stress"] = 1.0
        elif scenario == "exposure_spike":
            # temporarily treat the seed as having larger exposures (we'll amplify neighbor sensitivity)
            state[idx]["stress"] = min(1.0, state[idx]["stress"] + 0.25 * mag)
            # store an attribute to indicate exposure spike for neighbor calculation
            state[idx]["exposure_spike"] = mag
        elif scenario == "payment_freeze":
            # payment network node suffers big stress and adjacent nodes suffer delays (increase their stress)
            state[idx]["stress"] = min(1.0, state[idx]["stress"] + 0.5 * mag)
        elif scenario == "nbfc_collapse":
            state[idx]["stress"] = min(1.0, state[idx]["stress"] + 0.6 * mag)
            state[idx]["liquidity_ratio"] = max(0.01, state[idx]["liquidity_ratio"] * (1.0 - 0.8*mag))
        elif scenario == "mutualfund_run":
            state[idx]["stress"] = state[idx]["stress"] + 0.45 * mag
            state[idx]["liquidity_ratio"] = max(0.01, state[idx]["liquidity_ratio"] * (1.0 - 0.6*mag))
        else:
            # generic shock -> stress bump
            state[idx]["stress"] = min(1.0, state[idx]["stress"] + 0.2 * mag)

    # Storage for snapshots (we'll include the original snapshot as step 0)
    timeline = []

    # Step 0: baseline (with optional macro override added)
    snapshot0_nodes = []
    for i, name in enumerate(names):
        snapshot0_nodes.append({
            "id": name,
            "features": {
                "liquidity_ratio": state_feats[i]["liquidity_ratio"],
                "capital_adequacy": state_feats[i]["capital_adequacy"],
                "leverage": state_feats[i]["leverage"],
                "base_risk": state_feats[i]["base_risk"],
                "stress_t": float(min(1.0, state_feats[i]["stress"] + macro_override))
            },
            "label_next_high_risk": int(state_feats[i]["stress"] > 0.7)
        })
    timeline.append({"t": 0, "nodes": deepcopy(snapshot0_nodes), "edges": deepcopy(edges_list)})

    # Apply initial seed shock at step 0 (before propagation)
    apply_seed_shock(state_feats, seed_node_name, scenario_type, magnitude)

    # Now simulate multi-step contagion
    for step in range(1, steps+1):
        new_state = [dict(s) for s in state_feats]  # shallow copy of dicts

        # macro volatility small each step (the scenario may include its own shock)
        macro_shock = np.random.normal(0.0, 0.08)
        if random.random() < 0.03:
            macro_shock += random.choice([0.35, -0.35])  # rare systemic shock

        # For each node compute contagion-influenced stress_raw using similar non-linear mechanics
        for i, name in enumerate(names):
            base_risk = state_feats[i]["base_risk"]
            liquidity = state_feats[i]["liquidity_ratio"]
            leverage = state_feats[i]["leverage"]
            # scaled leverage effect
            lev_scaled = max(0.0, min(1.0, (leverage - 5.0)/25.0))
            sys_imp = len(neighbors[name]) / max(1, (len(names)-1))

            # neighbor linear & non-linear contributions
            lin_sum = 0.0; lin_den = 0.0; nl_sum = 0.0; nl_den = 0.0
            for nb, w in neighbors[name]:
                nb_idx = name_to_idx[nb]
                s_nb = state_feats[nb_idx]["stress"]
                lin_sum += w * s_nb
                lin_den += w
                if s_nb > 0.7:
                    nl_sum += w * (s_nb ** 2)
                    nl_den += w

            lin_effect = (lin_sum / lin_den) if lin_den > 0 else 0.0
            nl_effect = (nl_sum / nl_den) if nl_den > 0 else 0.0
            contagion = 0.55 * lin_effect + 0.45 * nl_effect

            # amplify if neighbors include a node that had exposure_spike
            spike_amp = 0.0
            for nb, w in neighbors[name]:
                if "exposure_spike" in state_feats[name_to_idx[nb]]:
                    spike_amp += state_feats[name_to_idx[nb]]["exposure_spike"] * 0.25 * (w/ (1.0 + w))

            # idiosyncratic jump
            idio = np.random.normal(0.0, 0.06)
            if random.random() < 0.1:
                idio += np.random.uniform(-0.3, 0.5)

            # recovery weaker
            recovery = 0.02 * liquidity

            # compute raw stress
            stress_raw = (
                0.2 * base_risk +
                0.4 * contagion * (1.0 + 0.5 * lev_scaled + 0.2 * sys_imp) +
                0.25 * state_feats[i]["stress"] +
                macro_shock +
                idio +
                spike_amp -
                recovery
            )

            # Clip and set
            new_state[i]["stress"] = float(np.clip(stress_raw, 0.0, 1.0))

        # After computing all new_state, set state_feats = new_state
        state_feats = new_state

        # Build snapshot nodes for this step (include macro_override)
        snapshot_nodes = []
        for i, name in enumerate(names):
            snapshot_nodes.append({
                "id": name,
                "features": {
                    "liquidity_ratio": state_feats[i]["liquidity_ratio"],
                    "capital_adequacy": state_feats[i]["capital_adequacy"],
                    "leverage": state_feats[i]["leverage"],
                    "base_risk": state_feats[i]["base_risk"],
                    "stress_t": float(min(1.0, state_feats[i]["stress"] + macro_override))
                },
                "label_next_high_risk": int(state_feats[i]["stress"] > 0.7)
            })

        timeline.append({"t": step, "nodes": deepcopy(snapshot_nodes), "edges": deepcopy(edges_list)})

    return timeline


@app.route("/stress_node", methods=["POST"])
def stress_node():
    """
    Request JSON:
    {
      "timestep": 3,                 # base snapshot to start from (1-based)
      "node": "HDFC Bank",           # institution name (must exist in snapshot)
      "scenario": "liquidity_drain", # see supported scenarios
      "magnitude": 0.6,              # 0..1
      "steps": 5,                    # number of contagion steps to simulate
      "macro": 0.0                   # optional macro override
    }

    Response:
    {
      "timeline": [
         {"t":0, "nodes":[{id, features, label_next_high_risk}], "edges":[...]},
         {"t":1, ...}, ...
      ]
    }
    """

    payload = request.json or {}
    t = int(payload.get("timestep", 1))
    node_name = payload.get("node", None)
    scenario = payload.get("scenario", "liquidity_drain")
    magnitude = float(payload.get("magnitude", 0.5))
    steps = int(payload.get("steps", 5))
    macro = float(payload.get("macro", 0.0))

    # validate snapshot
    data, nodes, edges = load_snapshot(t)
    if data is None:
        return jsonify({"error": "Invalid timestep"}), 400

    # Check node exists
    node_names = [n["id"] for n in nodes]
    if node_name not in node_names:
        return jsonify({"error": f"Node '{node_name}' not found in timestep {t}"}), 400

    # Run contagion simulation from the snapshot (seed shock + multi-step)
    timeline = _simulate_contagion_from_snapshot(nodes, edges, node_name,
                                                scenario_type=scenario,
                                                magnitude=magnitude,
                                                steps=steps,
                                                macro_override=macro)

    # For each timeline snapshot, compute the GNN probabilities (run model)
    # We'll reuse existing loader logic to build Data objects for each snapshot
    # and compute probs (model expects same node order as snapshot)
    timeline_with_preds = []
    for snap in timeline:
        # convert nodes to tensor x and edges to edge_index and weights same as train
        node_list = snap["nodes"]
        edge_list = snap["edges"]
        # build x
        x_list = []
        for n in node_list:
            f = n["features"]
            x_list.append([
                f["liquidity_ratio"],
                f["capital_adequacy"],
                f["leverage"],
                f["base_risk"],
                f["stress_t"]
            ])
        x = torch.tensor(x_list, dtype=torch.float32).to(DEVICE)

        # build edge_index and edge_weight from edge_list using node order
        name_to_idx = {n["id"]: idx for idx, n in enumerate(node_list)}
        ei0 = []; ei1 = []; ew = []
        for e in edge_list:
            u = name_to_idx[e["source"]]; v = name_to_idx[e["target"]]; w = float(e["exposure"])
            ei0.extend([u, v]); ei1.extend([v, u]); ew.extend([w, w])
        if len(ei0) == 0:
            edge_index = torch.empty((2,0), dtype=torch.long).to(DEVICE)
            edge_weight = None
        else:
            edge_index = torch.tensor([ei0, ei1], dtype=torch.long).to(DEVICE)
            edge_weight = torch.tensor(ew, dtype=torch.float32).to(DEVICE)

        # run model
        with torch.no_grad():
            logits = model(x, edge_index, edge_weight)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        # attach prob to node snapshot copy
        snap_nodes_with_preds = []
        for n_obj, p in zip(node_list, probs):
            sn = deepcopy(n_obj)
            sn["predicted_prob_high_risk"] = float(p)
            sn["stress"] = float(n_obj["features"]["stress_t"])
            sn["prob"] = float(p)
            sn["trueLabel"] = int(n_obj.get("label_next_high_risk", 0))
            snap_nodes_with_preds.append(sn)

        timeline_with_preds.append({
            "t": snap["t"],
            "nodes": snap_nodes_with_preds,
            "edges": edge_list
        })

    return jsonify({"timeline": timeline_with_preds})

if __name__ == "__main__":
    app.run(debug=True)
