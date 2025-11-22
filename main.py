import os
import json
import torch
from flask import Flask, render_template, jsonify, request
from torch_geometric.data import Data

from src.model.gnn_model import RiskGCN

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model metadata
META_PATH = os.path.join("models", "meta.json")
MODEL_PATH = os.path.join("models", "gnn_model.pt")
SNAPSHOT_DIR = os.path.join("data", "snapshots")

with open(META_PATH, "r") as f:
    meta = json.load(f)

IN_CHANNELS = meta["in_channels"]

model = RiskGCN(in_channels=IN_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def load_snapshot(timestep: int):
    """
    timestep is 1-based (G1.json, G2.json, ...)
    """
    fname = os.path.join(SNAPSHOT_DIR, f"G{timestep}.json")
    if not os.path.exists(fname):
        return None, None, None

    with open(fname, "r") as fp:
        snap = json.load(fp)

    nodes = snap["nodes"]
    edges = snap["edges"]

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
        u = e["source"]
        v = e["target"]
        w = e["exposure"]
        edge_index_list[0].extend([u, v])
        edge_index_list[1].extend([v, u])
        edge_weight_list.extend([w, w])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

    return data, nodes, edges


@app.route("/")
def index():
    # Discover how many snapshots we have
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.startswith("G") and f.endswith(".json")]
    timesteps = sorted(int(f[1:-5]) for f in files)  # from "G1.json" -> 1
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

    # Attach predictions to nodes
    results = []
    for node, p in zip(nodes, probs):
        results.append({
            "id": node["id"],
            "features": node["features"],
            "label_next_high_risk": node["label_next_high_risk"],
            "predicted_prob_high_risk": float(p)
        })

    return jsonify({
        "timestep": t,
        "nodes": results,
        "edges": edges
    })


if __name__ == "__main__":
    app.run(debug=True)
