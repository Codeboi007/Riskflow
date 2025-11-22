import os
import json
import glob
import torch
import numpy as np

from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import RiskGCN


def load_snapshots(snapshot_dir):
    files = sorted(glob.glob(os.path.join(snapshot_dir, "G*.json")))
    graphs = []

    for f in files:
        with open(f, "r") as fp:
            snap = json.load(fp)

        nodes = snap["nodes"]
        edges = snap["edges"]

        # Node features: [liquidity, capital_adequacy, leverage, base_risk, stress_t]
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

        # Edges
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
        graphs.append(data)

    return graphs


def train_model(graphs, model_save_dir="models", epochs=80, batch_size=2, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(model_save_dir, exist_ok=True)

    # Split graphs into train and test
    n_total = len(graphs)
    n_train = max(1, int(0.8 * n_total))
    n_test = n_total - n_train
    train_graphs, test_graphs = random_split(graphs, [n_train, n_test])

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    in_channels = graphs[0].x.shape[1]
    model = RiskGCN(in_channels=in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % 10 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | test_acc={acc:.4f}")

    # Save model and metadata
    model_path = os.path.join(model_save_dir, "gnn_model.pt")
    torch.save(model.state_dict(), model_path)

    meta = {
        "in_channels": in_channels,
        "epochs": epochs
    }
    with open(os.path.join(model_save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_path}")


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == batch.y).sum().item()
            total += batch.y.numel()

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    snapshot_dir = os.path.join("data", "snapshots")
    graphs = load_snapshots(snapshot_dir)
    print(f"Loaded {len(graphs)} temporal snapshots.")
    train_model(graphs)
