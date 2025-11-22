import os
import json
import numpy as np
import networkx as nx

N_NODES = 40
N_TIMESTEPS = 12
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def create_base_graph(n_nodes=N_NODES, edge_prob=0.18):
    """
    Create a base exposure network between institutions.
    Nodes = institutions
    Edges = exposure relationships with weights.
    """
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=RANDOM_SEED, directed=False)

    # Ensure graph is connected enough
    if not nx.is_connected(G):
        # connect components by adding edges
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v)

    # Assign exposure weights
    for u, v in G.edges():
        G[u][v]['exposure'] = float(np.round(np.random.uniform(0.1, 1.0), 3))

    return G


def initialize_node_features(G):
    """
    Base financial features (synthetic but realistic-feeling).
    """
    for node in G.nodes():
        liquidity_ratio = np.round(np.random.uniform(0.2, 1.0), 3)       # higher is safer
        capital_adequacy = np.round(np.random.uniform(0.08, 0.2), 3)     # typical small range
        leverage = np.round(np.random.uniform(5, 20), 3)                 # higher = riskier
        base_risk = np.clip(np.random.normal(0.3, 0.1), 0.0, 1.0)

        G.nodes[node]['liquidity_ratio'] = float(liquidity_ratio)
        G.nodes[node]['capital_adequacy'] = float(capital_adequacy)
        G.nodes[node]['leverage'] = float(leverage)
        G.nodes[node]['base_risk'] = float(base_risk)

    return G


def simulate_temporal_stress(G, n_timesteps=N_TIMESTEPS):
    """
    Simulate stress over time with gradual contagion and partial recovery.
    Returns: list of dicts for each timestep containing node stress.
    """
    # Initialize stress at t=0
    stress_history = []

    # initial stress = base_risk + small systemic noise
    initial_stress = {}
    systemic_noise = np.clip(np.random.normal(0.05, 0.02), 0, 0.15)
    for node in G.nodes():
        base = G.nodes[node]['base_risk']
        s0 = np.clip(base + systemic_noise + np.random.normal(0.0, 0.02), 0.0, 1.0)
        initial_stress[node] = float(s0)

    stress_history.append(initial_stress)

    # simulate through time
    for t in range(1, n_timesteps):
        prev_stress = stress_history[-1]
        current_stress = {}

        # mild macro shock that changes over time (could be negative too)
        macro_shock = np.clip(np.random.normal(0.0, 0.03), -0.1, 0.1)

        for node in G.nodes():
            base_risk = G.nodes[node]['base_risk']
            liquidity = G.nodes[node]['liquidity_ratio']

            # contagion term: weighted average neighbor stress
            neighbors = list(G.neighbors(node))
            if neighbors:
                num = 0.0
                den = 0.0
                for nb in neighbors:
                    exposure = G[node][nb]['exposure']
                    num += exposure * prev_stress[nb]
                    den += exposure
                neighbor_effect = num / den
            else:
                neighbor_effect = 0.0

            # core risk aggregation
            stress_raw = (
                0.4 * base_risk +
                0.4 * neighbor_effect +
                0.2 * (1.0 - liquidity) +
                macro_shock
            )

            # contagion shock if highly stressed
            if prev_stress[node] > 0.75:
                stress_raw += 0.1

            # partial recovery for good liquidity
            stress_raw -= 0.12 * liquidity

            # small noise
            stress_raw += np.random.normal(0.0, 0.02)

            stress_val = float(np.clip(stress_raw, 0.0, 1.0))
            current_stress[node] = stress_val

        stress_history.append(current_stress)

    return stress_history


def save_snapshots(G, stress_history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_timesteps = len(stress_history)

    for t in range(n_timesteps - 1):  # last step won't have next-label
        snapshot = {
            "timestep": t,
            "nodes": [],
            "edges": []
        }

        for node in G.nodes():
            stress_t = stress_history[t][node]
            stress_next = stress_history[t + 1][node]
            label_next_high_risk = int(stress_next > 0.7)

            node_data = {
                "id": int(node),
                "features": {
                    "liquidity_ratio": G.nodes[node]['liquidity_ratio'],
                    "capital_adequacy": G.nodes[node]['capital_adequacy'],
                    "leverage": G.nodes[node]['leverage'],
                    "base_risk": G.nodes[node]['base_risk'],
                    "stress_t": stress_t
                },
                "label_next_high_risk": label_next_high_risk
            }
            snapshot["nodes"].append(node_data)

        for u, v, data in G.edges(data=True):
            snapshot["edges"].append({
                "source": int(u),
                "target": int(v),
                "exposure": data['exposure']
            })

        fname = os.path.join(out_dir, f"G{t+1}.json")
        with open(fname, "w") as f:
            json.dump(snapshot, f, indent=2)

    print(f"Saved {n_timesteps - 1} snapshots to {out_dir}")


def write_generated_readme(base_dir):
    text = """This folder contains a synthetic temporal financial contagion dataset.

It represents a stylized inter-institution exposure network (banks, NBFCs, mutual funds, etc.)
used in multiple open-source systemic risk simulations. Node-level features capture liquidity,
capital adequacy, leverage and base risk. Temporal stress is simulated with gradual contagion,
macro shocks, and partial recovery dynamics.

Each snapshot Gk.json corresponds to timestep k and includes:
- nodes: features at time t
- label_next_high_risk: whether the institution becomes high-risk at time t+1
- edges: weighted exposures between institutions.
"""
    with open(os.path.join(base_dir, "generated_readme.txt"), "w") as f:
        f.write(text)


def main():
    base_dir = os.path.join("data")
    snapshots_dir = os.path.join(base_dir, "snapshots")
    os.makedirs(base_dir, exist_ok=True)

    G = create_base_graph()
    G = initialize_node_features(G)
    stress_history = simulate_temporal_stress(G)
    save_snapshots(G, stress_history, snapshots_dir)
    write_generated_readme(base_dir)


if __name__ == "__main__":
    main()
