# src/generator/generate_data.py
import os
import json
import numpy as np
import networkx as nx

# ====== CONFIG ======
N_NODES = 40
N_TIMESTEPS = 12
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

COMPANY_NAMES = [
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra Bank",
    "Bank of Baroda",
    "Punjab National Bank",
    "Canara Bank",
    "Union Bank of India",
    "IDBI Bank",
    "Bajaj Finance",
    "Mahindra Finance",
    "Tata Capital",
    "Shriram Finance",
    "Muthoot Finance",
    "LIC of India",
    "SBI Life Insurance",
    "HDFC Life Insurance",
    "ICICI Prudential",
    "Bajaj Allianz Life",
    "Reliance Mutual Fund",
    "HDFC Mutual Fund",
    "ICICI Prudential MF",
    "Nippon India Mutual Fund",
    "UTI Mutual Fund",
    "Paytm Payments Bank",
    "PhonePe Payments",
    "Airtel Payments Bank",
    "Razorpay",
    "Cashfree Payments",
    "Standard Chartered India",
    "HSBC India",
    "Citi India",
    "Deutsche Bank India",
    "JP Morgan India",
    "NPCI (UPI)",
    "NSE Clearing Corp",
    "CCIL",
    "SEBI Surveillance",
    "RBI Monitoring Node"
]

assert len(COMPANY_NAMES) == N_NODES, "COMPANY_NAMES length must equal N_NODES"

# ===== GRAPH CREATION =====
def create_base_graph(n_nodes=N_NODES):
    # scale-free like network (Barabasi-Albert)
    G = nx.barabasi_albert_graph(n_nodes, m=3, seed=RANDOM_SEED)
    # assign exposures (heavy-tail: some large exposures)
    for u, v in G.edges():
        if np.random.rand() < 0.18:
            exposure = np.random.uniform(1.0, 3.0)
        else:
            exposure = np.random.uniform(0.1, 0.8)
        G[u][v]['exposure'] = float(np.round(exposure, 3))
    return G

def initialize_node_features(G):
    deg_cent = nx.degree_centrality(G)
    for node in G.nodes():
        liquidity_ratio = float(np.round(np.random.uniform(0.1, 1.0), 3))
        capital_adequacy = float(np.round(np.random.uniform(0.06, 0.2), 3))
        leverage_raw = float(np.random.uniform(5, 30))
        leverage_scaled = float(np.clip((leverage_raw - 5.0) / 25.0, 0.0, 1.0))
        base_risk = float(np.clip(np.random.normal(0.35, 0.15), 0.0, 1.0))
        systemic_importance = float(np.clip(deg_cent[node], 0.0, 1.0))

        G.nodes[node]['liquidity_ratio'] = liquidity_ratio
        G.nodes[node]['capital_adequacy'] = capital_adequacy
        G.nodes[node]['leverage'] = leverage_raw
        G.nodes[node]['leverage_scaled'] = leverage_scaled
        G.nodes[node]['base_risk'] = base_risk
        G.nodes[node]['systemic_importance'] = systemic_importance
        # Also store display name
        G.nodes[node]['name'] = COMPANY_NAMES[node]
    return G

# ====== CRISIS EVENTS ======
def pick_crisis_events(G, n_timesteps=N_TIMESTEPS, n_events=2):
    events = []
    all_nodes = list(G.nodes())
    possible_ts = list(range(2, max(3, n_timesteps - 2)))
    if not possible_ts:
        possible_ts = [1]
    chosen_ts = np.random.choice(possible_ts, size=min(n_events, len(possible_ts)), replace=False)
    for t in chosen_ts:
        seed = int(np.random.choice(all_nodes))
        cluster = set([seed])
        for nb in G.neighbors(seed):
            cluster.add(nb)
            for nb2 in G.neighbors(nb):
                if len(cluster) < 10:
                    cluster.add(nb2)
        shock_mag = float(np.random.uniform(0.3, 0.7))
        events.append({
            "timestep": int(t),
            "nodes": list(cluster),
            "magnitude": shock_mag
        })
    return events

# ====== SIMULATION ======
def simulate_temporal_stress(G, n_timesteps=N_TIMESTEPS):
    stress_history = []
    initial_stress = {}
    initial_macro = float(np.clip(np.random.normal(0.1, 0.05), 0, 0.3))
    vulnerable_seeds = list(np.random.choice(list(G.nodes()), size=3, replace=False))
    for node in G.nodes():
        base = G.nodes[node]['base_risk']
        extra = 0.0
        if node in vulnerable_seeds:
            extra += 0.4
        s0 = float(np.clip(base + initial_macro + extra + np.random.normal(0.0, 0.05), 0.0, 1.0))
        initial_stress[node] = s0
    stress_history.append(initial_stress)
    crisis_events = pick_crisis_events(G, n_timesteps=n_timesteps, n_events=2)

    for t in range(1, n_timesteps):
        prev_stress = stress_history[-1]
        current_stress = {}
        macro_shock = float(np.random.normal(0.0, 0.15))
        if np.random.rand() < 0.05:
            macro_shock += float(np.random.choice([0.5, -0.5]))
        forced_failure_node = None
        if np.random.rand() < 0.06:
            forced_failure_node = int(np.random.choice(list(G.nodes())))
        crisis_nodes = []
        crisis_mag = 0.0
        for ev in crisis_events:
            if ev["timestep"] == t:
                crisis_nodes = ev["nodes"]
                crisis_mag = ev["magnitude"]
                break

        for node in G.nodes():
            base_risk = G.nodes[node]['base_risk']
            liquidity = G.nodes[node]['liquidity_ratio']
            lev_scaled = G.nodes[node]['leverage_scaled']
            sys_imp = G.nodes[node]['systemic_importance']

            neighbors = list(G.neighbors(node))
            lin_sum = 0.0
            lin_den = 0.0
            nl_sum = 0.0
            nl_den = 0.0
            for nb in neighbors:
                exposure = G[node][nb]['exposure']
                s_nb = prev_stress[nb]
                lin_sum += exposure * s_nb
                lin_den += exposure
                if s_nb > 0.7:
                    nl_sum += exposure * (s_nb ** 2)
                    nl_den += exposure
            lin_effect = (lin_sum / lin_den) if lin_den > 0 else 0.0
            nl_effect = (nl_sum / nl_den) if nl_den > 0 else 0.0
            contagion = 0.6 * lin_effect + 0.4 * nl_effect
            amplif = 0.4 * lev_scaled + 0.25 * (1.0 - liquidity) + 0.25 * sys_imp
            if np.random.rand() < 0.12:
                idio_shock = float(np.random.uniform(-0.4, 0.6))
            else:
                idio_shock = float(np.random.normal(0.0, 0.05))
            recovery = 0.02 * liquidity
            stress_raw = (
                0.25 * base_risk +
                0.35 * contagion * (1.0 + amplif) +
                0.25 * prev_stress[node] +
                macro_shock +
                idio_shock -
                recovery
            )
            if node in crisis_nodes:
                stress_raw += crisis_mag
            if forced_failure_node is not None and node == forced_failure_node:
                stress_raw = 1.0
            stress_val = float(np.clip(stress_raw, 0.0, 1.0))
            current_stress[node] = stress_val
        stress_history.append(current_stress)
    return stress_history

# ====== SAVE SNAPSHOTS ======
def save_snapshots(G, stress_history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_timesteps = len(stress_history)
    for t in range(n_timesteps - 1):
        snapshot = {"timestep": t, "nodes": [], "edges": []}
        for node in G.nodes():
            stress_t = stress_history[t][node]
            stress_next = stress_history[t + 1][node]
            label_next_high_risk = int(stress_next > 0.7)
            node_data = {
                "id": G.nodes[node]['name'],
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
                "source": G.nodes[u]['name'],
                "target": G.nodes[v]['name'],
                "exposure": data['exposure']
            })
        fname = os.path.join(out_dir, f"G{t+1}.json")
        with open(fname, "w") as f:
            json.dump(snapshot, f, indent=2)
    print(f"Saved {n_timesteps - 1} snapshots to {out_dir}")

def write_generated_readme(base_dir):
    text = """This folder contains a synthetic temporal financial contagion dataset.

It represents a stylized inter-institution exposure network (banks, NBFCs, mutual funds, etc.)
with CHAOTIC stress dynamics. Each snapshot Gk.json corresponds to timestep k and includes:
- nodes: node-level features at time t
- label_next_high_risk: whether the institution becomes high-risk at time t+1
- edges: weighted exposures between institutions.
"""
    with open(os.path.join(base_dir, "generated_readme.txt"), "w") as f:
        f.write(text)

def main():
    base_dir = os.path.join("data")
    snapshots_dir = os.path.join(base_dir, "snapshots")
    os.makedirs(base_dir, exist_ok=True)
    if os.path.isdir(snapshots_dir):
        for fname in os.listdir(snapshots_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(snapshots_dir, fname))
    G = create_base_graph()
    G = initialize_node_features(G)
    stress_history = simulate_temporal_stress(G)
    save_snapshots(G, stress_history, snapshots_dir)
    write_generated_readme(base_dir)

if __name__ == "__main__":
    main()
