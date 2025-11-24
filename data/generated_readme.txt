This folder contains a synthetic temporal financial contagion dataset.

It represents a stylized inter-institution exposure network (banks, NBFCs, mutual funds, etc.)
with CHAOTIC stress dynamics. Each snapshot Gk.json corresponds to timestep k and includes:
- nodes: node-level features at time t
- label_next_high_risk: whether the institution becomes high-risk at time t+1
- edges: weighted exposures between institutions.
