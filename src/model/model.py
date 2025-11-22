import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class RiskGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)

        # Node-level prediction
        out = self.lin(x).view(-1)  # shape [num_nodes]
        return out  # logits (for BCEWithLogitsLoss)
