import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=1, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, temporal_seq=None):
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        return self.lin(h).squeeze(-1)

