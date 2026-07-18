import torch
import torch.nn as nn


class SheafLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, rank),
            nn.ReLU(),
            nn.Linear(rank, rank)
        )
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index

        xi = x[row]
        xj = x[col]

        edge_feat = torch.cat([xi, xj], dim=-1)
        weights = self.edge_mlp(edge_feat)  # [E, rank]

        msg = xj.unsqueeze(-1) * weights.unsqueeze(1)
        msg = msg.mean(dim=-1)

        out = torch.zeros_like(x)
        out.index_add_(0, row, msg)

        return self.lin(out)


class SheafTemporal(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=1, dropout=0.2):
        super().__init__()
        self.layer1 = SheafLayer(in_dim, hidden)
        self.layer2 = SheafLayer(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, temporal_seq=None):
        h = self.layer1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        h = self.layer2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        return self.lin(h).squeeze(-1)

