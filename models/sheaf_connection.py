# models/sheaf_connection.py
# ------------------------------------------------------------
# Sheaf-Connection (Sheaf-lite)
# - Tương thích project hiện tại (run_global_gat.py gọi forward(..., temporal_seq=...))
# - Không full SheafNN, nhưng đúng tinh thần "connection-style":
#   + restriction map nhỏ theo cạnh (rotation 2D theo từng stalk)
#   + residual + layer norm
#   + dùng temporal context (mean theo time) -> project vào hidden
# - Fix mismatch temporal feature dim bằng LazyLinear
# ------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SheafConnectionTemporal(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 1,
        dropout: float = 0.2,
        stalk_dim: int = 2,
    ):
        super().__init__()

        if stalk_dim != 2:
            raise ValueError(
                "Phiên bản tối giản này chỉ hỗ trợ stalk_dim=2 để dùng rotation ổn định."
            )

        # hidden phải chia hết cho stalk_dim để reshape thành stalks (S,2)
        hidden_eff = (hidden // stalk_dim) * stalk_dim
        if hidden_eff < stalk_dim:
            hidden_eff = stalk_dim

        self.hidden = hidden_eff
        self.out_dim = out_dim
        self.dropout = dropout
        self.stalk_dim = stalk_dim
        self.num_stalks = self.hidden // self.stalk_dim  # S

        # ----- input projection -----
        self.in_proj = nn.Linear(in_dim, self.hidden)

        # ----- temporal projection (lazy) -----
        # temporal_seq.mean(dim=1) có thể ra [N, 1] hoặc [N, Ft]
        # LazyLinear tự suy ra Ft ở lần forward đầu tiên
        self.temporal_proj = nn.LazyLinear(self.hidden)

        # ----- edge MLP sinh góc quay cho mỗi stalk trên mỗi cạnh -----
        # edge feature: [h_src, h_dst, |h_src - h_dst|] => 3H
        edge_feat_dim = self.hidden * 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.num_stalks),  # theta per stalk
        )

        # ----- norm + output head -----
        self.norm = nn.LayerNorm(self.hidden)
        self.out_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, out_dim),
        )

    def _build_temporal_context(self, temporal_seq: torch.Tensor | None) -> torch.Tensor | None:
        """
        temporal_seq: [N, T, Ft] hoặc None
        """
        if temporal_seq is None:
            return None
        if temporal_seq.dim() != 3:
            return None

        # mean pooling theo time -> [N, Ft]
        t_ctx = temporal_seq.mean(dim=1)
        if t_ctx.dim() != 2:
            return None

        # LazyLinear sẽ tự nhận Ft
        return self.temporal_proj(t_ctx.float())

    def _edge_rotations(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """
        Sinh rotation matrix 2x2 cho từng stalk trên mỗi cạnh.
        Input:
          h_src, h_dst: [E, H]
        Output:
          rot: [E, S, 2, 2]
        """
        edge_feat = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst)], dim=-1)  # [E, 3H]
        theta = self.edge_mlp(edge_feat)  # [E, S]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rot = torch.zeros(
            theta.size(0), self.num_stalks, 2, 2,
            device=theta.device,
            dtype=theta.dtype
        )
        # [ cos -sin ]
        # [ sin  cos ]
        rot[:, :, 0, 0] = cos_t
        rot[:, :, 0, 1] = -sin_t
        rot[:, :, 1, 0] = sin_t
        rot[:, :, 1, 1] = cos_t
        return rot

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: [N, F]
        edge_index: [2, E]
        temporal_seq: [N, T, Ft] hoặc None
        """
        if x.dim() != 2:
            raise ValueError(f"x phải có shape [N, F], nhận được {tuple(x.shape)}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index phải có shape [2, E], nhận được {tuple(edge_index.shape)}")

        # ----- base embedding -----
        h = self.in_proj(x)  # [N, H]

        # ----- add temporal context -----
        t_ctx = self._build_temporal_context(temporal_seq)
        if t_ctx is not None:
            h = h + t_ctx

        h = F.relu(h)

        # ----- edges -----
        src = edge_index[0].long().to(h.device)
        dst = edge_index[1].long().to(h.device)

        h_src = h[src]  # [E, H]
        h_dst = h[dst]  # [E, H]

        # reshape thành stalks nhỏ 2D
        h_src_stalk = h_src.view(-1, self.num_stalks, self.stalk_dim)  # [E, S, 2]
        h_dst_stalk = h_dst.view(-1, self.num_stalks, self.stalk_dim)  # [E, S, 2]

        # restriction maps kiểu rotation cho từng cạnh
        rot = self._edge_rotations(h_src, h_dst)  # [E, S, 2, 2]

        # mapped_src = R_e(h_src)
        mapped_src = torch.einsum("esab,esb->esa", rot, h_src_stalk)  # [E, S, 2]

        # disagreement: R(h_src) - h_dst
        diff = mapped_src - h_dst_stalk  # [E, S, 2]

        # aggregate về node đích: giảm disagreement
        agg_dst = torch.zeros_like(h).view(-1, self.num_stalks, self.stalk_dim)  # [N, S, 2]
        agg_dst.index_add_(0, dst, -diff)

        # aggregate về node nguồn (đối xứng mềm)
        agg_src = torch.zeros_like(h).view(-1, self.num_stalks, self.stalk_dim)  # [N, S, 2]
        agg_src.index_add_(0, src, diff)

        # tổng hợp + reshape về [N, H]
        agg = 0.5 * (agg_dst + agg_src)  # [N, S, 2]
        agg = agg.reshape(-1, self.hidden)  # [N, H]

        # residual + norm
        h = h + F.dropout(agg, p=self.dropout, training=self.training)
        h = self.norm(h)
        h = F.relu(h)

        out = self.out_head(h)  # [N, out_dim]

        # chuẩn output cho run_global_gat: nếu out_dim=1 -> [N]
        if out.dim() == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)

        return out