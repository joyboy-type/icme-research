import math
import torch
from torch import nn
from torch.nn import functional as F


class TaskConditionedMLP(nn.Module):
    """Exact replica of soco_grid_corrected/run.py Adapter('task_conditioned_single')."""
    def __init__(self, d_in=772, r=64, h=384):
        super().__init__()
        self.h = h
        self.a = nn.Parameter(torch.randn(d_in, r) * 0.05)
        self.b = nn.Parameter(torch.zeros(r, h))

    def forward(self, x, task):
        bits = F.one_hot(torch.tensor(task, device=x.device), 2).float().expand(*x.shape[:-1], 2)
        x = torch.cat([x[..., :-2], bits], -1)
        z = F.gelu(x @ self.a) @ self.b
        return F.normalize(x[..., :self.h] + z, dim=-1)


def build_2d_rope(grid_h=20, grid_w=20, head_dim=32, base=100.0):
    """Precompute 2D Rotary Position Embeddings (cos, sin) of shape (1, 1, H*W, head_dim)."""
    assert head_dim % 4 == 0, f"head_dim {head_dim} must be divisible by 4 for 2D RoPE"
    half = head_dim // 2
    quarter = head_dim // 4
    inv_freq = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))

    gy, gx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing='ij',
    )
    gx = gx.flatten()  # (400,)
    gy = gy.flatten()  # (400,)

    freqs_x = torch.outer(gx, inv_freq)  # (400, quarter)
    freqs_y = torch.outer(gy, inv_freq)  # (400, quarter)
    freqs = torch.cat([freqs_x, freqs_x, freqs_y, freqs_y], dim=-1)  # (400, head_dim)
    cos = freqs.cos()[None, None]  # (1, 1, 400, head_dim)
    sin = freqs.sin()[None, None]  # (1, 1, 400, head_dim)
    return cos, sin


def apply_2d_rope(t, cos, sin):
    """t: (B, heads, N, head_dim)."""
    half = t.shape[-1] // 2
    t_x, t_y = t[..., :half], t[..., half:]
    q = half // 2

    def rot(u):
        u1, u2 = u[..., :q], u[..., q:]
        return torch.cat([-u2, u1], dim=-1)

    t_rot = torch.cat([rot(t_x), rot(t_y)], dim=-1)
    return t * cos + t_rot * sin


def build_2d_relpos_table_and_index(grid_h=20, grid_w=20):
    """
    Precompute the (2H-1)*(2W-1) unique relative offset features (shape: (M, 5))
    and the (N, N) lookup index mapping each pair (i, j) to its offset in [0, M).
    105x faster than running relpos_mlp on (400, 400, 5) directly!
    """
    n_dy = 2 * grid_h - 1
    n_dx = 2 * grid_w - 1
    dys = torch.arange(-(grid_h - 1), grid_h, dtype=torch.float32) / max(grid_h - 1, 1)
    dxs = torch.arange(-(grid_w - 1), grid_w, dtype=torch.float32) / max(grid_w - 1, 1)
    gdy, gdx = torch.meshgrid(dys, dxs, indexing='ij')
    dx = gdx.flatten()
    dy = gdy.flatten()
    adx = dx.abs()
    ady = dy.abs()
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    rel_table = torch.stack([dx, dy, adx, ady, dist], dim=-1)  # ((2H-1)*(2W-1), 5)

    gy, gx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.long),
        torch.arange(grid_w, dtype=torch.long),
        indexing='ij',
    )
    gx_f, gy_f = gx.flatten(), gy.flatten()
    idx_dx = (gx_f[:, None] - gx_f[None, :]) + (grid_w - 1)  # [0, 2W-2]
    idx_dy = (gy_f[:, None] - gy_f[None, :]) + (grid_h - 1)  # [0, 2H-2]
    rel_idx = idx_dy * n_dx + idx_dx  # (N, N)
    return rel_table, rel_idx


class RelPosBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.1, grid_size=20, use_relpos=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_relpos = use_relpos
        assert self.head_dim * num_heads == dim

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        if use_relpos:
            self.relpos_mlp = nn.Sequential(
                nn.Linear(5, 32),
                nn.GELU(),
                nn.Linear(32, num_heads),
            )
            # Initialize bias output near zero for stable start
            nn.init.zeros_(self.relpos_mlp[-1].weight)
            nn.init.zeros_(self.relpos_mlp[-1].bias)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        if use_relpos:
            cos, sin = build_2d_rope(grid_size, grid_size, self.head_dim)
            rel_table, rel_idx = build_2d_relpos_table_and_index(grid_size, grid_size)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
            self.register_buffer('rel_table', rel_table, persistent=False)
            self.register_buffer('rel_idx', rel_idx, persistent=False)

    def forward(self, x):
        # x: (B, 400, dim)
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        if self.use_relpos:
            q = apply_2d_rope(q, self.rope_cos, self.rope_sin)
            k = apply_2d_rope(k, self.rope_cos, self.rope_sin)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)
        if self.use_relpos:
            bias_table = self.relpos_mlp(self.rel_table)  # (1521, heads)
            rel_bias = bias_table[self.rel_idx].permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
            attn = attn + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj_drop(self.proj(out))
        x = x + self.ffn(self.norm2(x))
        return x


class RelPosBackbone(nn.Module):
    """Shared 2-Layer Cross-Patch Adapter operating on 20x20=400 patch tokens."""
    def __init__(self, in_dim=384, dim=128, num_heads=4, num_layers=2, mlp_ratio=2.0, dropout=0.1, use_relpos=True):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            RelPosBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, use_relpos=use_relpos)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, feat_map_bchw):
        # feat_map_bchw: (B, 384, 20, 20), L2-normalized along C
        B, C, H, W = feat_map_bchw.shape
        tokens = feat_map_bchw.flatten(2).transpose(1, 2)  # (B, 400, 384)
        h = self.in_norm(self.in_proj(tokens))
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return tokens, h


class RelPosSingleAdapter(nn.Module):
    """Model (3): 2-Layer 2D-RelPos Cross-Patch Adapter (single head / mixed supervision)."""
    def __init__(self, in_dim=384, dim=128, num_heads=4, num_layers=2, mlp_ratio=2.0, dropout=0.1, use_relpos=True):
        super().__init__()
        self.backbone = RelPosBackbone(in_dim, dim, num_heads, num_layers, mlp_ratio, dropout, use_relpos=use_relpos)
        self.out_proj = nn.Linear(dim, in_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, feat_map_bchw):
        B, C, H, W = feat_map_bchw.shape
        tokens, h = self.backbone(feat_map_bchw)
        out = F.normalize(tokens + self.out_proj(h), dim=-1)
        return out.transpose(1, 2).reshape(B, C, H, W)


class RelPosOrthSubspaceAdapter(nn.Module):
    """
    Model (4): 2-Layer Cross-Patch Adapter + Dual-Head / Orthogonal Semantic-Role Subspace Decomposition.
    - use_relpos: toggle 2D-RoPE + continuous 2D relative position bias (False => plain self-attention).
    - use_orth: toggle pointwise Gram-Schmidt projection + global subspace orthogonality loss
      (False => plain dual-head with L2-normalized z_role and orth_loss=0).
    """
    def __init__(
        self,
        in_dim=384,
        dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
        freeze_sem=False,
        use_relpos=True,
        use_orth=True,
    ):
        super().__init__()
        self.freeze_sem = freeze_sem
        self.use_relpos = use_relpos
        self.use_orth = use_orth
        self.backbone = RelPosBackbone(in_dim, dim, num_heads, num_layers, mlp_ratio, dropout, use_relpos=use_relpos)
        if not freeze_sem:
            self.sem_proj = nn.Linear(dim, in_dim, bias=False)
            nn.init.zeros_(self.sem_proj.weight)
        self.role_proj = nn.Linear(dim, in_dim, bias=True)
        nn.init.normal_(self.role_proj.weight, std=0.02)
        nn.init.zeros_(self.role_proj.bias)

    def forward(self, feat_map_bchw):
        B, C, H, W = feat_map_bchw.shape
        tokens, h = self.backbone(feat_map_bchw)

        if self.freeze_sem:
            z_sem = tokens
        else:
            z_sem = F.normalize(tokens + self.sem_proj(h), dim=-1)

        raw_role = self.role_proj(h)
        if self.use_orth:
            # Pointwise Gram-Schmidt orthogonal projection against z_sem.detach()
            z_sem_det = z_sem.detach()
            proj = (raw_role * z_sem_det).sum(dim=-1, keepdim=True) * z_sem_det
            r_orth = raw_role - proj
            z_role = F.normalize(r_orth, dim=-1)

            # Global subspace cross-covariance orthogonality loss
            c_mat = torch.bmm(z_sem_det.transpose(1, 2), z_role) / float(H * W)
            orth_loss = (c_mat ** 2).sum(dim=(1, 2)).mean()
        else:
            z_role = F.normalize(raw_role, dim=-1)
            orth_loss = torch.zeros((), device=feat_map_bchw.device, dtype=feat_map_bchw.dtype)

        # Cosine anchor loss to frozen DINOv2
        anchor_loss = (1.0 - (z_sem * tokens).sum(dim=-1)).mean()

        sem_map = z_sem.transpose(1, 2).reshape(B, C, H, W)
        role_map = z_role.transpose(1, 2).reshape(B, C, H, W)
        return sem_map, role_map, orth_loss, anchor_loss


if __name__ == '__main__':
    m2 = TaskConditionedMLP()
    m3 = RelPosSingleAdapter(dim=128, num_heads=4)
    m4 = RelPosOrthSubspaceAdapter(dim=128, num_heads=4)
    for name, m in [('TaskConditionedMLP', m2), ('RelPosSingleAdapter(d=128)', m3), ('RelPosOrthSubspaceAdapter(d=128)', m4)]:
        print(name, 'params:', sum(p.numel() for p in m.parameters()))
