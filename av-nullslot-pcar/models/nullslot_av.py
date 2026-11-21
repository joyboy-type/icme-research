"""Null-Slot Cross-Attention & Audio-Visual Event Localization / Parsing Model.

Supports attention modes:
  - "standard"     : Standard Multi-Head Cross-Attention (softmax sums to 1 over opposite modality)
  - "nullslot"     : Null-Slot Cross-Attention (learnable per-head abstain key with zero value)
  - "sigmoid_gate" : Query-only Sigmoid Gate applied to standard cross-attention output (W1 baseline)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NullSlotCrossAttention(nn.Module):
    """Multi-head cross-attention with configurable abstention / gating operator."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_null_slot: bool = True,
        attn_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        if attn_mode is None:
            attn_mode = "nullslot" if use_null_slot else "standard"
        assert attn_mode in ("standard", "nullslot", "sigmoid_gate"), f"Unknown attn_mode={attn_mode}"
        self.attn_mode = attn_mode
        self.use_null_slot = attn_mode == "nullslot"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        if self.attn_mode == "nullslot":
            self.null_k = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head) * 0.02)
            self.null_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
            self.q_gate = None
        elif self.attn_mode == "sigmoid_gate":
            self.register_parameter("null_k", None)
            self.register_parameter("null_bias", None)
            # Query-only per-head gate g(q) in (0, 1); abstain mass = 1 - g(q)
            self.q_gate = nn.Linear(d_model, n_heads)
        else:
            self.register_parameter("null_k", None)
            self.register_parameter("null_bias", None)
            self.q_gate = None

        self.last_aux: Dict[str, Optional[torch.Tensor]] = {}

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            q_in:  [B, T_q, D]
            kv_in: [B, T_k, D]
        Returns:
            out:        [B, T_q, D]
            alpha_null: [B, T_q] mean abstention mass across heads, or None
        """
        B, T_q, D = q_in.shape
        T_k = kv_in.shape[1]

        q = self.q_proj(q_in).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T_q, T_k]

        if self.attn_mode == "nullslot":
            null_k = self.null_k.expand(B, -1, -1, -1)  # [B, H, 1, d_h]
            null_logit = torch.matmul(q, null_k.transpose(-2, -1)) * scale + self.null_bias  # [B, H, T_q, 1]
            logits_aug = torch.cat([logits, null_logit], dim=-1)  # [B, H, T_q, T_k + 1]
            probs_aug = F.softmax(logits_aug, dim=-1)
            alpha_null_h = probs_aug[..., -1]  # [B, H, T_q]
            # Exact log-odds: logit(alpha_null) = s_null - logsumexp_j(s_j)
            null_logodds = null_logit.squeeze(-1) - torch.logsumexp(logits, dim=-1)  # [B, H, T_q]
            alpha_null = alpha_null_h.mean(dim=1)  # [B, T_q]
            probs = self.attn_drop(probs_aug[..., :-1])
            ctx = torch.matmul(probs, v)  # [B, H, T_q, d_h]
        elif self.attn_mode == "sigmoid_gate":
            probs = self.attn_drop(F.softmax(logits, dim=-1))
            ctx = torch.matmul(probs, v)  # [B, H, T_q, d_h]
            gate_logits = self.q_gate(q_in).transpose(1, 2)  # [B, H, T_q]
            gate = torch.sigmoid(gate_logits)  # [B, H, T_q]
            ctx = ctx * gate.unsqueeze(-1)
            alpha_null_h = 1.0 - gate  # [B, H, T_q]
            null_logodds = -gate_logits  # logit(1 - sigmoid(g)) = -g
            alpha_null = alpha_null_h.mean(dim=1)  # [B, T_q]
        else:
            probs = self.attn_drop(F.softmax(logits, dim=-1))
            ctx = torch.matmul(probs, v)
            alpha_null = None
            alpha_null_h = None
            null_logodds = None

        ctx = ctx.transpose(1, 2).contiguous().view(B, T_q, D)
        out = self.out_proj(ctx)
        fusion_ratio = out.norm(dim=-1) / (q_in.norm(dim=-1) + 1e-6)  # [B, T_q]
        self.last_aux = {
            "alpha_h": alpha_null_h,
            "logodds": null_logodds,
            "fusion_ratio": fusion_ratio,
        }
        return out, alpha_null


class HybridAVLayer(nn.Module):
    """One layer of per-modality temporal self-attention + bidirectional cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_null_slot: bool = True,
        attn_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sa_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.sa_v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_sa_a = nn.LayerNorm(d_model)
        self.ln_sa_v = nn.LayerNorm(d_model)

        self.xa_a2v = NullSlotCrossAttention(
            d_model, n_heads, dropout, use_null_slot=use_null_slot, attn_mode=attn_mode
        )
        self.xa_v2a = NullSlotCrossAttention(
            d_model, n_heads, dropout, use_null_slot=use_null_slot, attn_mode=attn_mode
        )
        self.ln_xa_a = nn.LayerNorm(d_model)
        self.ln_xa_v = nn.LayerNorm(d_model)

        self.ln_ffn_a = nn.LayerNorm(d_model)
        self.ln_ffn_v = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, h_a: torch.Tensor, h_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Temporal self-attention within each modality
        na = self.ln_sa_a(h_a)
        nv = self.ln_sa_v(h_v)
        h_a = h_a + self.drop(self.sa_a(na, na, na, need_weights=False)[0])
        h_v = h_v + self.drop(self.sa_v(nv, nv, nv, need_weights=False)[0])

        # Bidirectional cross-attention
        qa = self.ln_xa_a(h_a)
        qv = self.ln_xa_v(h_v)
        ctx_a, null_a2v = self.xa_a2v(qa, qv)
        ctx_v, null_v2a = self.xa_v2a(qv, qa)
        h_a = h_a + self.drop(ctx_a)
        h_v = h_v + self.drop(ctx_v)

        # Feed-forward
        h_a = h_a + self.ffn_a(self.ln_ffn_a(h_a))
        h_v = h_v + self.ffn_v(self.ln_ffn_v(h_v))
        return h_a, h_v, null_a2v, null_v2a


class AVNullSlotModel(nn.Module):
    """Audio-Visual Temporal Model supporting both AVE (segment localization) and LLP (AVVP)."""

    def __init__(
        self,
        d_audio: int,
        d_visual: int,
        num_classes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 16,
        dropout: float = 0.2,
        use_null_slot: bool = True,
        attn_mode: Optional[str] = None,
        use_arc: bool = False,
    ) -> None:
        super().__init__()
        if attn_mode is None:
            attn_mode = "nullslot" if use_null_slot else "standard"
        self.attn_mode = attn_mode
        self.use_null_slot = attn_mode == "nullslot"
        self.use_arc = use_arc and (attn_mode in ("nullslot", "sigmoid_gate"))

        self.audio_proj = nn.Sequential(
            nn.Linear(d_audio, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(d_visual, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [
                HybridAVLayer(
                    d_model, n_heads, dropout, use_null_slot=self.use_null_slot, attn_mode=attn_mode
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_out_a = nn.LayerNorm(d_model)
        self.ln_out_v = nn.LayerNorm(d_model)

        # Stream-specific classification heads
        self.head_a = nn.Linear(d_model, num_classes)
        self.head_v = nn.Linear(d_model, num_classes)

        # Joint audio-visual classification head
        self.head_av = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.arc_gamma = nn.Parameter(torch.zeros(1)) if self.use_arc else None

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> Dict[str, object]:
        B, T, _ = audio.shape
        h_a = self.audio_proj(audio) + self.pos_emb[:, :T, :]
        h_v = self.visual_proj(visual) + self.pos_emb[:, :T, :]

        null_a2v_list: List[torch.Tensor] = []
        null_v2a_list: List[torch.Tensor] = []
        aux_a2v: List[Dict[str, Optional[torch.Tensor]]] = []
        aux_v2a: List[Dict[str, Optional[torch.Tensor]]] = []

        for layer in self.layers:
            h_a, h_v, na, nv = layer(h_a, h_v)
            aux_a2v.append(layer.xa_a2v.last_aux)
            aux_v2a.append(layer.xa_v2a.last_aux)
            if na is not None:
                null_a2v_list.append(na)
                null_v2a_list.append(nv)

        h_a = self.ln_out_a(h_a)
        h_v = self.ln_out_v(h_v)

        logits_a = self.head_a(h_a)
        logits_v = self.head_v(h_v)
        joint_feat = torch.cat([h_a, h_v, h_a * h_v], dim=-1)
        logits_av = self.head_av(joint_feat)

        alpha_null_a2v = torch.stack(null_a2v_list, dim=0).mean(dim=0) if null_a2v_list else None
        alpha_null_v2a = torch.stack(null_v2a_list, dim=0).mean(dim=0) if null_v2a_list else None

        if self.arc_gamma is not None and alpha_null_a2v is not None and alpha_null_v2a is not None:
            la = torch.logit(alpha_null_a2v.clamp(1e-4, 1.0 - 1e-4))
            lv = torch.logit(alpha_null_v2a.clamp(1e-4, 1.0 - 1e-4))
            bg_shift = torch.zeros_like(logits_av)
            bg_shift[..., -1] = self.arc_gamma * 0.5 * (la + lv)
            logits_av = logits_av + bg_shift

        return {
            "logits_a": logits_a,
            "logits_v": logits_v,
            "logits_av": logits_av,
            "h_a": h_a,
            "h_v": h_v,
            "alpha_null_a2v": alpha_null_a2v,
            "alpha_null_v2a": alpha_null_v2a,
            "aux_a2v": aux_a2v,
            "aux_v2a": aux_v2a,
        }
