"""
Hierarchical Transformer + Pointer network for CRP-D (hier_pg).

Architecture
------------
One shared Transformer encoder encodes all stack nodes.
Two separate pointer decoders handle the two-level decision:

    high_decoder: HIGH-level — select which TARGET-GROUP stack to process next.
        Mask contains only stacks that hold a container of the current target group.

    low_decoder:  LOW-level  — select where to RELOCATE a blocking container.
        Mask contains all non-full stacks except the source stack.

Input features per stack (INPUT_DIM = 12)
------------------------------------------
    0  stack_id          normalised stack index                [0, 1]
    1  row               fixed = 1 in CRP-D yard               [0, 1]
    2  height            current number of containers           [0, 1]
    3  free_space        max_tiers - height                     [0, 1]
    4  has_target        1 if top container is current group    {0,1}
    5  top_group         group of top container                 [0, 1]
    6  min_group         smallest group in stack                [0, 1]
    7  num_target        # containers of current group in stack [0, 1]
    8  target_depth      distance of shallowest target from top [0, 1]
    9  blockers_above    # blockers on top of shallowest target [0, 1]
    10 is_well_ordered   1 if stack has no "bad pair" (later grp below earlier) {0,1}
    11 num_bad_pairs     # bad pairs (later grp below earlier grp)  [0, 1]

The last node (index -1) encodes the vessel/target-group context and uses the
same 12-feature format (most fields set to 0 except stack_id and top_group).

The network normalises each feature by feature_scale before encoding.
Existing CRP-D environments provide 5-feature obs; _reshape() zero-pads to
INPUT_DIM automatically so the network works with both old and new envs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ================================================================ #
#  Shared attention and encoder blocks (identical to pg_fgb)        #
# ================================================================ #

class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout   = nn.Dropout(dropout)
        self.q_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        def split(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        Q = split(self.q_proj(query))
        K = split(self.k_proj(key))
        V = split(self.v_proj(value))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(out)


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.self_attn = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, ffn_dim=None, dropout=0.0):
        super().__init__()
        ffn_dim = ffn_dim or 4 * embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        h = self.input_norm(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, mask)
        return h


# ================================================================ #
#  Pointer decoder — shared implementation used by both levels      #
# ================================================================ #

class _PointerDecoder(nn.Module):
    """
    Cross-attention pointer that selects one stack from a masked set.

    query_mode: how to build the context query.
        "target_global" — target node + mean of all stack nodes (default for
                          HIGH-level: "which target-group stack to process?")
        "global_only"   — mean of all stack nodes (for LOW-level: "where to
                          relocate the blocker?")
    """

    def __init__(
        self,
        embed_dim:    int,
        num_heads:    int,
        clip_constant: float = 10.0,
        dropout:      float = 0.0,
        query_mode:   str   = "target_global",
    ):
        super().__init__()
        assert query_mode in ("target_global", "global_only")
        self.embed_dim     = embed_dim
        self.clip_constant = clip_constant
        self.query_mode    = query_mode
        self.cross_attn    = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm          = nn.LayerNorm(embed_dim)
        self.pointer_k     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pointer_q     = nn.Linear(embed_dim, embed_dim, bias=False)

    def _build_query(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Build (B, 1, D) context query."""
        stack_emb  = encoder_output[:, :-1, :]               # (B, S, D)
        global_emb = stack_emb.mean(dim=1, keepdim=True)     # (B, 1, D)
        if self.query_mode == "target_global":
            target_emb = encoder_output[:, -1:, :]           # (B, 1, D) — last node
            return target_emb + global_emb
        else:
            return global_emb

    def _pointer_scores(self, query, encoder_output, action_mask):
        n_actions = action_mask.shape[-1]
        act_emb = encoder_output[:, :n_actions, :]           # (B, A, D)
        scale   = math.sqrt(self.embed_dim)
        q       = self.pointer_q(query)                      # (B, 1, D)
        k       = self.pointer_k(act_emb)                    # (B, A, D)
        scores  = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / scale
        scores  = self.clip_constant * torch.tanh(scores)
        scores  = scores.masked_fill(action_mask.bool(), float("-inf"))
        return F.log_softmax(scores, dim=-1)

    def forward(self, encoder_output, action_mask, greedy=False):
        query     = self._build_query(encoder_output)
        context   = self.cross_attn(query, encoder_output, encoder_output)
        context   = self.norm(query + context)
        log_probs = self._pointer_scores(context, encoder_output, action_mask)
        if greedy:
            action = log_probs.argmax(dim=-1)
        else:
            action = Categorical(logits=log_probs).sample()
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob

    def evaluate(self, encoder_output, action_mask, actions):
        """Re-evaluate log-prob and entropy for a batch of actions."""
        query     = self._build_query(encoder_output)
        context   = self.cross_attn(query, encoder_output, encoder_output)
        context   = self.norm(query + context)
        log_probs = self._pointer_scores(context, encoder_output, action_mask)
        selected  = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs.clamp(min=torch.finfo(log_probs.dtype).min)).sum(-1)
        return selected, entropy


# ================================================================ #
#  Hierarchical policy network                                       #
# ================================================================ #

class HierPolicyNetwork(nn.Module):
    """
    Hierarchical Transformer + Pointer network for CRP-D.

    One shared Transformer encoder + two separate pointer decoders:

        high_decoder  (query_mode="target_global"):
            Selects SOURCE stack among stacks that hold current-group containers.
            Called once per target-container selection.

        low_decoder   (query_mode="global_only"):
            Selects DESTINATION stack for a blocking container to be relocated.
            Called when the chosen source stack is not yet accessible.

    Parameters
    ----------
    embed_dim       : Transformer hidden dim (default 128).
    num_enc_layers  : Encoder layers (default 2).
    num_heads       : Attention heads (default 4).
    ffn_dim         : FFN inner dim (default 4×embed_dim).
    clip_constant   : Tanh clipping for pointer scores (default 10).
    feature_scale   : Per-feature normalisation.  Length must equal INPUT_DIM.
                      Defaults to all-ones.  Old 5-feature obs is auto-padded.
    """

    INPUT_DIM = 12   # extended feature vector per stack node

    # Default scale divisors for 12 features.
    # Features 0..4 match pg_fgb (stack_id, row, height, has_target, top_group).
    # Features 5..11 are new hierarchical features.
    _DEFAULT_SCALE = [10., 80., 10., 1., 10.,  # original 5
                      10., 10., 10., 10., 10.,  # free_space, min_group, num_target,
                                                 # target_depth, blockers_above
                      1., 10.]                  # is_well_ordered, num_bad_pairs

    def __init__(
        self,
        embed_dim:      int   = 128,
        num_enc_layers: int   = 2,
        num_heads:      int   = 4,
        ffn_dim:        int   = 256,
        clip_constant:  float = 10.0,
        dropout:        float = 0.0,
        feature_scale:  "torch.Tensor | None" = None,
    ):
        super().__init__()
        if feature_scale is None:
            feature_scale = torch.tensor(self._DEFAULT_SCALE, dtype=torch.float32)
        self.register_buffer("feature_scale", feature_scale.float())

        self.encoder = _TransformerEncoder(
            self.INPUT_DIM, embed_dim, num_enc_layers, num_heads, ffn_dim, dropout
        )
        # High-level: target node context + global mean → choose source stack
        self.high_decoder = _PointerDecoder(
            embed_dim, num_heads, clip_constant, dropout, query_mode="target_global"
        )
        # Low-level: global mean → choose destination stack for relocation
        self.low_decoder = _PointerDecoder(
            embed_dim, num_heads, clip_constant, dropout, query_mode="global_only"
        )

    def _reshape(self, flat_obs: "torch.Tensor") -> "torch.Tensor":
        """
        Accept flat observation of any feature width and reshape to (B, N, INPUT_DIM).

        If the incoming obs has fewer features per node than INPUT_DIM (e.g. the
        original 5-feature CRP-D obs), the extra dimensions are zero-padded so
        the network can be used with the existing platform environments without
        modification.
        """
        if flat_obs.dim() == 3:
            raw = flat_obs.float()
        else:
            raw_features_per_node = self.feature_scale.shape[0]
            # detect actual features/node from obs width
            obs_width = flat_obs.shape[-1]
            # try INPUT_DIM first, fall back to 5 (old obs)
            for f in (self.INPUT_DIM, 5):
                if obs_width % f == 0:
                    n_nodes = obs_width // f
                    raw = flat_obs.float().view(flat_obs.shape[0], n_nodes, f)
                    break
            else:
                raise ValueError(f"Cannot reshape obs width {obs_width} to any known feature dim.")

        # zero-pad to INPUT_DIM if needed
        B, N, F = raw.shape
        if F < self.INPUT_DIM:
            pad = torch.zeros(B, N, self.INPUT_DIM - F, dtype=raw.dtype, device=raw.device)
            raw = torch.cat([raw, pad], dim=-1)
        return raw / self.feature_scale

    def encode(self, flat_obs: "torch.Tensor") -> "torch.Tensor":
        """Return (B, N, D) encoder output shared by both decoders."""
        return self.encoder(self._reshape(flat_obs))

    # ── High-level: source stack selection ──────────────────────── #

    def forward_high(
        self,
        flat_obs:    "torch.Tensor",
        high_mask:   "torch.Tensor",   # (B, S) True = forbidden
        greedy:      bool = False,
        enc_out:     "torch.Tensor | None" = None,
    ):
        """Select source stack (which target-group position to process)."""
        if enc_out is None:
            enc_out = self.encode(flat_obs)
        return self.high_decoder(enc_out, high_mask, greedy=greedy)

    def evaluate_high(
        self,
        flat_obs:  "torch.Tensor",
        high_mask: "torch.Tensor",
        actions:   "torch.Tensor",
    ):
        enc_out = self.encode(flat_obs)
        return self.high_decoder.evaluate(enc_out, high_mask, actions)

    # ── Low-level: destination stack selection ───────────────────── #

    def forward_low(
        self,
        flat_obs:    "torch.Tensor",
        low_mask:    "torch.Tensor",   # (B, S) True = forbidden
        greedy:      bool = False,
        enc_out:     "torch.Tensor | None" = None,
    ):
        """Select destination stack for a blocking container relocation."""
        if enc_out is None:
            enc_out = self.encode(flat_obs)
        return self.low_decoder(enc_out, low_mask, greedy=greedy)

    def evaluate_low(
        self,
        flat_obs: "torch.Tensor",
        low_mask: "torch.Tensor",
        actions:  "torch.Tensor",
    ):
        enc_out = self.encode(flat_obs)
        return self.low_decoder.evaluate(enc_out, low_mask, actions)

    # ── Convenience: auto-dispatch based on mode ─────────────────── #

    def forward(
        self,
        flat_obs:    "torch.Tensor",
        action_mask: "torch.Tensor",
        greedy:      bool = False,
        enc_out:     "torch.Tensor | None" = None,
        mode:        str  = "high",
    ):
        """
        Dispatch to high or low decoder.

        mode: "high" → source-stack selection (default, matches env._mode="high")
              "low"  → destination-stack selection (env._mode="low")
        """
        if enc_out is None:
            enc_out = self.encode(flat_obs)
        if mode == "high":
            return self.high_decoder(enc_out, action_mask, greedy=greedy)
        else:
            return self.low_decoder(enc_out, action_mask, greedy=greedy)

    def evaluate_actions(
        self,
        flat_obs:    "torch.Tensor",
        action_mask: "torch.Tensor",
        actions:     "torch.Tensor",
        mode:        str = "high",
    ):
        enc_out = self.encode(flat_obs)
        if mode == "high":
            return self.high_decoder.evaluate(enc_out, action_mask, actions)
        else:
            return self.low_decoder.evaluate(enc_out, action_mask, actions)
