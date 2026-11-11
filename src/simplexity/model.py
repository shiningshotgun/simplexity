"""Model architecture definition and configuration.

Paper Appendix A.6 specification:
"context window size of 10, used ReLU as the activation function, and had a head
dimension of 8 and a model dimension of 64. There was 1 attention head in each
of 4 layers. The model had MLPs of dimension 256 and used causal attention masking.
Layer normalization was applied."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SimplexityModelConfig:
    d_vocab: int
    n_ctx: int = 10
    d_model: int = 64
    d_head: int = 8
    n_heads: int = 1
    n_layers: int = 4
    d_mlp: int = 256
    act_fn: str = "relu"
    use_bias: bool = True
    layer_norm_eps: float = 1e-5
    seed: int = 0


def build_hooked_transformer(
    config: SimplexityModelConfig,
    device: str = "cpu",
):
    """Instantiate a TransformerLens HookedTransformer with exact Appendix A.6 settings.

    Raises:
        ImportError if transformer_lens is not installed.
    """
    try:
        from transformer_lens import HookedTransformer, HookedTransformerConfig
    except ImportError as e:
        raise ImportError(
            "transformer_lens is not installed in the active environment. "
            "See REPRODUCTION_REPORT.md for environment status and Hark handoff."
        ) from e

    tl_cfg = HookedTransformerConfig(
        n_layers=config.n_layers,
        d_model=config.d_model,
        n_ctx=config.n_ctx,
        d_head=config.d_head,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
        d_vocab=config.d_vocab,
        act_fn=config.act_fn,
        normalization_type="LN",
        eps=config.layer_norm_eps,
        device=device,
        seed=config.seed,
        default_prepend_bos=False,
    )
    model = HookedTransformer(tl_cfg)
    return model


class SimplexityCausalSelfAttention(nn.Module):
    """1-head causal self-attention with d_head=8 and d_model=64."""

    def __init__(self, config: SimplexityModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.inner_dim = self.n_heads * self.d_head  # 1 * 8 = 8

        self.W_Q = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.W_K = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.W_V = nn.Linear(self.d_model, self.inner_dim, bias=config.use_bias)
        self.W_O = nn.Linear(self.inner_dim, self.d_model, bias=config.use_bias)

        # Causal mask buffer
        mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx)).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        B, T, C = x.shape
        q = self.W_Q(x)  # (B, T, 8)
        k = self.W_K(x)  # (B, T, 8)
        v = self.W_V(x)  # (B, T, 8)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, T, T)
        causal_mask = self.mask[:T, :T]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.bmm(attn_weights, v)  # (B, T, 8)
        return self.W_O(out)  # (B, T, d_model)


class SimplexityMLP(nn.Module):
    """MLP block with dimension 256 and ReLU activation."""

    def __init__(self, config: SimplexityModelConfig) -> None:
        super().__init__()
        self.W_in = nn.Linear(config.d_model, config.d_mlp, bias=config.use_bias)
        self.W_out = nn.Linear(config.d_mlp, config.d_model, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.relu(self.W_in(x)))


class SimplexityBlock(nn.Module):
    """Standard pre-LayerNorm transformer block with causal attention and MLP."""

    def __init__(self, config: SimplexityModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = SimplexityCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = SimplexityMLP(config)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN residual block
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        resid_post = x + mlp_out
        return resid_post, resid_post


class SimplexityTransformer(nn.Module):
    """Clean-room PyTorch fallback model matching HookedTransformer architecture.

    Exposes exact residual stream hookpoints:
        blocks.{0,1,2,3}.hook_resid_post
        ln_final.hook_normalized
    """

    def __init__(self, config: SimplexityModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.d_vocab, config.d_model)
        self.pos_embed = nn.Embedding(config.n_ctx, config.d_model)

        self.blocks = nn.ModuleList(
            [SimplexityBlock(config, i) for i in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.unembed = nn.Linear(config.d_model, config.d_vocab, bias=config.use_bias)

        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(self.config.seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            tokens: (batch_size, seq_len) long tensor of token IDs.
            return_cache: whether to return residual stream activations.

        Returns:
            logits: (batch_size, seq_len, d_vocab)
            cache: dict containing:
                blocks.0.hook_resid_post
                blocks.1.hook_resid_post
                blocks.2.hook_resid_post
                blocks.3.hook_resid_post
                ln_final.hook_normalized
        """
        B, T = tokens.shape
        pos = torch.arange(0, T, device=tokens.device)

        tok_emb = self.embed(tokens)  # (B, T, d_model)
        pos_emb = self.pos_embed(pos)  # (T, d_model)
        x = tok_emb + pos_emb

        cache: Optional[Dict[str, torch.Tensor]] = {} if return_cache else None

        for i, block in enumerate(self.blocks):
            x, resid_post = block(x)
            if return_cache and cache is not None:
                cache[f"blocks.{i}.hook_resid_post"] = resid_post

        normalized = self.ln_final(x)
        if return_cache and cache is not None:
            cache["ln_final.hook_normalized"] = normalized

        logits = self.unembed(normalized)
        return logits, cache


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
