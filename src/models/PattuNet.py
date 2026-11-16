import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PattuAttention(nn.Module):
    """
    PATTU v3: Depth-Aware Zero-Trust Reliability-Gated Historical Self-Attention.
    Combines token-wise inverse normalized Shannon entropy (Sharpness S_l) and
    cross-layer cosine agreement (Consistency C_l) to dynamically gate historical
    attention probabilities without log-probability noise amplification.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        layer_idx: int = 0,
        depth: int = 6,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Depth-progressive prior initialization:
        # Layer 0 starts near -2.5 (Zero-Trust gate ~0.08) to block shallow noise;
        # Deeper layers initialize smoothly higher so semantic priors propagate.
        depth_ratio = layer_idx / max(depth - 1, 1)
        init_bias = -2.5 + 2.2 * depth_ratio

        self.w_s = nn.Parameter(torch.full((1, num_heads, 1), 2.0))
        self.w_c = nn.Parameter(torch.full((1, num_heads, 1), 2.0))
        self.b = nn.Parameter(torch.full((1, num_heads, 1), init_bias))
        # Learnable maximum blending ceiling bounded in (0, 0.80)
        self.alpha_max_logit = nn.Parameter(torch.full((1, num_heads, 1, 1), 1.0))

    def forward(self, x: torch.Tensor, prev_H: torch.Tensor = None, prev_r: torch.Tensor = None):
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Equation (1): Scaled Dot-Product Self-Attention
        raw_logits = (q @ k.transpose(-2, -1)) * self.scale
        A_l = F.softmax(raw_logits, dim=-1)

        # Equation (2): Token-Wise Attention Sharpness (Inverse Normalized Entropy)
        log_N = math.log(max(N, 2))
        entropy = -torch.sum(A_l * torch.log(A_l + self.eps), dim=-1)
        S_l = torch.clamp(1.0 - (entropy / log_N), min=0.0, max=1.0)

        # Zero-Trust Layer 1 Initialization (no historical consensus exists yet)
        if prev_H is None or prev_r is None:
            C_l = torch.zeros_like(S_l)
            r_l = torch.sigmoid(self.w_s * S_l + self.w_c * C_l + self.b)
            A_tilde = A_l
            H_l = A_l
        else:
            # Equation (3): Cross-Layer Cosine Consistency
            dot_product = torch.sum(A_l * prev_H, dim=-1)
            norm_A = torch.norm(A_l, p=2, dim=-1)
            norm_H = torch.norm(prev_H, p=2, dim=-1)
            C_l = dot_product / (norm_A * norm_H + self.eps)

            # Equation (4): Unified Token-Wise Reliability Gate r_l in (0, 1)
            r_l = torch.sigmoid(self.w_s * S_l + self.w_c * C_l + self.b)

            # Equation (5): Bounded Reliability-Gated Probability Blending
            # Smoothly suppresses high-frequency noise when r_l is high,
            # while blocking shallow noise when r_l is near 0.
            alpha_cap = 0.80 * torch.sigmoid(self.alpha_max_logit)
            gate = alpha_cap * r_l.unsqueeze(-1)
            A_tilde = (1.0 - gate) * A_l + gate * prev_H

            # Equation (6): Historical Attention Trajectory Update
            H_l = A_tilde

        attn_weights = self.attn_drop(A_tilde)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))

        return out, H_l, r_l


class PattuBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_idx: int = 0,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PattuAttention(
            dim=dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            depth=depth,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, prev_H: torch.Tensor = None, prev_r: torch.Tensor = None):
        attn_out, H_l, r_l = self.attn(self.norm1(x), prev_H=prev_H, prev_r=prev_r)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, H_l, r_l


class PattuNet(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PattuBlock(
                dim=embed_dim,
                num_heads=num_heads,
                layer_idx=idx,
                depth=depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for idx in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        prev_H, prev_r = None, None
        layer_reliabilities = []

        for block in self.blocks:
            x, prev_H, prev_r = block(x, prev_H=prev_H, prev_r=prev_r)
            if return_diagnostics:
                layer_reliabilities.append(prev_r.mean().item())

        logits = self.head(self.norm(x).select(1, 0))
        if return_diagnostics:
            return logits, layer_reliabilities
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PattuNet().to(device)
    dummy = torch.randn(8, 3, 32, 32, device=device)
    logits, rel = model(dummy, return_diagnostics=True)
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"PattuNet v3 Initial Reliability Curve (r_l): {[round(r, 4) for r in rel]}")
