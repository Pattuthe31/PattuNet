import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PattuAttention(nn.Module):
    """
    Reliability-Gated Historical Self-Attention (PATTU).
    Implements Equations (1) through (6) of PattuNet.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        # Standard QKV and output projections
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable reliability gate parameters per head: w_s, w_c, b, and beta
        self.w_s = nn.Parameter(torch.ones(1, num_heads, 1))
        self.w_c = nn.Parameter(torch.ones(1, num_heads, 1))
        self.b = nn.Parameter(torch.zeros(1, num_heads, 1))
        self.beta_logit = nn.Parameter(torch.zeros(1, num_heads, 1))  # sigmoid -> (0, 1)

    def forward(self, x: torch.Tensor, prev_H: torch.Tensor = None, prev_r: torch.Tensor = None):
        """
        Args:
            x: Input token embeddings of shape (B, N, D)
            prev_H: Historical attention matrix H_{l-1} of shape (B, H, N, N) or None at layer 1
            prev_r: Previous layer reliability r_{l-1} of shape (B, H, N) or None at layer 1
        Returns:
            out: Refined token representations (B, N, D)
            H_l: Updated historical attention matrix (B, H, N, N)
            r_l: Current token-wise reliability scores (B, H, N)
        """
        B, N, D = x.shape

        # Equation (1): Standard Scaled Dot-Product Attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, head_dim)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        A_l = F.softmax(attn_logits, dim=-1)  # (B, H, N, N)

        # Equation (2): Attention Sharpness (Inverse Normalized Shannon Entropy)
        log_N = math.log(max(N, 2))
        entropy = -torch.sum(A_l * torch.log(A_l + self.eps), dim=-1)  # (B, H, N)
        S_l = 1.0 - (entropy / log_N)  # Normalized in [0, 1]

        # Initialize historical state at Layer 1
        if prev_H is None or prev_r is None:
            C_l = torch.ones_like(S_l)  # Self-consistency at layer 1 is 1.0
            r_l = torch.sigmoid(self.w_s * S_l + self.w_c * C_l + self.b)
            H_l = A_l
        else:
            # Equation (3): Cross-Layer Cosine Consistency
            dot_product = torch.sum(A_l * prev_H, dim=-1)
            norm_A = torch.norm(A_l, p=2, dim=-1)
            norm_H = torch.norm(prev_H, p=2, dim=-1)
            C_l = dot_product / (norm_A * norm_H + self.eps)  # (B, H, N)

            # Equation (4): Unified Token-Wise Reliability Score
            r_l = torch.sigmoid(self.w_s * S_l + self.w_c * C_l + self.b)  # (B, H, N)

            # Equation (5): Reliability-Gated Historical Blending
            beta = torch.sigmoid(self.beta_logit)  # Bound beta strictly in (0, 1)
            gate = (beta * prev_r).unsqueeze(-1)  # (B, H, N, 1)
            H_l = (1.0 - gate) * A_l + gate * prev_H  # (B, H, N, N)

        # Equation (6): Reliability-Weighted Output Aggregation
        r_diag = r_l.unsqueeze(-1)  # (B, H, N, 1)
        A_tilde = r_diag * H_l + (1.0 - r_diag) * A_l
        A_tilde = self.attn_drop(A_tilde)

        out = (A_tilde @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))

        return out, H_l, r_l


class PattuBlock(nn.Module):
    """Transformer Encoder Block powered by PattuAttention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PattuAttention(dim=dim, num_heads=num_heads, dropout=dropout)
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
    """
    Complete Vision Transformer Architecture with Progressive Attention-Reliability Units (PATTU).
    """

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
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stack of PattuBlocks
        self.blocks = nn.ModuleList([
            PattuBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        prev_H, prev_r = None, None
        layer_reliabilities = []

        for block in self.blocks:
            x, prev_H, prev_r = block(x, prev_H=prev_H, prev_r=prev_r)
            if return_diagnostics:
                # Average reliability across batch, heads, and tokens for paper plots
                layer_reliabilities.append(prev_r.mean().item())

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if return_diagnostics:
            return logits, layer_reliabilities
        return logits


# =====================================================================
# Sanity Check & End-to-End Verification Script
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running PattuNet verification on: {device}")

    # 1. Instantiate PattuNet (configured for 32x32 images like CIFAR-10)
    model = PattuNet(
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        depth=6,
        num_heads=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # 2. Test Forward Pass & Extract Layer-wise Reliability Diagnostics
    dummy_images = torch.randn(8, 3, 32, 32, device=device)
    logits, reliabilities = model(dummy_images, return_diagnostics=True)
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Layer-wise Mean Reliability Scores (r_l): {[round(r, 4) for r in reliabilities]}")

    # 3. Quick Overfit Sanity Test (Verifies Backward Pass & Gradient Flow)
    dataset = TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 4):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Sanity Epoch {epoch}/3 | Loss: {total_loss / len(loader):.4f}")

    print("PattuNet verification completed successfully!")
