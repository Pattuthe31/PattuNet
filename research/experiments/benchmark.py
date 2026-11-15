import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Allow importing PattuNet from src/models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.PattuNet import PattuNet


# =====================================================================
# Baseline 1 & 2: Standard ViT (mode='vit') & Static HAViT (mode='havit')
# =====================================================================
class BaselineAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, mode: str = "vit", alpha: float = 0.45):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mode = mode
        self.alpha = alpha

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, prev_H: torch.Tensor = None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv, qkv, qkv

        A_l = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)

        if self.mode == "havit" and prev_H is not None:
            H_l = (1.0 - self.alpha) * A_l + self.alpha * prev_H
        else:
            H_l = A_l

        out = (self.attn_drop(H_l) @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out)), H_l


class BaselineBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mode: str = "vit", mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BaselineAttention(dim, num_heads, dropout, mode=mode)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, prev_H: torch.Tensor = None):
        attn_out, H_l = self.attn(self.norm1(x), prev_H)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, H_l


class BaselineViT(nn.Module):
    def __init__(self, mode: str = "vit", img_size: int = 32, patch_size: int = 4, num_classes: int = 10,
                 embed_dim: int = 128, depth: int = 6, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([BaselineBlock(embed_dim, num_heads, mode=mode, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        B = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        prev_H = None
        for block in self.blocks:
            x, prev_H = block(x, prev_H)
        return self.head(self.norm(x)[:, 0])


# =====================================================================
# Training & Evaluation Engine
# =====================================================================
def train_and_evaluate(name: str, model: nn.Module, train_loader, test_loader, epochs: int, device: torch.device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start_time = time.time()

    print(f"\n--- Training {name} ({sum(p.numel() for p in model.parameters()):,} params) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    elapsed = time.time() - start_time
    return best_acc, elapsed


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Device: {device}")

    # CIFAR-10 Dataset with Standard Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    EPOCHS = 30  # Increase to 100 for final paper numbers
    models = {
        "1. Standard ViT": BaselineViT(mode="vit"),
        "2. Static HAViT (alpha=0.45)": BaselineViT(mode="havit"),
        "3. PattuNet (Ours)": PattuNet(),
    }

    results = {}
    for name, model in models.items():
        best_acc, elapsed = train_and_evaluate(name, model, train_loader, test_loader, EPOCHS, device)
        results[name] = (best_acc, elapsed)

    print("\n" + "=" * 55)
    print(f"{'Model':<30} | {'Best Acc (%)':<12} | {'Time (s)':<8}")
    print("=" * 55)
    for name, (acc, t) in results.items():
        print(f"{name:<30} | {acc:<12.2f} | {t:<8.1f}")
    print("=" * 55)

    # Extract final learned layer-wise reliability curve from PattuNet
    pattu_model = models["3. PattuNet (Ours)"].eval()
    sample_imgs, _ = next(iter(test_loader))
    with torch.no_grad():
        _, rel_scores = pattu_model(sample_imgs.to(device), return_diagnostics=True)
    print(f"\nLearned PattuNet Layer-wise Reliability (r_l): {[round(r, 4) for r in rel_scores]}")
