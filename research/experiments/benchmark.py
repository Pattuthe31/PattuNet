import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.PattuNet import PattuNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_tensor(imgs: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    return (imgs - mean) / std


# =====================================================================
# Baseline Architectures: Standard ViT & Static HAViT
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
        q, k, v = qkv.unbind(0)

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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        prev_H = None
        for block in self.blocks:
            x, prev_H = block(x, prev_H)
        return self.head(self.norm(x).select(1, 0))


# =====================================================================
# Scientifically Calibrated Corruptions in Raw Pixel Space
# =====================================================================
def apply_pixel_corruption(raw_imgs: torch.Tensor, c_type: str, severity: int, gen: torch.Generator) -> torch.Tensor:
    if c_type == "clean" or severity == 0:
        return normalize_tensor(raw_imgs)

    if c_type == "gaussian_noise":
        stds = {1: 0.04, 2: 0.08, 3: 0.14}
        sigma = stds.get(severity, 0.08)
        noise = torch.randn(raw_imgs.shape, generator=gen, device=raw_imgs.device, dtype=raw_imgs.dtype) * sigma
        corrupted = torch.clamp(raw_imgs + noise, 0.0, 1.0)
        return normalize_tensor(corrupted)

    if c_type == "motion_blur":
        k_sizes = {1: 3, 2: 5, 3: 7}
        k = k_sizes.get(severity, 5)
        kernel = torch.zeros((3, 1, k, k), device=raw_imgs.device, dtype=raw_imgs.dtype)
        mid = k // 2
        kernel.select(2, mid).fill_(1.0 / k)
        corrupted = torch.clamp(F.conv2d(raw_imgs, kernel, padding=mid, groups=3), 0.0, 1.0)
        return normalize_tensor(corrupted)

    if c_type == "occlusion":
        ratios = {1: 0.25, 2: 0.40, 3: 0.55}
        H = raw_imgs.size(2)
        patch_len = int(H * ratios.get(severity, 0.40))
        corrupted = raw_imgs.clone()
        start = (H - patch_len) // 2
        corrupted.narrow(2, start, patch_len).narrow(3, start, patch_len).fill_(0.5)
        return normalize_tensor(corrupted)

    return normalize_tensor(raw_imgs)


# =====================================================================
# Evaluation & Validation-Only Training Engine (Zero Test Leakage)
# =====================================================================
def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device,
                   c_type: str = "clean", severity: int = 0, num_classes: int = 10, track_rel: bool = False):
    model.eval()
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    correct, total = 0, 0
    rel_sums = [0.0] * 6
    batches = 0

    gen = torch.Generator(device=device)
    gen.manual_seed(12345 + severity * 100)
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for raw_imgs, labels in loader:
            raw_imgs = raw_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            norm_imgs = apply_pixel_corruption(raw_imgs, c_type, severity, gen)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if track_rel:
                    logits, rel_scores = model(norm_imgs, return_diagnostics=True)
                    for idx, val in enumerate(rel_scores):
                        rel_sums[idx] = rel_sums[idx] + val
                    batches += 1
                else:
                    logits = model(norm_imgs)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(num_classes):
                is_pred = (preds == c)
                is_true = (labels == c)
                tp.select(0, c).add_((is_pred & is_true).sum())
                fp.select(0, c).add_((is_pred & (~is_true)).sum())
                fn.select(0, c).add_(((~is_pred) & is_true).sum())

    acc = 100.0 * correct / total
    f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    macro_f1 = 100.0 * f1_per_class.mean().item()
    avg_rel = [round(s / max(batches, 1), 4) for s in rel_sums] if track_rel else None
    return acc, macro_f1, avg_rel


def train_with_val_split(name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                         epochs: int, device: torch.device, ckpt_path: str):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    val_curve = []

    print(f"\n--- Training {name} ({sum(p.numel() for p in model.parameters()):,} params) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        val_acc, _, _ = evaluate_split(model, val_loader, device, "clean", 0)
        val_curve.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% (Best Val: {best_val_acc:.2f}%)")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, val_curve


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "research/experiments/claim_b_v3_results"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Claim B v3 Benchmark Device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_raw = transforms.ToTensor()

    full_train_aug = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    full_train_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_raw)
    test_set_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_raw)

    set_seed(42)
    perm = torch.randperm(len(full_train_aug)).tolist()
    train_indices = perm[:45000]
    val_indices = perm[45000:]

    train_loader = DataLoader(Subset(full_train_aug, train_indices), batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Subset(full_train_raw, val_indices), batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set_raw, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    EPOCHS = 60
    specs = [
        ("Standard ViT", BaselineViT(mode="vit"), os.path.join(out_dir, "vit_best_val.pt")),
        ("Static HAViT", BaselineViT(mode="havit"), os.path.join(out_dir, "havit_best_val.pt")),
        ("PattuNet v3 (Ours)", PattuNet(), os.path.join(out_dir, "pattunet_v3_best_val.pt")),
    ]

    trained_models = {}
    val_curves = {}
    for name, model, ckpt in specs:
        set_seed(42)
        trained_models[name], val_curves[name] = train_with_val_split(
            name, model, train_loader, val_loader, EPOCHS, device, ckpt
        )

    conditions = [
        ("Clean (Test Set)", "clean", 0),
        ("Gaussian Noise (Sev 1)", "gaussian_noise", 1),
        ("Gaussian Noise (Sev 2)", "gaussian_noise", 2),
        ("Gaussian Noise (Sev 3)", "gaussian_noise", 3),
        ("Motion Blur (Sev 1)", "motion_blur", 1),
        ("Motion Blur (Sev 2)", "motion_blur", 2),
        ("Motion Blur (Sev 3)", "motion_blur", 3),
        ("Patch Occlusion (Sev 1)", "occlusion", 1),
        ("Patch Occlusion (Sev 2)", "occlusion", 2),
        ("Patch Occlusion (Sev 3)", "occlusion", 3),
    ]

    print("\n" + "=" * 92)
    print(f"{'Condition':<26} | {'Standard ViT':<19} | {'Static HAViT':<19} | {'PattuNet v3 (Ours)':<19}")
    print(f"{'':<26} | {'Acc% / Macro-F1':<19} | {'Acc% / Macro-F1':<19} | {'Acc% / Macro-F1':<19}")
    print("=" * 92)

    summary_acc = {name: [] for name, _, _ in specs}
    rel_logs = {}
    csv_rows = []

    for cond_label, c_type, sev in conditions:
        row_strs = []
        csv_row = {"Condition": cond_label}
        for name, _, _ in specs:
            is_pattu = "PattuNet" in name
            acc, f1, rel = evaluate_split(trained_models[name], test_loader, device, c_type, sev, track_rel=is_pattu)
            summary_acc[name].append(acc)
            row_strs.append(f"{acc:5.2f}% / {f1:5.2f}")
            csv_row[f"{name}_Acc"] = round(acc, 2)
            csv_row[f"{name}_F1"] = round(f1, 2)
            if is_pattu:
                rel_logs[cond_label] = rel
        print(f"{cond_label:<26} | " + " | ".join(r.rjust(19) for r in row_strs))
        csv_rows.append(csv_row)

    print("=" * 92)

    csv_path = os.path.join(out_dir, "robustness_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(csv_rows)).keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n--- Mean Corruption Accuracy (mCA) & Degradation Summary ---")
    for name, _, _ in specs:
        accs = summary_acc[name]
        clean_acc = next(iter(accs))
        corrupt_accs = list(accs)[1:]
        mca = sum(corrupt_accs) / len(corrupt_accs)
        drop = clean_acc - mca
        print(f"{name:<22} | Clean: {clean_acc:.2f}% | mCA: {mca:.2f}% | Avg Drop: -{drop:.2f}%")

    print("\n--- Learned PattuNet v3 Layer-Wise Reliability Gate (r_l) ---")
    for cond_key in ("Clean (Test Set)", "Gaussian Noise (Sev 2)", "Motion Blur (Sev 2)", "Patch Occlusion (Sev 2)"):
        print(f"  {cond_key:<26}: {rel_logs.get(cond_key)}")

    fig, (ax_val, ax_rob, ax_rel) = plt.subplots(1, 3, figsize=(17, 4.8))

    for name, _, _ in specs:
        ax_val.plot(range(1, EPOCHS + 1), val_curves[name], linewidth=2, label=name)
    ax_val.set_title("Validation Learning Curves (Zero Test Leakage)")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Accuracy (%)")
    ax_val.legend()
    ax_val.grid(True, alpha=0.3)

    x_idx = list(range(len(conditions)))
    short_labels = ["Clean", "GN-1", "GN-2", "GN-3", "MB-1", "MB-2", "MB-3", "Occ-1", "Occ-2", "Occ-3"]
    for name, _, _ in specs:
        ax_rob.plot(x_idx, summary_acc[name], marker="o", linewidth=2, label=name)
    ax_rob.set_xticks(x_idx)
    ax_rob.set_xticklabels(short_labels, rotation=30)
    ax_rob.set_ylabel("Test Top-1 Accuracy (%)")
    ax_rob.set_title("Clean vs. Corrupted Test Accuracy")
    ax_rob.legend()
    ax_rob.grid(True, alpha=0.3)

    layers = list(range(1, 7))
    for cond_key in ("Clean (Test Set)", "Gaussian Noise (Sev 2)", "Motion Blur (Sev 2)", "Patch Occlusion (Sev 2)"):
        ax_rel.plot(layers, rel_logs.get(cond_key), marker="s", linewidth=2, label=cond_key)
    ax_rel.set_xlabel("Transformer Layer (l)")
    ax_rel.set_ylabel("Mean Reliability Score (r_l)")
    ax_rel.set_title("PattuNet v3 Reliability Gate Across Layers")
    ax_rel.set_ylim(0, 1.0)
    ax_rel.legend()
    ax_rel.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "claim_b_v3_publication_figures.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Figures to: {fig_path}")