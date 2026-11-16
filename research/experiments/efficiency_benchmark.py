import os
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.PattuNet import PattuNet
from research.experiments.benchmark import BaselineViT, CIFAR10_MEAN, CIFAR10_STD


def estimate_block_flops(num_tokens: int, dim: int = 128, mlp_ratio: float = 4.0, active_mlp_tokens: float = None) -> float:
    """
    Computes exact MACs/FLOPs for a single Transformer block given active MLP token count.
    Attention QKV + Proj + QK^T + AV runs on all N tokens; MLP runs only on active_mlp_tokens.
    """
    if active_mlp_tokens is None:
        active_mlp_tokens = float(num_tokens)

    hidden = int(dim * mlp_ratio)
    # Attention projections (QKV + Out): 4 * N * D^2
    attn_proj_flops = 4.0 * num_tokens * (dim ** 2)
    # Attention matrix multiplications (QK^T + AV): 2 * N^2 * D
    attn_mat_flops = 2.0 * (num_tokens ** 2) * dim
    # Pointwise MLP on active tokens only: 2 * N_active * D * hidden
    mlp_flops = 2.0 * active_mlp_tokens * dim * hidden
    return attn_proj_flops + attn_mat_flops + mlp_flops


def evaluate_pattu_with_token_halting(
    model: PattuNet,
    loader: DataLoader,
    device: torch.device,
    tau: float = 1.0,
    occlusion_ratio: float = 0.0,
):
    """
    Evaluates PattuNet v3 with Reliability-Guided Token Halting:
    Tokens whose head-averaged reliability r_l(i) exceeds threshold tau in layers >= 1
    have stabilized semantically and bypass redundant MLP refinement (identity residual path),
    saving compute while preserving CLS token aggregation.
    """
    model.eval()
    correct, total = 0, 0
    depth = len(model.blocks)
    num_tokens = 65  # 1 CLS + 64 patches (32x32 with 4x4 patch)
    skipped_tokens_per_layer = [0.0] * depth
    total_tokens_per_layer = [0.0] * depth

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if occlusion_ratio > 0.0:
                H = imgs.size(2)
                patch_len = int(H * occlusion_ratio)
                start = (H - patch_len) // 2
                imgs = imgs.clone()
                imgs.narrow(2, start, patch_len).narrow(3, start, patch_len).fill_(0.0)

            B = imgs.size(0)
            x = model.patch_embed(imgs).flatten(2).transpose(1, 2)
            x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
            x = model.pos_drop(x + model.pos_embed)

            prev_H, prev_r = None, None
            for idx, block in enumerate(model.blocks):
                attn_out, prev_H, prev_r = block.attn(block.norm1(x), prev_H=prev_H, prev_r=prev_r)
                x = x + attn_out

                # Head-averaged token reliability: shape (B, N)
                token_rel = prev_r.mean(dim=1)

                # Never skip the CLS token (index 0) or Layer 0 (Zero-Trust warmup)
                if idx == 0 or tau >= 1.0:
                    x = x + block.mlp(block.norm2(x))
                    active_mask = torch.ones((B, num_tokens), device=device, dtype=torch.bool)
                else:
                    # Tokens with reliability <= tau continue through MLP; converged tokens skip MLP
                    active_mask = (token_rel <= tau)
                    active_mask.select(1, 0).fill_(True)  # Keep CLS token active

                    # Execute MLP only on active tokens to realize genuine compute reduction
                    normed = block.norm2(x)
                    mlp_delta = torch.zeros_like(x)
                    if active_mask.any():
                        selected_tokens = normed[active_mask]
                        mlp_delta[active_mask] = block.mlp(selected_tokens)
                    x = x + mlp_delta

                active_count = active_mask.float().sum().item()
                total_count = float(B * num_tokens)
                skipped_tokens_per_layer[idx] += (total_count - active_count)
                total_tokens_per_layer[idx] += total_count

            logits = model.head(model.norm(x).select(1, 0))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    avg_skip_pct = 100.0 * sum(skipped_tokens_per_layer) / sum(total_tokens_per_layer)

    # Compute exact MFLOPs per image (Patch Embed + 6 Transformer Blocks + Head)
    patch_flops = 3.0 * 128.0 * 16.0 * 64.0
    total_flops = patch_flops + (128.0 * 10.0)
    for idx in range(depth):
        skip_frac = skipped_tokens_per_layer[idx] / total_tokens_per_layer[idx]
        active_toks = num_tokens * (1.0 - skip_frac)
        total_flops += estimate_block_flops(num_tokens, dim=128, mlp_ratio=4.0, active_mlp_tokens=active_toks)

    mflops = total_flops / 1e6
    return acc, avg_skip_pct, mflops


def measure_throughput(model: nn.Module, device: torch.device, batch_size: int = 256, iters: int = 100, tau: float = None):
    """Measures GPU inference throughput (images/sec) with CUDA synchronization."""
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            if tau is None or tau >= 1.0:
                _ = model(dummy)
            else:
                B = dummy.size(0)
                x = model.patch_embed(dummy).flatten(2).transpose(1, 2)
                x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
                x = model.pos_drop(x + model.pos_embed)
                prev_H, prev_r = None, None
                for idx, block in enumerate(model.blocks):
                    attn_out, prev_H, prev_r = block.attn(block.norm1(x), prev_H=prev_H, prev_r=prev_r)
                    x = x + attn_out
                    if idx == 0:
                        x = x + block.mlp(block.norm2(x))
                    else:
                        mask = (prev_r.mean(dim=1) <= tau)
                        mask.select(1, 0).fill_(True)
                        normed = block.norm2(x)
                        delta = torch.zeros_like(x)
                        if mask.any():
                            delta[mask] = block.mlp(normed[mask])
                        x = x + delta
                _ = model.head(model.norm(x).select(1, 0))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (batch_size * iters) / elapsed


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "research/experiments/claim_b_v3_results"
    out_dir = "research/experiments/claim_a_efficiency_results"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Claim A Efficiency & Token-Halting Benchmark Device: {device}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    # Load existing v3 checkpoints
    vit_model = BaselineViT(mode="vit").to(device)
    vit_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "vit_best_val.pt"), map_location=device, weights_only=True))

    havit_model = BaselineViT(mode="havit").to(device)
    havit_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "havit_best_val.pt"), map_location=device, weights_only=True))

    pattu_model = PattuNet().to(device)
    pattu_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pattunet_v3_best_val.pt"), map_location=device, weights_only=True))

    # Full-compute baseline MFLOPs
    base_mflops = (3.0 * 128.0 * 16.0 * 64.0 + 128.0 * 10.0 + 6.0 * estimate_block_flops(65, 128, 4.0)) / 1e6

    # Evaluate Baselines
    def eval_baseline(m, occ=0.0):
        m.eval()
        c, t = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if occ > 0.0:
                    H = imgs.size(2)
                    plen = int(H * occ)
                    st = (H - plen) // 2
                    imgs = imgs.clone()
                    imgs.narrow(2, st, plen).narrow(3, st, plen).fill_(0.0)
                preds = m(imgs).argmax(dim=1)
                c += (preds == labels).sum().item()
                t += labels.size(0)
        return 100.0 * c / t

    vit_clean = eval_baseline(vit_model, 0.0)
    vit_occ = eval_baseline(vit_model, 0.40)
    vit_tput = measure_throughput(vit_model, device)

    havit_clean = eval_baseline(havit_model, 0.0)
    havit_occ = eval_baseline(havit_model, 0.40)
    havit_tput = measure_throughput(havit_model, device)

    rows = [
        {"Model": "Standard ViT", "Tau": "N/A", "Clean_Acc": round(vit_clean, 2), "Occ2_Acc": round(vit_occ, 2),
         "MLP_Skip_Pct": 0.0, "MFLOPs": round(base_mflops, 2), "FLOPs_Saved_Pct": 0.0, "Throughput_img_s": round(vit_tput, 1)},
        {"Model": "Static HAViT", "Tau": "N/A", "Clean_Acc": round(havit_clean, 2), "Occ2_Acc": round(havit_occ, 2),
         "MLP_Skip_Pct": 0.0, "MFLOPs": round(base_mflops, 2), "FLOPs_Saved_Pct": 0.0, "Throughput_img_s": round(havit_tput, 1)},
    ]

    thresholds = [1.00, 0.55, 0.45, 0.35, 0.25]
    for tau in thresholds:
        c_acc, skip_pct, mflops = evaluate_pattu_with_token_halting(pattu_model, test_loader, device, tau=tau, occlusion_ratio=0.0)
        o_acc, _, _ = evaluate_pattu_with_token_halting(pattu_model, test_loader, device, tau=tau, occlusion_ratio=0.40)
        tput = measure_throughput(pattu_model, device, tau=tau)
        saved_pct = 100.0 * (base_mflops - mflops) / base_mflops
        label = "PattuNet v3 (Full)" if tau >= 1.0 else f"PattuNet v3 (tau={tau:.2f})"
        rows.append({
            "Model": label,
            "Tau": str(tau),
            "Clean_Acc": round(c_acc, 2),
            "Occ2_Acc": round(o_acc, 2),
            "MLP_Skip_Pct": round(skip_pct, 2),
            "MFLOPs": round(mflops, 2),
            "FLOPs_Saved_Pct": round(saved_pct, 2),
            "Throughput_img_s": round(tput, 1),
        })

    print("\n" + "=" * 106)
    print(f"{'Model / Operating Point':<24} | {'Clean Acc':<10} | {'Occ-2 Acc':<10} | {'MLP Skip%':<10} | {'MFLOPs':<8} | {'Saved%':<8} | {'Img/sec':<10}")
    print("=" * 106)
    for r in rows:
        print(f"{r['Model']:<24} | {r['Clean_Acc']:>8.2f}% | {r['Occ2_Acc']:>8.2f}% | {r['MLP_Skip_Pct']:>8.2f}% | {r['MFLOPs']:>8.2f} | {r['FLOPs_Saved_Pct']:>6.2f}% | {r['Throughput_img_s']:>10.1f}")
    print("=" * 106)

    csv_path = os.path.join(out_dir, "efficiency_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(rows)).keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot Pareto Frontier: MFLOPs vs Clean & Occlusion Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    pattu_rows = [r for r in rows if "PattuNet" in r["Model"]]
    p_mflops = [r["MFLOPs"] for r in pattu_rows]
    p_clean = [r["Clean_Acc"] for r in pattu_rows]
    p_occ = [r["Occ2_Acc"] for r in pattu_rows]

    ax1.plot(p_mflops, p_clean, marker="o", linewidth=2.5, color="#1f77b4", label="PattuNet v3 (Dynamic Halting)")
    ax1.scatter([base_mflops], [vit_clean], color="#d62728", s=110, marker="s", zorder=5, label=f"Standard ViT ({vit_clean:.2f}%)")
    ax1.scatter([base_mflops], [havit_clean], color="#2ca02c", s=110, marker="^", zorder=5, label=f"Static HAViT ({havit_clean:.2f}%)")
    for r in pattu_rows:
        ax1.annotate(f"τ={r['Tau']}", (r["MFLOPs"], r["Clean_Acc"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax1.set_xlabel("Compute Cost per Image (MFLOPs) ↓")
    ax1.set_ylabel("Clean Top-1 Accuracy (%) ↑")
    ax1.set_title("Accuracy–Compute Pareto Frontier (Clean CIFAR-10)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(p_mflops, p_occ, marker="o", linewidth=2.5, color="#9467bd", label="PattuNet v3 (Dynamic Halting)")
    ax2.scatter([base_mflops], [vit_occ], color="#d62728", s=110, marker="s", zorder=5, label=f"Standard ViT ({vit_occ:.2f}%)")
    ax2.scatter([base_mflops], [havit_occ], color="#2ca02c", s=110, marker="^", zorder=5, label=f"Static HAViT ({havit_occ:.2f}%)")
    for r in pattu_rows:
        ax2.annotate(f"τ={r['Tau']}", (r["MFLOPs"], r["Occ2_Acc"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax2.set_xlabel("Compute Cost per Image (MFLOPs) ↓")
    ax2.set_ylabel("Patch Occlusion (Sev 2) Accuracy (%) ↑")
    ax2.set_title("Robustness–Compute Pareto Frontier (Occluded CIFAR-10)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "claim_a_efficiency_pareto.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\nSaved Efficiency CSV to: {csv_path}")
    print(f"Saved Pareto Figures to: {fig_path}")