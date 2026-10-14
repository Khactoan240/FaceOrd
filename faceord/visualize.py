"""
Qualitative figures of a FaceOrd outer-fold checkpoint: Grad-CAM (optionally next
to a comparison model), t-SNE of the fused features, failure cases and the
regional self-attention.

    python -m faceord.visualize --ckpt results/faceord/fold_1/best.pt \
        --compare-ckpt results/vgg19/fold_1/best.pt --compare-name VGG19 \
        --data data/UTKFace --out-dir results/figures --mode all

Samples are drawn from the checkpoint's held-out test fold.
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_checkpoint
from .dataset import UTKFaceDataset, GROUP_LABELS
from .models import FaceOrd

REGION_NAMES = ["L.Eye+Brow", "R.Eye+Brow", "Nose", "Mouth", "L.Cheek/Jaw", "R.Cheek/Jaw"]


def _unnorm(t: torch.Tensor) -> np.ndarray:
    """ImageNet-normalised (3, H, W) tensor -> (H, W, 3) uint8 RGB image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = t.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _gradcam_target(model):
    """Last stage of the FaceOrd global encoder, or the last convolutional block of a comparison model."""
    if isinstance(model, FaceOrd):
        return list(model.global_encoder.features.children())[-1]
    feats = getattr(model, "features", None)
    if feats is not None:   # VGG: the ReLU right after the last convolution (14x14)
        last_conv = max(i for i, m in enumerate(feats) if isinstance(m, torch.nn.Conv2d))
        return feats[last_conv + 1]
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1]


class GradCAM:
    """Grad-CAM for an age-group score; gradients are captured with a tensor hook."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        _gradcam_target(model).register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def __call__(self, global_img: torch.Tensor, regions: torch.Tensor,
                 target_class: int) -> np.ndarray:
        """Normalised 224x224 heat map of `target_class` for a single image."""
        self.model.zero_grad()
        if isinstance(self.model, FaceOrd):
            score = self.model(global_img, regions)["age_p_hat"][0, target_class]
        else:
            score = self.model(global_img)["age_logits"][0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)          # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (224, 224))


def _predict_age(model, img, regs) -> int:
    with torch.no_grad():
        out = model(img, regs) if isinstance(model, FaceOrd) else model(img)
    key = "age_p_hat" if "age_p_hat" in out else "age_logits"
    return int(out[key][0].argmax())


def _overlay(img_np, cam):
    heatmap = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8),
                                             cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)


def run_gradcam(model, dataset, indices, device, out_dir, n_samples=12,
                compare_model=None, compare_name="VGG19"):
    """
    Grad-CAM for the true age group. With `compare_model`, the comparison model
    (top row) and FaceOrd (bottom row) are shown on the same six images.
    """
    print("[Grad-CAM] generating...")
    models  = ([(compare_name, compare_model)] if compare_model is not None else []) \
              + [("FaceOrd", model)]
    cams    = [GradCAM(m) for _, m in models]
    n_cols  = 6 if compare_model is not None else n_samples
    chosen  = random.sample(list(indices), n_cols)

    if compare_model is None:
        fig, axes = plt.subplots(3, n_cols // 3, figsize=(n_cols * 1.6, 7))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(len(models), n_cols, figsize=(n_cols * 1.9, 4.2))

    for j, idx in enumerate(chosen):
        sample = dataset[idx]
        img    = sample["global_img"].unsqueeze(0).to(device)
        regs   = sample["regions"].unsqueeze(0).to(device)
        age_t  = sample["age_group"].item()
        img_np = _unnorm(sample["global_img"])
        for r, ((name, m), cam_gen) in enumerate(zip(models, cams)):
            ax = axes[j] if compare_model is None else axes[r, j]
            age_p = _predict_age(m, img, regs)
            ax.imshow(_overlay(img_np, cam_gen(img, regs, age_t)))
            ax.set_title(f"{name}  T:{GROUP_LABELS[age_t]} P:{GROUP_LABELS[age_p]}",
                         fontsize=6.5, color="green" if age_t == age_p else "red")
            ax.axis("off")

    plt.suptitle("Grad-CAM: age group (green=correct, red=wrong)", fontsize=10)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "gradcam_age.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {fpath}")


def run_tsne(model, dataset, indices, device, out_dir, n_samples=2000, workers=0):
    """t-SNE of the fused feature f, coloured by age group and by gender."""
    print("[t-SNE] extracting features...")
    chosen = random.sample(list(indices), min(n_samples, len(indices)))
    loader = DataLoader(Subset(dataset, chosen), batch_size=64,
                        shuffle=False, num_workers=workers)

    feats, age_labels, gender_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["global_img"].to(device), batch["regions"].to(device))
            feats.append(out["fused_feat"].cpu().numpy())
            age_labels.extend(batch["age_group"].numpy())
            gender_labels.extend(batch["gender"].numpy())

    feats         = np.vstack(feats)
    age_labels    = np.array(age_labels)
    gender_labels = np.array(gender_labels)

    print("  running t-SNE (perplexity=40)...")
    emb = TSNE(n_components=2, perplexity=40, max_iter=1000,
               random_state=42, n_jobs=-1).fit_transform(feats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap_age = matplotlib.colormaps["tab10"].resampled(8)
    for g in range(8):
        mask = age_labels == g
        ax1.scatter(emb[mask, 0], emb[mask, 1], c=[cmap_age(g)],
                    label=GROUP_LABELS[g], s=8, alpha=0.6)
    ax1.set_title("t-SNE by Age Group", fontsize=11)
    ax1.legend(markerscale=2, fontsize=7, loc="best")
    ax1.axis("off")

    colors_g = ["steelblue", "tomato"]
    for g, name in enumerate(["Male", "Female"]):
        mask = gender_labels == g
        ax2.scatter(emb[mask, 0], emb[mask, 1], c=colors_g[g],
                    label=name, s=8, alpha=0.6)
    ax2.set_title("t-SNE by Gender", fontsize=11)
    ax2.legend(markerscale=2, fontsize=8)
    ax2.axis("off")

    plt.suptitle(f"t-SNE of FaceOrd fused features (n={len(feats)})", fontsize=12)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "tsne_features.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {fpath}")


def run_failures(model, dataset, indices, device, out_dir, n_show=16):
    """Grid of misclassified test images, sorted by group distance and confidence."""
    print("[Failures] finding failure cases...")
    order = list(indices)
    random.shuffle(order)

    failures = []
    with torch.no_grad():
        for idx in order:
            if len(failures) >= n_show * 3:
                break
            sample = dataset[idx]
            img    = sample["global_img"].unsqueeze(0).to(device)
            regs   = sample["regions"].unsqueeze(0).to(device)
            age_t  = sample["age_group"].item()
            gen_t  = sample["gender"].item()

            out      = model(img, regs)
            age_pred = out["age_p_hat"][0].argmax().item()
            gen_pred = (out["gender_prob"][0] >= 0.5).long().item()
            age_conf = out["age_p_hat"][0, age_pred].item()
            error    = abs(age_pred - age_t)

            if age_pred != age_t:
                failures.append({
                    "img": _unnorm(sample["global_img"]),
                    "age_t": age_t, "age_p": age_pred,
                    "gen_t": gen_t, "gen_p": gen_pred,
                    "conf": age_conf, "error": error,
                })

    failures.sort(key=lambda x: (-x["error"], -x["conf"]))
    failures = failures[:n_show]

    cols = 4
    rows_n = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.5, rows_n * 3))
    axes = axes.flatten()

    gender_names = ["M", "F"]
    for i, f in enumerate(failures):
        ax = axes[i]
        ax.imshow(f["img"])
        gt_str = f"{GROUP_LABELS[f['age_t']]}({gender_names[f['gen_t']]})"
        pd_str = f"{GROUP_LABELS[f['age_p']]}({gender_names[f['gen_p']]})"
        err    = f['error']
        ax.set_title(
            f"GT:{gt_str}\nPred:{pd_str} Δ={err}\nconf={f['conf']:.2f}",
            fontsize=7, color="red" if err > 1 else "darkorange"
        )
        ax.axis("off")

    for ax in axes[len(failures):]:
        ax.axis("off")

    plt.suptitle("Failure Cases (age prediction errors, sorted by Δ group)", fontsize=10)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "failure_cases.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {fpath}")


class AttentionExtractor:
    """Forward hooks that store the head-averaged attention maps of every Transformer layer."""

    def __init__(self, model: FaceOrd):
        self.model  = model
        self._hooks = []
        self.attn_weights = {}  # layer index -> (B, M, M)

        for i, layer in enumerate(model.transformer.layers):
            self._hooks.append(layer.attn.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx):
        def hook(module, input, output):
            # nn.MultiheadAttention returns (output, weights averaged over heads)
            _, weights = output
            if weights is not None:
                self.attn_weights[idx] = weights.detach().cpu()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def run_attention(model, dataset, indices, device, out_dir, n_samples=200):
    """Mean region-to-region attention, overall and for four age groups, per layer."""
    print("[Attention] extracting attention weights...")
    extractor = AttentionExtractor(model)

    n_layers    = len(model.transformer.layers)
    all_weights = {i: [] for i in range(n_layers)}
    age_labels  = []
    chosen      = random.sample(list(indices), min(n_samples, len(indices)))

    with torch.no_grad():
        for idx in chosen:
            sample = dataset[idx]
            model(sample["global_img"].unsqueeze(0).to(device),
                  sample["regions"].unsqueeze(0).to(device))
            for layer_idx, w in extractor.attn_weights.items():
                all_weights[layer_idx].append(w[0].numpy())  # (M, M)
            age_labels.append(sample["age_group"].item())

    extractor.remove()

    M = model.M
    region_names = REGION_NAMES[:M]
    groups = [0, 3, 4, 7]

    fig, axes = plt.subplots(n_layers, 1 + len(groups), figsize=(18, 3.5 * n_layers), squeeze=False)

    def _show(ax, matrix, title):
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(M)); ax.set_xticklabels(region_names, rotation=45, fontsize=7)
        ax.set_yticks(range(M)); ax.set_yticklabels(region_names, fontsize=7)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for layer_idx in range(n_layers):
        ws = np.stack(all_weights[layer_idx])  # (N, M, M)
        _show(axes[layer_idx, 0], ws.mean(axis=0), f"Layer {layer_idx + 1}: all samples")
        for col, g in enumerate(groups, start=1):
            mask = np.array(age_labels) == g
            if mask.sum() < 3:
                axes[layer_idx, col].axis("off")
                continue
            _show(axes[layer_idx, col], ws[mask].mean(axis=0),
                  f"L{layer_idx + 1}: age {GROUP_LABELS[g]} (n={mask.sum()})")

    plt.suptitle("Transformer self-attention across facial regions", fontsize=12)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "attention_heatmap.pdf")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {fpath}")

    importance = np.stack(all_weights[0]).mean(axis=0).mean(axis=0)  # attention received per region
    print("\n  Mean attention received per region (layer 1):")
    for name, value in zip(region_names, importance):
        print(f"    {name}: {value:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="FaceOrd outer-fold checkpoint (best.pt)")
    parser.add_argument("--compare-ckpt", help="Comparison-model checkpoint of the same fold for Grad-CAM")
    parser.add_argument("--compare-name", default="VGG19")
    parser.add_argument("--data", help="UTKFace directory (default: DATA_ROOT of the checkpoint)")
    parser.add_argument("--landmark-cache", help="Override LANDMARK_CACHE of the checkpoint")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--mode", default="all",
                        choices=["all", "gradcam", "tsne", "failures", "attention"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt}...")
    model, cfg, payload = load_checkpoint(args.ckpt, device)
    if not isinstance(model, FaceOrd):
        raise ValueError("--ckpt must be a FaceOrd checkpoint")
    if args.landmark_cache:
        cfg.LANDMARK_CACHE = args.landmark_cache

    dataset = UTKFaceDataset(args.data or cfg.DATA_ROOT, train=False, cfg=cfg)
    indices = payload["metadata"].get("test_indices") or list(range(len(dataset)))
    print(f"  {len(indices)} held-out samples")

    if args.mode in ("all", "gradcam"):
        compare = load_checkpoint(args.compare_ckpt, device)[0] if args.compare_ckpt else None
        run_gradcam(model, dataset, indices, device, args.out_dir, compare_model=compare,
                    compare_name=args.compare_name)

    if args.mode in ("all", "tsne"):
        run_tsne(model, dataset, indices, device, args.out_dir, workers=cfg.NUM_WORKERS)

    if args.mode in ("all", "failures"):
        run_failures(model, dataset, indices, device, args.out_dir)

    if args.mode in ("all", "attention") and hasattr(model, "transformer"):
        run_attention(model, dataset, indices, device, args.out_dir)

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
