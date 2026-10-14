"""Task-weight dynamics and tau sensitivity figure.

    python -m analysis.plot_training_dynamics --run-dir results/faceord \
        --tau-summary results/tau_sweep/tau_sweep_summary.json

  (a) sigma_age and sigma_gender per epoch, averaged over the outer folds
      (fold_*/history.json written by faceord.kfold)
  (b) inner-fold age AUC (mean +/- std) for every tau of faceord.tau_sweep
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sigma_curves(run_dir: Path) -> tuple:
    """Epochs and fold-averaged sigma_age / sigma_gender of an uncertainty-weighted run."""
    histories = [json.loads(path.read_text()) for path in sorted(run_dir.glob("fold_*/history.json"))]
    if not histories:
        raise FileNotFoundError(f"No fold_*/history.json in {run_dir}")
    n_ep = min(len(h) for h in histories)
    sig_a = np.array([[r["sigma_age"] for r in h[:n_ep]] for h in histories])
    sig_g = np.array([[r["sigma_gender"] for r in h[:n_ep]] for h in histories])
    return np.arange(1, n_ep + 1), sig_a.mean(0), sig_g.mean(0)


def tau_curve(summary_path: Path) -> tuple:
    """tau values with mean and std (in %) of the inner-fold age AUC across all outer folds."""
    results = json.loads(summary_path.read_text())
    scores = defaultdict(list)
    for result in results.values():
        for fold in result["folds"]:
            if fold["age"]["auc"] is not None:
                scores[float(fold["tau"])].append(fold["age"]["auc"])
    taus = sorted(scores)
    auc = np.array([np.mean(scores[t]) for t in taus]) * 100
    sd = np.array([np.std(scores[t], ddof=1) if len(scores[t]) > 1 else 0.0 for t in taus]) * 100
    return np.array(taus), auc, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="Output directory of faceord.kfold")
    ap.add_argument("--tau-summary", type=Path, required=True, help="tau_sweep_summary.json")
    ap.add_argument("--out", type=Path, default=Path("results/figures/training_dynamics"))
    args = ap.parse_args()

    ep, sa, sg = sigma_curves(args.run_dir)
    taus, auc, auc_sd = tau_curve(args.tau_summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.semilogy(ep, sa, label=r"$\sigma_{\mathrm{age}}$")
    ax1.semilogy(ep, sg, "--", label=r"$\sigma_{\mathrm{gender}}$")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$\sigma$ (log scale)")
    ax1.grid(alpha=0.3, ls="--")
    ax1.legend(fontsize=8)
    ax2.errorbar(taus, auc, yerr=auc_sd, marker="o", capsize=3)
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel("Age AUC (%)")
    ax2.grid(alpha=0.3, ls="--")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"  -> {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
