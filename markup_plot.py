# -*- coding: utf-8 -*-
"""
markup_plot.py
Refactored to use relative paths within the platform_attention repository.
Assumes current working directory == .../platform_attention when executed.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Core imports from your package (no external path edits needed) ---
from threshold import estimate_kappa_star
from eval import load_greedy_agents
from figures import plot_kappa_threshold
from env import SimParams
from agents import AgentParams
from train import TrainParams


# =============================================================================
# Helper paths
# =============================================================================
BASE_DIR = Path.cwd()
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
#  Function: run_single_threshold
# =============================================================================
def run_single_threshold(n: int, K: int, rho: float = 0.0, sigma: float = 0.0):
    """
    Run kappa sweep for given (n, K, rho, sigma), generate threshold plot.
    Saves outputs in results/figures/calvano_n{n}_K{K}[_shock]/.
    """
    tag_suffix = f"n{n}_K{K}"
    if sigma > 0 or rho > 0:
        tag_suffix += "_shock"

    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    sim = SimParams(
        n=n,
        K=K,
        price_grid=(1.424, 1.464, 1.505, 1.545, 1.586, 1.626, 1.667,
                    1.707, 1.747, 1.788, 1.828, 1.869, 1.909, 1.950, 1.990),
        kappa=np.inf,
        rho=rho,
        sigma=sigma,
        b=0.0,
        c=1.0,
        phi=1.0,
        observed_shocks=(rho > 0 or sigma > 0),
        seed=42,
    )

    ap = AgentParams(
        n_actions=len(sim.price_grid),
        alpha=0.05,
        delta=0.95,
        tau0=1.0,
        tau_min=0.01,
        gamma=0.99997,
    )

    kap_grid = [np.inf, 100, 50, 25, 10, 5, 1, 0]
    tr_sweep = TrainParams(T_max=150_000, replications=8, base_seed=99, n_jobs=-1)

    print(f">>> Estimating kappa* for n={n}, K={K}, rho={rho}, sigma={sigma}")
    df, kstar = estimate_kappa_star(
        out_dir=str(out_dir),
        sim_base=sim,
        ap=ap,
        kap_grid=kap_grid,
        tr=tr_sweep,
        delta=ap.delta,
        mu=sim.mu,
        tag_base=f"calvano_{tag_suffix}",
        tol_markup=0.05,
        n_jobs=-1,
    )

    print(df)
    print("Estimated kappa* =", kstar)

    print(">>> Plotting threshold curve ...")
    pdf, png = plot_kappa_threshold(
        df, kstar, out_dir=str(out_dir),
        tag=f"calvano_{tag_suffix}", tol_markup=0.05
    )
    print("Saved threshold plot:", pdf, png)


# =============================================================================
#  Function: combine_panels
# =============================================================================
def combine_panels(fig_name: str, subpaths: list[str], titles: list[str], suptitle: str):
    """Combine individual PNGs into a 2x2 grid and save under results/figures."""
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    fig_out = FIGURES_DIR / fig_name
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 8.2), dpi=300,
        gridspec_kw={"hspace": 0.2, "wspace": 0.12}
    )

    for ax, rel_path, ttl, lab in zip(axes.flat, subpaths, titles, panel_labels):
        full_path = FIGURES_DIR / rel_path
        if not full_path.exists():
            print("Warning: missing", full_path)
            continue
        img = plt.imread(full_path)
        ax.imshow(img)
        ax.set_title(ttl, pad=6)
        ax.axis("off")
        h, w = img.shape[0], img.shape[1]
        ax.add_patch(Rectangle((0, 0), w, h, fill=False,
                               linewidth=0.8, edgecolor=(0, 0, 0, 0.35),
                               transform=ax.transData))

    fig.suptitle(suptitle, fontsize=16, y=0.98)
    plt.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))

    png_file = fig_out.with_suffix(".png")
    pdf_file = fig_out.with_suffix(".pdf")
    plt.savefig(png_file, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"Saved combined figure to:\n- {png_file}\n- {pdf_file}")


# =============================================================================
#  Main routine
# =============================================================================
def main():
    # Run baseline (no-shock) simulations for n=2..5, K=1
    for n in [2, 3, 4, 5]:
        run_single_threshold(n, K=1)

    # Shocked versions (rho=0.8, sigma=0.25)
    for n in [2, 3, 4, 5]:
        run_single_threshold(n, K=1, rho=0.8, sigma=0.25)

    for n in [2, 3, 4, 5]:
        run_single_threshold(n, K=2, rho=0.8, sigma=0.25)

    # Combine panels for K=1 (no shock)
    combine_panels(
        "combined_1",
        [f"calvano_n{i}_K1/kappa_threshold.png" for i in [2, 3, 4, 5]],
        ["(a) n = 2", "(b) n = 3", "(c) n = 4", "(d) n = 5"],
        "Long-run Markup vs. Attention Intensity $\\kappa$"
    )

    # Combine panels for K=1 shock
    combine_panels(
        "combined_2",
        [f"calvano_n{i}_K1_shock/kappa_threshold.png" for i in [2, 3, 4, 5]],
        ["(a) n = 2", "(b) n = 3", "(c) n = 4", "(d) n = 5"],
        "Long-run Markup vs. Attention Intensity $\\kappa$ (Under Observable Demand Shock, K=1)"
    )

    # Combine panels for K=2 shock
    combine_panels(
        "combined_3",
        [f"calvano_n{i}_K2_shock/kappa_threshold.png" for i in [2, 3, 4, 5]],
        ["(a) n = 2", "(b) n = 3", "(c) n = 4", "(d) n = 5"],
        "Long-run Markup vs. Attention Intensity $\\kappa$ (Under Observable Demand Shock, K=2)"
    )


if __name__ == "__main__":
    main()
