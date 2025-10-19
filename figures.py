import numpy as np, matplotlib.pyplot as plt, os
from .eval import impulse_response_avg_once, static_monopoly_price, static_nash_price
from .io import ensure_dir, save_json

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_kappa_threshold(df, kstar, out_dir: str, tag: str, tol_markup: float = 0.05):
    """
    Plot mean markup (±1 SD) versus kappa (descending), highlight the threshold kappa*,
    and save a PDF + PNG under results/figures/{tag}/.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include columns: ["kappa", "markup_mean", "markup_sd"].
        Typically returned by estimate_kappa_star(...).
        Assumed already sorted descending by 'kappa' (inf first); if not, we sort.
    kstar : float
        Estimated threshold. Can be np.inf or 0.0 in corner cases.
    out_dir : str
        Root results directory.
    tag : str
        Label for subfolders/filenames (e.g., "n2_K1").
    tol_markup : float
        Decision threshold for “near-Bertrand” (default 0.05 == +5%).
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    # Sort descending by kappa (∞ → ... → 0)
    def _kap_sort_key(x):
        return np.inf if (isinstance(x, float) and np.isinf(x)) else float(x)
    df = df.sort_values("kappa", key=lambda s: [-_kap_sort_key(x) for x in s]).reset_index(drop=True)

    # X positions are categorical indices; xticklabels are the κ labels
    x = np.arange(len(df))
    kap_labels = []
    for val in df["kappa"].tolist():
        if isinstance(val, float) and np.isinf(val):
            kap_labels.append(r"$\infty$")
        else:
            # keep integers pretty; floats with no trailing zeros
            lab = f"{val:.0f}" if float(val).is_integer() else f"{val:g}"
            kap_labels.append(lab)

    y = df["markup_mean"].to_numpy(dtype=float)
    yerr = df["markup_sd"].to_numpy(dtype=float)

    # Figure paths
    fig_dir = os.path.join(out_dir, "figures", tag)
    ensure_dir(fig_dir)
    pdf_path = os.path.join(fig_dir, "kappa_threshold.pdf")
    png_path = os.path.join(fig_dir, "kappa_threshold.png")

    # Plot
    plt.figure(figsize=(6.6, 3.6), dpi=150)
    plt.errorbar(x, y, yerr=yerr, fmt="-o", linewidth=2, capsize=4, label="Mean markup ±1 SD")
    plt.axhline(tol_markup, linestyle="dashed", linewidth=1.5, color="gray", label=f"Tolerance = {tol_markup:.2f}")

    # Vertical marker for kappa*
    if np.isfinite(kstar):
        # Find nearest tick index to kstar for visual alignment
        # (if kstar equals a grid point, it will match exactly)
        # Otherwise, we draw at fractional x with a label in the legend.
        kappa_vals = df["kappa"].to_list()
        # convert inf to a large proxy for numeric compare
        kappa_numeric = [1e12 if (isinstance(k, float) and np.isinf(k)) else float(k) for k in kappa_vals]
        kstar_num = float(kstar)
        idx_near = int(np.argmin(np.abs(np.array(kappa_numeric) - kstar_num)))
        xstar = idx_near if (kappa_numeric[idx_near] == kstar_num) else float(idx_near)  # simple anchor
        plt.axvline(x=xstar, linestyle=":", color="tab:red", linewidth=1.5, label=r"$\kappa^{*}$")
    else:
        # If kstar is infinite, we’re in the “always collusive” corner; no vertical line needed.
        pass

    plt.xticks(x, kap_labels)
    plt.xlabel(r"Attention intensity $\kappa$ (descending)")
    plt.ylabel("Long-run markup (The Lerner index:(p-c)/p)")
    plt.ylim(bottom=0.0, top=max(0.30, y.max() + (yerr.max() if len(yerr) else 0.0) + 0.03))
    plt.legend(frameon=False, loc="best")
    plt.title("Threshold of Steering Intensity to Restore Competition")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    return pdf_path, png_path

def fig_impulse_response(env, greedy_agents, start_state, out_dir: str, tag: str,
                         horizon_after=15, reps=24, warmup_greedy=1000, deviator_index=0, base_seed=777, n_jobs=-1):
    ensure_dir(out_dir)
    series_path = os.path.join(out_dir, "figures", tag, "impulse_series.json")
    fig_path = os.path.join(out_dir, "figures", tag, "impulse_response.pdf")
    ensure_dir(os.path.dirname(fig_path))
    dev, riv, theta_bar = impulse_response_avg_once(
        env, greedy_agents, start_state,
        horizon_after=horizon_after, reps=reps, warmup_greedy=warmup_greedy,
        deviator_index=deviator_index, base_seed=base_seed, n_jobs=n_jobs,
        out_path=series_path
    )
    p_mon = static_monopoly_price(env, theta_bar)
    p_ne = static_nash_price(env, theta_bar)
    p_lr = float((dev[-1] + riv[-1]) / 2)

    t = np.arange(len(dev))
    plt.figure(figsize=(6.6, 3.3), dpi=150)
    plt.plot(t, dev, marker='o', linewidth=2, label="Deviating agent")
    plt.plot(t, riv, marker='^', linewidth=2, label="Nondeviating agent")
    plt.hlines(p_ne,  xmin=t[0], xmax=t[-1], linestyles="dotted",  label="Nash price")
    plt.hlines(p_mon, xmin=t[0], xmax=t[-1], linestyles="dashdot", label="Monopoly price")
    plt.hlines(p_lr,  xmin=t[0], xmax=t[-1], colors="gray", linewidth=1.0, label="Long-run price")
    plt.xlabel("Time"); plt.ylabel("Price"); plt.xlim(t[0], t[-1])
    plt.legend(frameon=False, ncol=2); plt.title("Impulse-response (Calvano-style)")
    ensure_dir(os.path.dirname(fig_path))
    plt.tight_layout(); plt.savefig(fig_path); plt.close()
    return fig_path, series_path
