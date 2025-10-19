import numpy as np, pandas as pd
from typing import List, Tuple
from joblib import Parallel, delayed
from .env import SimParams, AttentionEnv
from .agents import AgentParams
from .train import TrainParams, Trainer
from .eval import load_greedy_agents, static_nash_price
from .io import ensure_dir

def _longrun_markup(env: AttentionEnv, greedy_agents, start_state,
                    burn: int = 2000, T: int = 6000):
    """Compute long-run markup using the average shock state."""
    s = start_state
    c = 1.0 
    prices, thetas = [], []
    g = lambda ag, st: ag.act_greedy(st)

    # burn-in
    for _ in range(burn):
        acts = [g(a, s) for a in greedy_agents]
        s, _, _ = env.step(acts)

    # evaluation horizon
    for _ in range(T):
        acts = [g(a, s) for a in greedy_agents]
        s, _, info = env.step(acts)
        prices.append(np.mean(info["p_vec"]))
        thetas.append(info["theta"])

    p_bar = float(np.mean(prices))
    theta_bar = float(np.mean(thetas))
    pB = static_nash_price(env, theta_bar)

    return (p_bar - c) / p_bar, theta_bar


def estimate_kappa_star(out_dir: str, sim_base: SimParams, ap: AgentParams,
                        kap_grid: List[float], tr: TrainParams,
                        delta: float, mu: float, tag_base: str,
                        tol_markup: float = 0.05, n_jobs: int = -1):
    """
    Robust κ* estimator:
    - Trains/reuses checkpoints per κ
    - Computes mean markup across seeds
    - Benchmarks against Nash price at θ̄
    - Scans from ∞ down to 0 to find the threshold where collusion breaks
    """

    ensure_dir(out_dir)

    # Sort κ descending (from ∞ down to smallest)
    kap_grid_sorted = sorted(kap_grid, key=lambda x: (np.inf if np.isinf(x) else x), reverse=True)
    results = []

    def _one_kappa(kappa):
        sim = SimParams(**{**sim_base.__dict__, "kappa": kappa})
        trainer = Trainer(sim, ap, out_dir=out_dir)
        tag = tag_base + f"_kap{'inf' if np.isinf(kappa) else str(kappa).replace('.','p')}"
        seeds = trainer.train(tr, tag)

        markups = []
        for sd in seeds:
            env, g_agents, s = load_greedy_agents(out_dir, tag, sd, sim)
            mk, _ = _longrun_markup(env, g_agents, s)
            markups.append(mk)
        mk_mean = float(np.mean(markups))
        mk_sd = float(np.std(markups))
        return dict(kappa=kappa, markup_mean=mk_mean, markup_sd=mk_sd, n_seeds=len(seeds))

    res = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one_kappa)(kap) for kap in kap_grid_sorted)
    df = pd.DataFrame(res).sort_values("kappa", ascending=False).reset_index(drop=True)

    # Identify threshold interval (collusive → competitive)
    above = df[df["markup_mean"] > tol_markup]
    below = df[df["markup_mean"] <= tol_markup]

    if not above.empty and not below.empty:
        k_hi = above["kappa"].iloc[0]           # highest κ still collusive
        k_lo = below["kappa"].iloc[-1]          # lowest κ that restores competition
        kstar = 0.5 * (k_hi + k_lo) if np.isfinite(k_hi) and np.isfinite(k_lo) else k_lo
    elif above.empty:
        kstar = 0.0
    else:
        kstar = np.inf

    df.attrs["kappa_star"] = kstar
    df.attrs["tol_markup"] = tol_markup
    return df, kstar
