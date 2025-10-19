#!/usr/bin/env python3
# generate_all.py
# One-shot generator for tables & volatility figures using the current codebase.

import os
import argparse
import numpy as np
import pandas as pd

from env import SimParams, AttentionEnv
from agents import AgentParams
from train import Trainer, TrainParams
from eval import load_greedy_agents, static_nash_price, static_monopoly_price
from io import ensure_dir, tag_from_params

# ---------- Common configuration ----------
OUT_DIR = "results"
TABLES_DIR = os.path.join(OUT_DIR, "tables")
ensure_dir(OUT_DIR)
ensure_dir(TABLES_DIR)

PRICE_GRID = (
    1.424,1.464,1.505,1.545,1.586,1.626,1.667,1.707,1.747,
    1.788,1.828,1.869,1.909,1.950,1.990
)

# Training hyperparameters (kept here for easy tweaks)
AP = AgentParams(
    n_actions=len(PRICE_GRID),
    alpha=0.05,
    delta=0.95,
    tau0=1.0,
    tau_min=0.01,
    gamma=0.99997,
)
TR = TrainParams(
    T_max=120_000,
    replications=8,   # reduce to 2-4 for quick smoke tests
    base_seed=42,
    n_jobs=-1
)

# ---------- Helpers ----------
def simulate_lr_means(env, greedy_agents, start_state, burn=2000, T=6000):
    """Simulate long-run path under greedy policies; return (mean Lerner, mean Price)."""
    s = start_state
    actg = lambda ag, st: ag.act_greedy(st)
    # burn into limit cycle / steady regime
    for _ in range(burn):
        acts = [actg(a, s) for a in greedy_agents]
        s, _, _ = env.step(acts)
    # record
    lerners, prices = [], []
    for _ in range(T):
        acts = [actg(a, s) for a in greedy_agents]
        s, _, info = env.step(acts)
        p = info["p_vec"].astype(float)
        prices.append(p.mean())
        lerners.append(((p - env.prm.c) / p).mean())
    return float(np.mean(lerners)), float(np.mean(prices))

def train_if_needed(sim: SimParams, tag: str):
    """Train policies for (sim, tag) combo if checkpoints are missing."""
    # Trainer handles idempotency (safe to call again)
    Trainer(sim, AP, out_dir=OUT_DIR).train(TR, tag)

# ---------- Table 2 ----------
def build_table2(ns=(2,3,4,5), K=1, kappa=0.0, mu=0.25, c=1.0, a=2.0, a0=0.0):
    rows = []
    for n in ns:
        sim = SimParams(n=n, K=K, kappa=kappa, mu=mu, price_grid=PRICE_GRID, c=c, a=a, a0=a0)
        tag = tag_from_params(n, K, PRICE_GRID, kappa=kappa, delta=AP.delta, mu=mu)
        train_if_needed(sim, tag)
        seeds = [TR.base_seed + 17*i for i in range(TR.replications)]
        Ls, Ps = [], []
        for sd in seeds:
            env, g_agents, s0 = load_greedy_agents(OUT_DIR, tag, sd, sim)
            Lbar, Pbar = simulate_lr_means(env, g_agents, s0)
            Ls.append(Lbar); Ps.append(Pbar)
        rows.append({
            "Number of Firms": n,
            "Mean Lerner Index": float(np.mean(Ls)),
            "Mean Price": float(np.mean(Ps)),
        })
    df = pd.DataFrame(rows).sort_values("Number of Firms").reset_index(drop=True)
    csv_path = os.path.join(TABLES_DIR, "table2_longrun_markup_meanprice.csv")
    tex_path = os.path.join(TABLES_DIR, "table2_longrun_markup_meanprice.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.6f",
                            caption="Long-run Markup and Implied Mean Price by Number of Firms",
                            label="tab:longrun_markup"))
    return df, csv_path, tex_path)

# ---------- Table 3 ----------
def build_table3(ns=(2,3,4,5), K=1, kappa=0.0, mu=0.25, c=1.0, a=2.0, a0=0.0, theta_eval=0.0):
    rows = []
    for n in ns:
        sim = SimParams(n=n, K=K, kappa=kappa, mu=mu, price_grid=PRICE_GRID, c=c, a=a, a0=a0)
        env = AttentionEnv(sim, seed=123)
        pB = static_nash_price(env, theta_eval)
        pM = static_monopoly_price(env, theta_eval)
        rows.append({"Number of Firms": float(n), "pB": float(pB), "pM": float(pM)})
    df = pd.DataFrame(rows).sort_values("Number of Firms").reset_index(drop=True)
    csv_path = os.path.join(TABLES_DIR, "table3_pB_pM_by_n.csv")
    tex_path = os.path.join(TABLES_DIR, "table3_pB_pM_by_n.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f",
                            caption="Bertrand and Joint–Monopoly Prices by Number of Firms",
                            label="tab:pB_pM"))
    return df, csv_path, tex_path

# ---------- Volatility (no shock vs shock) ----------
def build_volatility(n=2, K=1, sigma_shock=0.15, rho_shock=0.85, mu=0.25, c=1.0, a=2.0, a0=0.0):
    # train once at sigma=0
    sim_train = SimParams(n=n, K=K, kappa=0.0, mu=mu, price_grid=PRICE_GRID, c=c, a=a, a0=a0, sigma=0.0, rho=0.0)
    tag = f"voltest_n{n}_K{K}_sigma0"
    # slight override: use smaller reps to speed volatility demo
    Trainer(sim_train, AP, out_dir=OUT_DIR).train(TrainParams(T_max=120_000, replications=4, base_seed=2025, n_jobs=-1), tag)
    seeds = [2025 + 17*i for i in range(4)]

    def eval_var(sim_eval, tag, seeds, T=6000, burn=2000):
        vars_ = []
        for sd in seeds:
            env, g_agents, s = load_greedy_agents(OUT_DIR, tag, sd, sim_eval)
            actg = lambda ag, st: ag.act_greedy(st)
            for _ in range(burn):
                acts = [actg(a, s) for a in g_agents]
                s, _, _ = env.step(acts)
            profits, thetas = [], []
            for _ in range(T):
                acts = [actg(a, s) for a in g_agents]
                s, _, info = env.step(acts)
                p = info["p_vec"].astype(float)
                th = float(info["theta"])
                pi = env.static_profits(p, theta=th).mean()
                profits.append(pi); thetas.append(th)
            theta_bar = float(np.mean(thetas))
            pB = static_nash_price(env, theta_bar)
            piB = env.static_profits(np.full(env.n, pB), theta=theta_bar).mean()
            gain = (np.array(profits) - piB) / max(piB, 1e-12)
            vars_.append(np.var(gain))
        return float(np.mean(vars_)), float(np.std(vars_))

    # evaluate
    sim_base  = SimParams(**{**sim_train.__dict__})
    base_mean,  base_sd  = eval_var(sim_base, tag, seeds)
    sim_shock = SimParams(**{**sim_train.__dict__}, sigma=sigma_shock, rho=rho_shock)
    shock_mean, shock_sd = eval_var(sim_shock, tag, seeds)

    df = pd.DataFrame({
        "scenario": ["no_shock (σ=0, ρ=0)", f"shock (σ={sigma_shock}, ρ={rho_shock})"],
        "mean_var_profit_gain": [base_mean, shock_mean],
        "sd_over_seeds": [base_sd, shock_sd],
    })
    csv_path = os.path.join(TABLES_DIR, "profit_gain_volatility.csv")
    tex_path = os.path.join(TABLES_DIR, "profit_gain_volatility.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.5f",
                            caption="Profit-gain variance without and with shocks",
                            label="tab:volatility"))
    return df, csv_path, tex_path

# ---------- Optional: kappa sweep via memory_check.py ----------
def run_kappa_sweep(ns=(2,3,4,5), Ks=(1,2)):
    try:
        from memory_check import estimate_kappa_star, plot_kappa_threshold
        rows_all = []
        kap_grid = [np.inf, 100, 50, 25, 10, 5, 1, 0]
        for n in ns:
            for K in Ks:
                tag = f"n{n}_K{K}_shock_sweep"
                rows, kstar = estimate_kappa_star(
                    kap_grid=kap_grid,
                    n=n, K=K,
                    cushion_T=K,
                    episodes=150_000, seed=99,
                    sigma=0.25, rho=0.8, observed_shocks=True,
                    dyn_pdp=True
                )
                plot_kappa_threshold(rows, kstar, out_dir=OUT_DIR, tag=tag, tol_markup=0.05)
                rows_all.append((n, K, rows, kstar))
        return rows_all
    except Exception as e:
        print("[kappa sweep] skipped (memory_check.py not available or failed):", e)
        return None

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Generate replication outputs")
    p.add_argument("--table2", action="store_true", help="Build Table 2 (long-run markup & mean price by n)")
    p.add_argument("--table3", action="store_true", help="Build Table 3 (static pB & pM by n)")
    p.add_argument("--volatility", action="store_true", help="Build profit-gain volatility (no shock vs shock)")
    p.add_argument("--kappa-sweep", action="store_true", help="Run optional kappa* sweep plots")
    p.add_argument("--all", action="store_true", help="Run everything")
    args = p.parse_args()

    if not any([args.table2, args.table3, args.volatility, args.kappa_sweep, args.all]):
        p.print_help()
        return

    if args.all or args.table2:
        print(">> Building Table 2 …")
        df2, p2csv, p2tex = build_table2()
        print(df2, "\nSaved:", p2csv, p2tex)

    if args.all or args.table3:
        print(">> Building Table 3 …")
        df3, p3csv, p3tex = build_table3()
        print(df3, "\nSaved:", p3csv, p3tex)

    if args.all or args.volatility:
        print(">> Building Volatility comparison …")
        dfv, pvcsv, pvtex = build_volatility()
        print(dfv, "\nSaved:", pvcsv, pvtex)

    if args.all or args.kappa_sweep:
        print(">> Running kappa sweep …")
        _ = run_kappa_sweep()
        print("kappa sweep complete (plots saved under results/).")

if __name__ == "__main__":
    main()
