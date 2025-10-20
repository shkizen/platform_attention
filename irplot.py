# -*- coding: utf-8 -*-
"""
irplot.py
Refactored to use relative paths within the platform_attention repository.
Assumes current working directory == .../platform_attention when executed.

Two IR pipelines are reproduced:
  A) epsilon-greedy (Calvano-style)
  B) softmax (Boltzmann)

Outputs:
  - results/tables/bm_prices_by_n.tex
  - results/irplots/ir_n{n}_K1_Calvano.png and .json
  - results/irplots/ir_n{n}_K1_softmax.png and .json
  - results/irplots/ir_summary.json (index of all IR artifacts)
"""

import os, math, json
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import Tuple, List, Dict, Any

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.getcwd()  # assumed .../platform_attention
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IR_DIR = os.path.join(RESULTS_DIR, "irplots")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(IR_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


# ============================================================
# Shared building blocks
# ============================================================

@dataclass
class SimParams:
    n: int = 2
    c: float = 1.0
    a: float = 2.0       # inside-good utility intercept (symmetric)
    a0: float = 0.0      # outside option utility
    mu: float = 0.25     # logit scale (price sensitivity) > 0
    m: int = 15          # number of price grid points
    xi: float = 0.10     # grid extension ratio around [p^B, p^M]
    K: int = 1           # memory length (baseline uses K=1)
    seed: int = 42

@dataclass
class IRParams:
    horizon_after: int = 15
    deviator_index: int = 0

def symmetric_share(p: float, a: float, a0: float, mu: float, n: int) -> float:
    """Symmetric MNL share of a single inside good when all set price p."""
    x = mp.e**((a - p)/mu)
    return x / (mp.e**(a0/mu) + n*x)

def solve_pB_pM(a: float, a0: float, mu: float, c: float, n: int) -> Tuple[float,float]:
    """Solve symmetric Bertrand and joint-monopoly prices in continuous space."""
    fB = lambda p: p - (c + mu/(1 - symmetric_share(p, a, a0, mu, n)))
    fM = lambda p: p - (c + mu/(1 - n*symmetric_share(p, a, a0, mu, n)))
    # Robust multi-starts
    for gB in [c+mu*0.2, c+mu*0.6, c+mu*1.0, c+mu*1.4]:
        try:
            pB = mp.findroot(fB, gB)
            break
        except: 
            continue
    else:
        raise RuntimeError("Failed to solve p^B")
    for gM in [float(pB)+0.2, float(pB)+0.6, float(pB)+1.0, float(pB)+1.4]:
        try:
            pM = mp.findroot(fM, gM)
            break
        except:
            continue
    else:
        raise RuntimeError("Failed to solve p^M")
    return float(pB), float(pM)

def logit_shares_vector(p_vec: np.ndarray, a: float, a0: float, mu: float) -> np.ndarray:
    """Standard MNL with outside option (no attention)."""
    v = (a - np.asarray(p_vec))/mu
    num = np.exp(v - v.max())
    den = np.exp(a0/mu - v.max()) + num.sum()
    return num / den

class LogitEnv:
    """Symmetric sellers, MNL demand, no shocks, K=1 (joint profile as state)."""
    def __init__(self, prm: SimParams, price_grid: np.ndarray, seed: int = 0):
        assert prm.K == 1, "This IR code assumes K=1 (joint-profile state)."
        self.prm = prm
        self.A = np.array(price_grid, dtype=float)  # actions -> prices
        self.G = len(self.A)
        self.n = prm.n
        self.J = self.G**self.n
        self.powG = np.array([self.G**i for i in range(self.n)], dtype=int)
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # start from middle joint action
        mid = self.G//2
        self.state = int(np.dot(np.full(self.n, mid, dtype=int), self.powG))
        return self.state

    def _profile_index_from_actions(self, acts):
        return int(np.dot(np.asarray(acts, dtype=int), self.powG))

    def profits(self, p_vec: np.ndarray) -> np.ndarray:
        s = logit_shares_vector(p_vec, self.prm.a, self.prm.a0, self.prm.mu)
        return (p_vec - self.prm.c) * s

    def step(self, acts):
        acts = list(map(int, acts))
        p_vec = self.A[np.array(acts, dtype=int)]
        prof = self.profits(p_vec)
        self.state = self._profile_index_from_actions(acts)
        info = {"p_vec": p_vec.copy()}
        return self.state, prof, info

def static_best_response(env: LogitEnv, prof_idx: int, i: int) -> float:
    """Static BR of firm i to rivals' prices in profile 'prof_idx'."""
    acts = np.zeros(env.n, dtype=int)
    x = prof_idx
    for k in range(env.n):
        acts[k] = x % env.G
        x //= env.G
    best_p, best_v = None, -1e300
    for a in range(env.G):
        acts_i = acts.copy()
        acts_i[i] = a
        p_vec = env.A[acts_i]
        v = env.profits(p_vec)[i]
        if (v > best_v) or (np.isclose(v, best_v) and (best_p is not None) and (env.A[a] < best_p)):
            best_v = v; best_p = env.A[a]
    return float(best_p)

def static_monopoly_price(env: LogitEnv) -> float:
    """Joint monopoly on grid: argmax over all joint profiles."""
    best_v, best_p = -1e300, None
    for idx in range(env.J):
        acts = np.zeros(env.n, dtype=int)
        x = idx
        for k in range(env.n):
            acts[k] = x % env.G
            x //= env.G
        p_vec = env.A[acts]
        v = env.profits(p_vec).sum()
        if v > best_v:
            best_v, best_p = v, float(p_vec.mean())
    return best_p

def static_nash_price(env: LogitEnv) -> float:
    """Pure NE on grid if exists; else return least-squares fixed point."""
    best_err, best_idx = 1e300, None
    for idx in range(env.J):
        acts = np.zeros(env.n, dtype=int)
        x = idx
        for k in range(env.n):
            acts[k] = x % env.G
            x //= env.G
        ok, err = True, 0.0
        for i in range(env.n):
            br = static_best_response(env, idx, i)
            err += (br - env.A[acts[i]])**2
            if not np.isclose(br, env.A[acts[i]]):
                ok = False
        if ok:
            return float(env.A[acts].mean())
        if err < best_err:
            best_err, best_idx = err, idx
    acts = np.zeros(env.n, dtype=int)
    x = best_idx
    for k in range(env.n):
        acts[k] = x % env.G
        x //= env.G
    return float(env.A[acts].mean())

def build_price_grid_from_pbpm(pB: float, pM: float, m: int, xi: float) -> np.ndarray:
    lo = pB - xi*(pM - pB)
    hi = pM + xi*(pM - pB)
    return np.linspace(lo, hi, m)


# ============================================================
# Variant A: epsilon-greedy (Calvano-style)
# ============================================================
@dataclass
class AgentParamsEG:
    n_actions: int
    alpha: float = 0.05
    gamma_disc: float = 0.95
    eps_beta: float = 2e-5   # epsilon_t = exp(-beta * t)
    eps_min: float = 0.005

@dataclass
class TrainParamsEG:
    T_max: int = 400_000
    warmup_greedy: int = 1000
    sessions: int = 24
    seed: int = 42

class QLearnerEG:
    def __init__(self, ap: AgentParamsEG, n_states: int, rng):
        self.ap = ap
        self.rng = rng
        self.Q = np.zeros((n_states, ap.n_actions), dtype=float)

    def select_action(self, s, t):
        eps = max(self.ap.eps_min, math.exp(-self.ap.eps_beta * t))
        if self.rng.random() < eps:
            return int(self.rng.integers(self.ap.n_actions))
        return int(np.argmax(self.Q[s]))

    def act_greedy(self, s):
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next):
        q = self.Q[s]; qn = self.Q[s_next]
        td = r + self.ap.gamma_disc * np.max(qn)
        q[a] = (1 - self.ap.alpha) * q[a] + self.ap.alpha * td

def train_and_ir_eg(sim: SimParams, T: TrainParamsEG, ir: IRParams, price_grid: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    env = LogitEnv(sim, price_grid, seed=seed)
    n_states = env.J
    ap = AgentParamsEG(n_actions=len(price_grid))
    agents = [QLearnerEG(ap, n_states, rng) for _ in range(sim.n)]

    s = env.reset(seed + 123)
    for t in range(T.T_max):
        acts = [ag.select_action(s, t) for ag in agents]
        s_next, profits, _ = env.step(acts)
        for i, ag in enumerate(agents):
            ag.update(s, acts[i], profits[i], s_next)
        s = s_next

    s = env.reset(seed + 234)
    for _ in range(T.warmup_greedy):
        acts = [ag.act_greedy(s) for ag in agents]
        s, _, _ = env.step(acts)

    devi = ir.deviator_index
    others = [j for j in range(sim.n) if j != devi]

    acts0 = [ag.act_greedy(s) for ag in agents]
    s1, _, info0 = env.step(acts0)
    p0 = info0["p_vec"].astype(float)
    dev_series = [p0[devi]]
    riv_series = [float(np.mean(p0[others]))]

    prof0 = env._profile_index_from_actions(acts0)
    p_dev_br = static_best_response(env, prof0, devi)
    dev_idx = int(np.abs(env.A - p_dev_br).argmin())
    acts1 = [ag.act_greedy(s1) for ag in agents]
    acts1[devi] = dev_idx
    s2, _, info1 = env.step(acts1)
    p1 = info1["p_vec"].astype(float)
    dev_series.append(p1[devi]); riv_series.append(float(np.mean(p1[others])))

    s = s2
    for _ in range(ir.horizon_after - 1):
        acts = [ag.act_greedy(s) for ag in agents]
        s, _, info = env.step(acts)
        p = info["p_vec"].astype(float)
        dev_series.append(p[devi]); riv_series.append(float(np.mean(p[others])))

    p_ne  = static_nash_price(env)
    p_mon = static_monopoly_price(env)
    p_lr  = float((dev_series[-1] + riv_series[-1]) / 2.0)
    return np.array(dev_series), np.array(riv_series), p_ne, p_mon, p_lr


# ============================================================
# Variant B: softmax (Boltzmann)
# ============================================================
@dataclass
class AgentParamsSM:
    n_actions: int
    alpha: float = 0.05
    gamma_disc: float = 0.95
    tau0: float = 1.0
    tau_min: float = 0.001
    tau_gamma: float = 0.99997

@dataclass
class TrainParamsSM:
    T_max: int = 120_000
    warmup_greedy: int = 1000
    sessions: int = 24
    seed: int = 42

class QLearnerSM:
    def __init__(self, ap: AgentParamsSM, n_states: int, rng):
        self.ap = ap
        self.rng = rng
        self.Q = np.zeros((n_states, ap.n_actions), dtype=float)
        self.tau = ap.tau0

    def _softmax_sample(self, q: np.ndarray) -> int:
        z = q / max(self.tau, 1e-8)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        return int(self.rng.choice(len(q), p=p))

    def select_action(self, s, t):
        a = self._softmax_sample(self.Q[s])
        self.tau = max(self.ap.tau_min, self.tau * self.ap.tau_gamma)
        return a

    def act_greedy(self, s):
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next):
        q = self.Q[s]; qn = self.Q[s_next]
        td = r + self.ap.gamma_disc * np.max(qn)
        q[a] = (1 - self.ap.alpha) * q[a] + self.ap.alpha * td

def train_and_ir_sm(sim: SimParams, T: TrainParamsSM, ir: IRParams, price_grid: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    env = LogitEnv(sim, price_grid, seed=seed)
    n_states = env.J
    ap = AgentParamsSM(n_actions=len(price_grid))
    agents = [QLearnerSM(ap, n_states, rng) for _ in range(sim.n)]

    s = env.reset(seed + 123)
    for t in range(T.T_max):
        acts = [ag.select_action(s, t) for ag in agents]
        s_next, profits, _ = env.step(acts)
        for i, ag in enumerate(agents):
            ag.update(s, acts[i], profits[i], s_next)
        s = s_next

    s = env.reset(seed + 234)
    for _ in range(T.warmup_greedy):
        acts = [ag.act_greedy(s) for ag in agents]
        s, _, _ = env.step(acts)

    devi = ir.deviator_index
    others = [j for j in range(sim.n) if j != devi]

    acts_greedy = [ag.act_greedy(s) for ag in agents]
    p_g = env.A[np.array(acts_greedy, dtype=int)]
    p_sym = float(np.mean(p_g))
    a_sym = int(np.abs(env.A - p_sym).argmin())
    acts0 = [a_sym for _ in range(sim.n)]
    s1, _, info0 = env.step(acts0)
    p0 = info0["p_vec"].astype(float)
    dev_series = [p0[devi]]; riv_series = [float(np.mean(p0[others]))]

    prof0_idx = env._profile_index_from_actions(acts0)
    p_dev_br = static_best_response(env, prof0_idx, devi)
    dev_idx = int(np.abs(env.A - p_dev_br).argmin())
    acts1 = acts0.copy(); acts1[devi] = dev_idx
    s2, _, info1 = env.step(acts1)
    p1 = info1["p_vec"].astype(float)
    dev_series.append(p1[devi]); riv_series.append(float(np.mean(p1[others])))

    s = s2
    for _ in range(ir.horizon_after - 1):
        acts = [ag.act_greedy(s) for ag in agents]
        s, _, info = env.step(acts)
        p = info["p_vec"].astype(float)
        dev_series.append(p[devi]); riv_series.append(float(np.mean(p[others])))

    p_ne  = static_nash_price(env)
    p_mon = static_monopoly_price(env)
    p_lr  = float((dev_series[-1] + riv_series[-1]) / 2.0)
    return np.array(dev_series), np.array(riv_series), p_ne, p_mon, p_lr


# ============================================================
# Utilities
# ============================================================

def write_bm_table(tex_path: str, rows):
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[!htbp]\centering
\caption{Bertrand and Joint--Monopoly Prices by Number of Firms (Symmetric Logit)}
\label{tab:bm_prices_by_n}
\begin{tabular}{@{} S[table-format=1.0] S[table-format=1.3] S[table-format=1.3] @{}}
\toprule
{Number of Firms} & {$p^{B}$} & {$p^{M}$} \\
\midrule
""")
        for n, pB, pM in rows:
            f.write(f"{n} & {pB:.3f} & {pM:.3f} \\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")


def export_ir_json(path_json: str, meta: Dict[str, Any]):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_ir_block(tag_suffix: str, variant: str, trainer, train_and_ir_func):
    BASE = SimParams(a=2.0, a0=0.0, mu=0.25, c=1.0, K=1, m=15, xi=0.10, seed=42)
    IRp  = IRParams(horizon_after=15, deviator_index=0)

    # 1) Table (continuous pB/pM)
    n_list = [2, 3, 4, 5]
    rows = [(n, *solve_pB_pM(BASE.a, BASE.a0, BASE.mu, BASE.c, n)) for n in n_list]
    tex_path = os.path.join(TAB_DIR, "bm_prices_by_n.tex")
    write_bm_table(tex_path, rows)
    print(f"[OK] LaTeX table saved: {tex_path}")

    # 2) IR plots + JSON (default n=2; change here to add more n values)
    json_index: List[Dict[str, Any]] = []
    for n in [2]:
        pB, pM = solve_pB_pM(BASE.a, BASE.a0, BASE.mu, BASE.c, n)
        grid = build_price_grid_from_pbpm(pB, pM, m=BASE.m, xi=BASE.xi)

        dev_paths, riv_paths = [], []
        pNEs, pMONs, pLRs = [], [], []
        seeds = [trainer.seed + 10*s for s in range(trainer.sessions)]
        for sd in seeds:
            sim = replace(BASE, n=n)  # K=1 fixed
            dev, riv, p_ne, p_mon, p_lr = train_and_ir_func(sim, trainer, IRp, grid, sd)
            dev_paths.append(dev); riv_paths.append(riv)
            pNEs.append(p_ne);     pMONs.append(p_mon); pLRs.append(p_lr)

        dev_stack = np.stack(dev_paths)
        riv_stack = np.stack(riv_paths)
        dev_mean, riv_mean = dev_stack.mean(axis=0), riv_stack.mean(axis=0)
        dev_std,  riv_std  = dev_stack.std(axis=0, ddof=0), riv_stack.std(axis=0, ddof=0)
        dev_var,  riv_var  = dev_stack.var(axis=0, ddof=0), riv_stack.var(axis=0, ddof=0)
        p_ne = float(np.mean(pNEs)); p_mon = float(np.mean(pMONs)); p_lr = float(np.mean(pLRs))

        # Plot
        t = np.arange(len(dev_mean))
        plt.figure(figsize=(7, 3.6), dpi=150)
        plt.plot(t, dev_mean, marker='o', linewidth=2, label="Deviator (mean)")
        plt.plot(t, riv_mean, marker='^', linewidth=2, label=("Rivals avg (mean)" if n>2 else "Rival (mean)"))
        plt.hlines(p_ne,  xmin=t[0], xmax=t[-1], linestyles="dotted",  label="Nash price")
        plt.hlines(p_mon, xmin=t[0], xmax=t[-1], linestyles="dashdot", label="Monopoly price")
        plt.hlines(p_lr,  xmin=t[0], xmax=t[-1], linewidth=1.0, label="Long-run price")
        plt.title(f"Impulse response (n={n}, K=1, {variant})")
        plt.xlabel(r"$\tau$"); plt.ylabel("Price")
        plt.xlim(t[0], t[-1]); plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        out_png  = os.path.join(IR_DIR, f"ir_n{n}_K1_{tag_suffix}.png")
        out_json = os.path.join(IR_DIR, f"ir_n{n}_K1_{tag_suffix}.json")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] IR plot saved: {out_png}")

        # JSON export with mean/std/var
        meta = {
            "variant": variant,
            "tag": tag_suffix,
            "n": n,
            "K": 1,
            "t": t.tolist(),
            "deviator_mean": dev_mean.tolist(),
            "deviator_std":  dev_std.tolist(),
            "deviator_var":  dev_var.tolist(),
            "rivals_mean":   riv_mean.tolist(),
            "rivals_std":    riv_std.tolist(),
            "rivals_var":    riv_var.tolist(),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "p_nash_mean": p_ne,
            "p_monopoly_mean": p_mon,
            "p_longrun_mean": p_lr,
            "file_png": os.path.relpath(out_png, BASE_DIR)
        }
        export_ir_json(out_json, meta)
        json_index.append({
            "variant": variant,
            "n": n,
            "json": os.path.relpath(out_json, BASE_DIR),
            "png": os.path.relpath(out_png, BASE_DIR)
        })

    # 3) Save summary index for this variant (append to the global one later in main)
    return json_index


# ============================================================
# Main
# ============================================================
def main():
    # A) epsilon-greedy block
    TR_EG = TrainParamsEG(T_max=400_000, warmup_greedy=1000, sessions=24, seed=42)
    index_eg = run_ir_block("Calvano", "epsilon-greedy", TR_EG, train_and_ir_eg)

    # B) softmax block
    TR_SM = TrainParamsSM(T_max=120_000, warmup_greedy=1000, sessions=24, seed=42)
    index_sm = run_ir_block("softmax", "softmax", TR_SM, train_and_ir_sm)

    # C) Write a global summary index
    summary_path = os.path.join(IR_DIR, "ir_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(index_eg + index_sm, f, indent=2)

    print("[DONE] Results saved under:", RESULTS_DIR)
    print("       IR index:", os.path.relpath(summary_path, BASE_DIR))


if __name__ == "__main__":
    main()
