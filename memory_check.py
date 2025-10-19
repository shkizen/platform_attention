# sweep_kappa.py
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict
from collections import defaultdict

# ---------- 1) SimParams ----------
@dataclass(frozen=True)
class SimParams:
    n: int = 2
    c: float = 1.0
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    price_grid: Tuple[float, ...] = (
        1.424,1.464,1.505,1.545,1.586,1.626,1.667,1.707,1.747,
        1.788,1.828,1.869,1.909,1.950,1.990
    )
    K: int = 1
    seed: int = 42
    sigma: float = 0.0
    rho: float = 0.0
    observed_shocks: bool = False
    kappa: float = 0.0
    # Dynamic platform steering (Johnson-style)
    dyn_pdp: bool  = True
    cushion_tau: float = 0.05
    cushion_T:   int   = 1
    bonus_pct:   float = 0.25

# ---------- 2) Attention rules ----------
def _attn_weights_mean1(p_vec: np.ndarray, kappa: float) -> np.ndarray:
    p_vec = np.asarray(p_vec, float); n = p_vec.size
    if kappa == 0.0: return np.ones(n, float)
    if np.isinf(kappa):
        w = (p_vec == p_vec.min()).astype(float); w /= w.sum(); return n*w
    gaps = p_vec - p_vec.min()
    z = -kappa * gaps; z -= z.max()
    raw = np.exp(z)
    return n * (raw / raw.sum())

def _attn_weights_dynamic(p_vec, kappa, last_min_price, last_winners, tau, T, bonus_pct):
    base = _attn_weights_mean1(p_vec, kappa)
    if (last_min_price is None) or (not last_winners) or (T <= 0): return base
    recent = set(i for win in last_winners[-T:] for i in win)
    close = (p_vec <= (last_min_price + tau))
    bonus = np.ones_like(base)
    for i in range(len(base)):
        if (i in recent) and close[i]:
            bonus[i] *= (1.0 + float(bonus_pct))
    out = base * bonus
    return len(base) * (out / out.sum())

# ---------- 3) Logit shares ----------
def _logit_shares_with_attention(p_vec, a, a0, mu, w, theta=0.0):
    p_vec = np.asarray(p_vec, float); assert mu > 0
    v = (a + theta - p_vec) / mu; v0 = a0 / mu
    mx = max(v.max(), v0)
    num = w * np.exp(v - mx)
    den = np.exp(v0 - mx) + num.sum()
    return num / den

# ---------- 4) Environment ----------
class AttentionEnv:
    def __init__(self, prm: SimParams, seed: int = None):
        self.prm = prm
        self.A = np.array(prm.price_grid, float)
        self.n = int(prm.n)
        self.rng = np.random.default_rng(prm.seed if seed is None else seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mid = int(len(self.A)//2)
        init = (mid,)*self.n
        self.theta = 0.0
        self.hist = [init for _ in range(max(1, self.prm.K))]
        self.last_min_price = None
        self.last_winners = []
        return (tuple(self.hist), float(self.theta)) if self.prm.observed_shocks else tuple(self.hist)

    def step(self, acts: List[int]):
        acts = list(map(int, acts))
        p_vec = self.A[np.array(acts, int)]
        if self.prm.dyn_pdp:
            w = _attn_weights_dynamic(
                p_vec, self.prm.kappa, self.last_min_price, self.last_winners,
                tau=self.prm.cushion_tau, T=min(self.prm.cushion_T, self.prm.K),
                bonus_pct=self.prm.bonus_pct
            )
        else:
            w = _attn_weights_mean1(p_vec, self.prm.kappa)

        shares = _logit_shares_with_attention(
            p_vec, self.prm.a, self.prm.a0, self.prm.mu, w, theta=self.theta
        )
        prof = (p_vec - self.prm.c) * shares

        pmin = float(p_vec.min())
        winners = tuple(np.flatnonzero(p_vec == pmin).tolist())
        self.last_min_price = pmin
        self.last_winners.append(winners)
        if len(self.last_winners) > max(1, self.prm.K): self.last_winners.pop(0)

        if self.prm.sigma > 0.0:
            eps = self.rng.normal()
            self.theta = self.prm.rho * self.theta + self.prm.sigma * eps

        self.hist = self.hist[1:] + [tuple(acts)]
        info = {"p_vec": p_vec.copy(), "theta": float(self.theta), "winners": winners}
        obs = (tuple(self.hist), float(self.theta)) if self.prm.observed_shocks else tuple(self.hist)
        return obs, prof, info

# ---------- 5) Q-learning ----------
class QAgent:
    def __init__(self, n_actions, epsilon=0.05, alpha=0.05, gamma=0.99, seed=0):
        self.n_actions = n_actions
        self.epsilon = epsilon; self.alpha = alpha; self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, float))
        self._last = None

    def _eps_greedy(self, s):
        q = self.Q[s]
        if self.rng.random() < self.epsilon: return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def act(self, s):
        a = self._eps_greedy(s); self._last = (s, a); return a

    def update(self, s2, r):
        if self._last is None: return
        s, a = self._last
        qsa = self.Q[s][a]; maxn = self.Q[s2].max()
        self.Q[s][a] = (1-self.alpha)*qsa + self.alpha*(r + self.gamma*maxn)
        self._last = None

def _state_key(obs, include_theta: bool):
    if include_theta:
        hist, theta = obs; return (tuple(hist), round(float(theta), 3))
    return tuple(obs)

def train(env: AttentionEnv, episodes=20000, epsilon=0.05, alpha=0.05, gamma=0.99, seed=0):
    agents = [QAgent(len(env.A), epsilon, alpha, gamma, seed+10*i) for i in range(env.n)]
    obs = env.reset(seed)
    s = _state_key(obs, env.prm.observed_shocks)
    for _ in range(episodes):
        acts = [ag.act(s) for ag in agents]
        obs2, prof, _ = env.step(acts)
        s2 = _state_key(obs2, env.prm.observed_shocks)
        for i, ag in enumerate(agents): ag.update(s2, float(prof[i]))
        s = s2
    return agents

def rollout(env: AttentionEnv, agents: List[QAgent], T=5000, seed=123):
    obs = env.reset(seed)
    s = _state_key(obs, env.prm.observed_shocks)
    prices, lerner = [], []
    for _ in range(T):
        acts = [int(np.argmax(ag.Q[s])) if s in ag.Q else int(np.argmax(next(iter(ag.Q.values()))))
                for ag in agents]
        obs2, prof, info = env.step(acts)
        p = info["p_vec"]; prices.append(p.copy())
        lerner.append(((p - env.prm.c) / p).copy())
        s = _state_key(obs2, env.prm.observed_shocks)
    return np.array(prices), np.array(lerner)

# ---------- 6) kappa sweep & threshold ----------
def run_for_kappa(n=2, K=1, kappa=0.0, cushion_T=1, episodes=20000, seed=0,
                  sigma=0.0, rho=0.0, observed_shocks=False, dyn_pdp=True):
    prm = SimParams(n=n, K=K, kappa=kappa, cushion_T=cushion_T,
                    sigma=sigma, rho=rho, observed_shocks=observed_shocks,
                    dyn_pdp=dyn_pdp)
    env = AttentionEnv(prm, seed=seed)
    agents = train(env, episodes=episodes, epsilon=0.05, alpha=0.05, gamma=0.99, seed=seed)
    P, L = rollout(env, agents, T=5000, seed=seed+1)
    return float(L.mean()), float(L.std())

def estimate_kappa_star(kap_grid, target_markup=0.05, **kwargs):
    """
    Return dataframe-like dict with mean/sd per kappa (descending order respected),
    and kappa* = smallest kappa s.t. mean_markup <= target.
    """
    rows = []
    for kap in kap_grid:
        m, s = run_for_kappa(kappa=kap, **kwargs)
        rows.append({"kappa": kap, "markup_mean": m, "markup_sd": s})
        print(f"kappa={kap:>6}: mean={m:.3f}, sd={s:.3f}")
    # find threshold
    kstar = None
    for r in rows:
        if r["markup_mean"] <= target_markup:
            kstar = r["kappa"]; break
    return rows, kstar

def plot_kappa_threshold(rows, kstar, out_dir, tag, tol_markup=0.05):
    os.makedirs(out_dir, exist_ok=True)
    # Ensure descending display (∞, 100, 50, ... , 0)
    def _kap_sort(x):
        return (1e12 if np.isinf(x) else float(x))
    rows_sorted = sorted(rows, key=lambda r: -_kap_sort(r["kappa"]))
    xs = [r["kappa"] for r in rows_sorted]
    means = [r["markup_mean"] for r in rows_sorted]
    sds = [r["markup_sd"] for r in rows_sorted]

    xticks = [("∞" if np.isinf(x) else str(x)) for x in xs]
    fig = plt.figure(figsize=(7,4), dpi=200)
    plt.errorbar(range(len(xs)), means, yerr=sds, marker="o", linewidth=2)
    plt.axhline(tol_markup, linestyle="--")
    plt.title("Threshold of Steering Intensity to Restore Competition")
    plt.ylabel("Long-run markup (Lerner index: (p-c)/p)")
    plt.xlabel("Attention intensity κ (descending)")
    plt.xticks(range(len(xs)), xticks)
    if kstar is not None:
        idx = xs.index(kstar)
        plt.scatter([idx], [means[idx]], s=60)
        plt.text(idx, means[idx]+0.01, f"κ* = {xticks[idx]}", ha="center")
    pdf = os.path.join(out_dir, f"{tag}_kappa_threshold.pdf")
    png = os.path.join(out_dir, f"{tag}_kappa_threshold.png")
    plt.tight_layout(); plt.savefig(pdf); plt.savefig(png); plt.close()
    return pdf, png
# ---------- 6.5) Debug helpers: verify that state holds last K actions ----------
def _extract_hist_from_obs(obs, observed_shocks: bool):
    """Return just the history tuple from an env observation."""
    return obs[0] if observed_shocks else obs

def verify_state_memory(n=2, K=2, seed=777):
    """
    Create an env with given K and show that the state (and our state key)
    encodes the last K action profiles. We take two manual steps with
    known actions to make it obvious.
    """
    prm = SimParams(n=n, K=K, observed_shocks=True, dyn_pdp=True)  # turn shocks on so obs format is (hist, theta)
    env = AttentionEnv(prm, seed=seed)

    obs = env.reset(seed=seed)
    hist = _extract_hist_from_obs(obs, observed_shocks=True)
    print(f"[verify] K={K} after reset: len(hist)={len(hist)} (expected {K})")
    print(f"[verify] hist[0..]: {hist}")

    # Step 1: everyone picks the lowest-price index 0
    acts1 = [0]*n
    obs1, prof1, info1 = env.step(acts1)
    h1 = _extract_hist_from_obs(obs1, observed_shocks=True)
    print(f"[verify] after step1 acts={acts1}: hist={h1}")

    # Step 2: everyone picks the highest-price index (len(A)-1)
    acts2 = [len(env.A)-1]*n
    obs2, prof2, info2 = env.step(acts2)
    h2 = _extract_hist_from_obs(obs2, observed_shocks=True)
    print(f"[verify] after step2 acts={acts2}: hist={h2}")

    # What the learner actually sees as state key:
    skey0 = _state_key(obs,  True)
    skey1 = _state_key(obs1, True)
    skey2 = _state_key(obs2, True)
    print(f"[verify] state_key reset: {skey0}")
    print(f"[verify] state_key s1   : {skey1}")
    print(f"[verify] state_key s2   : {skey2}")
def compare_K_for_n(n=2):
    """
    Run the same kappa sweep for K=1 vs K=2 and print the mean Lerner results,
    then run a short memory verification so you can see the last-K action tuples.
    """
    OUT_DIR = "results"
    kap_grid = [np.inf, 100, 50, 25, 10, 5, 1, 0]

    for K in (1, 2):
        tag = f"n{n}_K{K}_shock"
        print(f"\n=== K={K} (n={n}) : kappa sweep ===")
        rows, kstar = estimate_kappa_star(
            kap_grid=kap_grid,
            n=n, K=K,
            cushion_T=K,                 # platform memory aligned with K
            episodes=150_000, seed=99,
            sigma=0.25, rho=0.8, observed_shocks=True,
            dyn_pdp=True
        )
        print("Estimated kappa* =", kstar)
        pdf, png = plot_kappa_threshold(rows, kstar, out_dir=OUT_DIR, tag=tag, tol_markup=0.05)
        print("Saved plots:", pdf, png)

        # Also show a quick state-memory trace for this K
        print(f"\n--- Verifying state memory for K={K} (n={n}) ---")
        verify_state_memory(n=n, K=K, seed=777)

# ---------- 7) Main ----------
def main():
    OUT_DIR = "results"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Show from strong to weak steering (∞ → 0)
    kap_grid = [np.inf, 100, 50, 25, 10, 5, 1, 0]

    # Run both K=1 and K=2 for n = 2,3,4,5
    cfgs = []
    for n in (2, 3, 4, 5):
        for K in (1, 2):
            cfgs.append(dict(n=n, K=K, tag=f"calvano_n{n}_K{K}_shock_doublecheck"))

    for cfg in cfgs:
        print(f"\n=== Running n={cfg['n']}  K={cfg['K']} ===")

        # quick debug: print the K and cushion_T we pass through
        print(f"[debug] will pass K={cfg['K']} and cushion_T={cfg['K']}")

        rows, kstar = estimate_kappa_star(
            kap_grid=kap_grid,
            n=cfg["n"], K=cfg["K"],
            cushion_T=cfg["K"],              # platform memory aligns with K
            episodes=150_000, seed=99,
            sigma=0.25, rho=0.8, observed_shocks=True,
            dyn_pdp=True
        )
        print("Estimated kappa* =", kstar)

        pdf, png = plot_kappa_threshold(
            rows, kstar, out_dir=OUT_DIR, tag=cfg["tag"], tol_markup=0.05
        )
        print("Saved plots:", pdf, png)

        # ---- Verify the state truly contains the last K action profiles ----
        # Build a tiny env and take two manual steps so you can SEE the history.
        prm = SimParams(
            n=cfg["n"], K=cfg["K"], observed_shocks=True,
            dyn_pdp=True, cushion_T=cfg["K"]
        )
        env = AttentionEnv(prm, seed=777)
        obs = env.reset(seed=777)

        # history length should equal K
        hist0 = obs[0]  # because observed_shocks=True -> obs = (hist, theta)
        print(f"[verify] after reset: len(hist)={len(hist0)} (expected {cfg['K']})")
        print(f"[verify] hist = {hist0}")

        # step 1: all choose lowest price index
        acts1 = [0] * cfg["n"]
        obs1, _, _ = env.step(acts1)
        hist1 = obs1[0]
        print(f"[verify] after step1 acts={acts1}: hist={hist1}")

        # step 2: all choose highest price index
        acts2 = [len(env.A) - 1] * cfg["n"]
        obs2, _, _ = env.step(acts2)
        hist2 = obs2[0]
        print(f"[verify] after step2 acts={acts2}: hist={hist2}")

        # Also show the exact state keys the learner would use
        from_state = _state_key(obs,  include_theta=True)
        s1_state   = _state_key(obs1, include_theta=True)
        s2_state   = _state_key(obs2, include_theta=True)
        print(f"[verify] state_key reset: {from_state}")
        print(f"[verify] state_key s1   : {s1_state}")
        print(f"[verify] state_key s2   : {s2_state}")


if __name__ == "__main__":
    main()
