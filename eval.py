import numpy as np
from typing import List, Tuple
from joblib import Parallel, delayed
from .env import AttentionEnv, SimParams
from .agents import GreedyPolicy, QLearner, AgentParams
from .io_utils import load_json, ckpt_paths, ensure_dir, save_json

# ---------- Static benchmarks using the same primitives ----------
def _static_best_response(env: AttentionEnv, rival_price: float, i: int, theta: float) -> float:
    A = env.A; best_p, best_v = None, -1e300
    for p in A:
        p_vec = np.full(env.n, rival_price); p_vec[i] = p
        v = env.static_profits(p_vec, theta)[i]
        if (v > best_v) or (np.isclose(v, best_v) and (best_p is not None) and (p < best_p)):
            best_p, best_v = float(p), v
    return best_p

def static_monopoly_price(env: AttentionEnv, theta: float) -> float:
    A = env.A; best_v, best_pair = -1e300, None
    for p0 in A:
        for p1 in A:
            pv = np.array([p0, p1])
            v = env.static_profits(pv, theta).sum()
            if v > best_v:
                best_v, best_pair = v, pv.copy()
    return float(np.mean(best_pair))

def static_nash_price(env: AttentionEnv, theta: float) -> float:
    A = env.A
    BR0 = {float(p1): _static_best_response(env, p1, i=0, theta=theta) for p1 in A}
    BR1 = {float(p0): _static_best_response(env, p0, i=1, theta=theta) for p0 in A}
    best_pair, best_err = None, 1e300
    for p0 in A:
        for p1 in A:
            err = (BR0[float(p1)] - p0)**2 + (BR1[float(p0)] - p1)**2
            if err < best_err:
                best_err, best_pair = err, (p0, p1)
    return float(np.mean(best_pair))

# ---------- Impulse response (Calvano Fig-4 analog) ----------
def impulse_response_avg_once(env: AttentionEnv,
                              greedy_agents: List[GreedyPolicy],
                              start_state,
                              horizon_after: int = 15,
                              reps: int = 16,
                              warmup_greedy: int = 1000,
                              deviator_index: int = 0,
                              base_seed: int = 777,
                              n_jobs: int = -1,
                              out_path: str = None):
    # Warm once to learned cycle
    s = start_state
    def act_g(ag, st): return ag.act_greedy(st)
    for _ in range(warmup_greedy):
        acts = [act_g(a, s) for a in greedy_agents]
        s, _, _ = env.step(acts)

    steady_hist = list(env.hist); steady_theta = float(env.theta)
    A = env.A

    def run_once(rep_seed):
        env_r = AttentionEnv(env.prm, seed=rep_seed)
        env_r.hist = list(steady_hist); env_r.theta = steady_theta
        s0 = tuple(steady_hist)
        # τ=0
        acts0 = [act_g(a, s0) for a in greedy_agents]
        s1, _, info0 = env_r.step(acts0)
        p0 = info0["p_vec"].astype(float); th0 = float(info0["theta"])
        dev, riv = [p0[deviator_index]], [p0[1 - deviator_index]]
        # τ=1 (strict undercut by one tick, capped by BR)
        rival_p0 = p0[1 - deviator_index]
        r_idx = int(np.abs(A - rival_p0).argmin()); undercut_idx = max(0, r_idx - 1)
        p_dev_br = _static_best_response(env_r, rival_p0, i=deviator_index, theta=th0)
        br_idx = int(np.abs(A - p_dev_br).argmin())
        dev_idx = min(undercut_idx, br_idx)
        acts1 = [act_g(a, s1) for a in greedy_agents]; acts1[deviator_index] = dev_idx
        s2, _, info1 = env_r.step(acts1)
        p1 = info1["p_vec"].astype(float)
        dev.append(p1[deviator_index]); riv.append(p1[1 - deviator_index])
        # τ≥2 greedy
        s = s2
        for _ in range(horizon_after - 1):
            acts = [act_g(a, s) for a in greedy_agents]
            s, _, info = env_r.step(acts); p = info["p_vec"].astype(float)
            dev.append(p[deviator_index]); riv.append(p[1 - deviator_index])
        return np.array(dev), np.array(riv), th0

    seeds = [base_seed + 13*r for r in range(reps)]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(run_once)(sd) for sd in seeds
    )
    dev_paths, riv_paths, thetas = zip(*results)
    dev_mean = np.mean(np.stack(dev_paths), axis=0)
    riv_mean = np.mean(np.stack(riv_paths), axis=0)
    theta_bar = float(np.mean(thetas))

    if out_path:
        save_json(out_path, {
            "meta": env.prm.__dict__,
            "t": list(range(len(dev_mean))),
            "p_dev": dev_mean.tolist(),
            "p_riv": riv_mean.tolist(),
            "theta_bar": theta_bar
        })
    return dev_mean, riv_mean, theta_bar

# ---------- Utility to load saved Q and build greedy agents ----------
def load_greedy_agents(out_dir: str, tag: str, seed: int, sim: SimParams) -> Tuple[AttentionEnv, List[GreedyPolicy], tuple]:
    env = AttentionEnv(sim, seed=seed)
    agents = []
    for i in range(sim.n):
        path = ckpt_paths(out_dir, tag, seed, i)
        meta = load_json(path)["meta"]
        qtab = load_json(path)["Q"]
        Q = {eval(k): np.array(v, dtype=float) for k, v in qtab.items()}
        agents.append(GreedyPolicy(Q))
    s = env.reset(seed=seed)
    return env, agents, s
