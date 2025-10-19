import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from joblib import Parallel, delayed
from .env import AttentionEnv, SimParams
from .agents import QLearner, AgentParams
from .io import ensure_dir, ckpt_paths

@dataclass
class TrainParams:
    T_max: int = 120_000
    replications: int = 16
    base_seed: int = 42
    n_jobs: int = -1

class Trainer:
    def __init__(self, sim: SimParams, ap: AgentParams, out_dir: str = "results"):
        self.sim, self.ap = sim, ap
        self.out_dir = ensure_dir(out_dir)

    def _train_one(self, seed: int) -> Tuple[AttentionEnv, List[QLearner], tuple]:
        env = AttentionEnv(self.sim, seed=seed)
        rng = np.random.default_rng(seed)
        agents = [QLearner(AgentParams(n_actions=len(env.A),
                                       alpha=self.ap.alpha,
                                       delta=self.ap.delta,
                                       tau0=self.ap.tau0,
                                       tau_min=self.ap.tau_min,
                                       gamma=self.ap.gamma),
                           rng) for _ in range(self.sim.n)]
        s = env.reset(seed=seed)
        for t in range(self.Tmax):
            acts = [ag.select_action(s) for ag in agents]
            s_next, r, _ = env.step(acts)
            for i, ag in enumerate(agents):
                ag.update(s, acts[i], r[i], s_next)
            s = s_next
        return env, agents, s

    def train(self, tr: TrainParams, tag: str):
        self.Tmax = tr.T_max
        seeds = [tr.base_seed + 17*i for i in range(tr.replications)]

        def _run_and_save(sd):
            env, agents, s = self._train_one(sd)
            # save Q-tables
            for i, ag in enumerate(agents):
                path = ckpt_paths(self.out_dir, tag, sd, i)
                meta = dict(self.sim.__dict__)
                meta.update(dict(n_actions=len(env.A)))
                ag.save_Q(path, meta)
            return sd

        Parallel(n_jobs=tr.n_jobs, prefer="processes")(
            delayed(_run_and_save)(sd) for sd in seeds
        )
        return seeds
