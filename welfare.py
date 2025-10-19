import numpy as np, pandas as pd
from .env import AttentionEnv
from .eval import static_nash_price

class WelfareCalculator:
    def __init__(self, env: AttentionEnv):
        self.env = env

    def compute_stream(self, greedy_agents, start_state, T: int = 5000, burn: int = 2000):
        s = start_state; prices=[]; cs=[]; fs=[]; ts=[]
        def g(a, st): return a.act_greedy(st)
        # burn
        for _ in range(burn):
            acts = [g(a, s) for a in greedy_agents]; s, _, _ = self.env.step(acts)
        for _ in range(T):
            acts = [g(a, s) for a in greedy_agents]
            s, _, info = self.env.step(acts); p = info["p_vec"]
            theta = 0.0  # or track actual; for welfare we can average over θ path
            prof = self.env.static_profits(p, theta)
            FS = prof.sum(); # consumer surplus (reduced-form): integral under inverse-demand proxy
            Q = max(self.env.prm.b + theta - p.min()*self.env.prm.phi, 0.0)
            CS = 0.5 * Q * max(p.min() - self.env.prm.c, 0.0)  # stylized; replace if you have exact demand
            TS = CS + FS
            prices.append(p.mean()); cs.append(CS); fs.append(FS); ts.append(TS)
        return dict(price=np.mean(prices), CS=np.mean(cs), FS=np.mean(fs), TS=np.mean(ts))
