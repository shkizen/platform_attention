import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
import json, hashlib, os

@dataclass
class AgentParams:
    n_actions: int
    alpha: float = 0.05
    delta: float = 0.95
    tau0: float = 1.0
    tau_min: float = 0.01
    gamma: float = 0.99997  # slow cooling

class QLearner:
    def __init__(self, ap: AgentParams, rng=None):
        self.ap = ap
        self.rng = np.random.default_rng() if rng is None else rng
        self.Q: Dict[Tuple, np.ndarray] = {}
        self.tau = ap.tau0

    def _getQ(self, s):
        q = self.Q.get(s)
        if q is None:
            q = np.zeros(self.ap.n_actions, dtype=float)
            self.Q[s] = q
        return q

    def select_action(self, s):
        q = self._getQ(s)
        tau = max(self.ap.tau_min, self.tau)
        z = q / max(tau, 1e-8); z -= z.max()
        probs = np.exp(z); probs /= probs.sum()
        a = int(self.rng.choice(self.ap.n_actions, p=probs))
        self.tau = max(self.ap.tau_min, self.tau * self.ap.gamma)
        return a

    def act_greedy(self, s):
        q = self._getQ(s); return int(np.argmax(q))

    def update(self, s, a, r, s_next):
        q = self._getQ(s); q_next = self._getQ(s_next)
        td_target = r + self.ap.delta * np.max(q_next)
        q[a] = (1 - self.ap.alpha) * q[a] + self.ap.alpha * td_target

    # --------- persistence with fingerprint ---------
    @staticmethod
    def _fingerprint(meta: dict) -> str:
        s = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()

    def save_Q(self, path: str, meta: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serial = {repr(k): v.tolist() for k, v in self.Q.items()}
        meta = dict(meta); meta["n_actions"] = int(self.ap.n_actions)
        meta["fp"] = self._fingerprint(meta)
        payload = {"meta": meta, "Q": serial}
        with open(path, "w") as f: json.dump(payload, f)
        return meta["fp"]

    def load_Q(self, path: str):
        with open(path, "r") as f: payload = json.load(f)
        Q_raw = payload.get("Q", {})
        self.Q = {eval(k): np.array(v, dtype=float) for k, v in Q_raw.items()}
        return payload.get("meta", {})

class GreedyPolicy:
    """Read-only view of a trained Q table."""
    def __init__(self, Q: Dict):
        self.Q = Q
    def act_greedy(self, s):
        q = self.Q.get(s)
        return 0 if q is None else int(np.argmax(q))
