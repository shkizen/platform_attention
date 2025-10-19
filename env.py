import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict

# ---------------- SimParams ----------------
@dataclass(frozen=True)
class SimParams:
    # firms / costs
    n: int = 2
    c: float = 1.0

    # logit demand primitives
    a: float = 2.0       # mean utility level for each seller (symmetric)
    a0: float = 0.0      # outside option utility
    mu: float = 0.25     # logit scale (price sensitivity) > 0

    # price grid & memory
    price_grid: Tuple[float, ...] = (1.424,1.464,1.505,1.545,1.586,1.626,1.667,1.707,1.747,1.788,1.828,1.869,1.909,1.950,1.990)
    K: int = 1
    seed: int = 42

    # legacy/compat fields (unused in this pure-logit static env)
    m: int = 15
    xi: float = 0.1
    b: float = 0.0
    phi: float = 1.0
    sigma: float = 0.0
    rho: float = 0.0
    observed_shocks: bool = False

    # platform attention sensitivity (0 => neutral; inf => WTA)
    kappa: float = 0.0


# ------------- helper functions (attention + shares) -------------
def _attn_weights_mean1(p_vec: np.ndarray, kappa: float) -> np.ndarray:
    """
    Return attention weights normalized to have mean 1 across firms.
    - kappa = 0   : all ones (neutral attention)
    - kappa = inf : winner-take-all for the minimum price (tie split)
    - else        : softmax over relative price gaps
    """
    p_vec = np.asarray(p_vec, float)
    n = p_vec.size
    if kappa == 0.0:
        return np.ones(n, dtype=float)

    if np.isinf(kappa):
        w = (p_vec == p_vec.min()).astype(float)
        w /= w.sum()               # split ties
        return n * w               # mean 1

    gaps = p_vec - p_vec.min()
    z = -kappa * gaps
    z -= z.max()                   # numerical stability
    raw = np.exp(z)
    return n * (raw / raw.sum())   # mean 1


def _logit_shares_with_attention(
    p_vec: np.ndarray,
    a: float,
    a0: float,
    mu: float,
    kappa: float,
    theta: float = 0.0,   # <--- 新增
) -> np.ndarray:
    """
    Multinomial logit with outside option, scaled by platform
    'prominence' weights normalized to mean 1 (so kappa=0 is neutral).
    Common demand shock theta shifts the inside-good utility.
    """
    p_vec = np.asarray(p_vec, float)
    assert mu > 0, "mu must be > 0"

    v  = (a + theta - p_vec) / mu   # <--- 把 theta 加到 inside utility
    v0 = a0 / mu
    mx = max(v.max(), v0)           # stability shift

    w   = _attn_weights_mean1(p_vec, kappa)
    num = w * np.exp(v - mx)
    den = np.exp(v0 - mx) + num.sum()
    return num / den


# ----------------- single, clean AttentionEnv -----------------
class AttentionEnv:
    def __init__(self, prm: SimParams, seed: int = None):
        self.prm = prm
        self.A   = np.array(prm.price_grid, dtype=float)
        self.n   = int(prm.n)
        self.rng = np.random.default_rng(prm.seed if seed is None else seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mid  = int(len(self.A) // 2)
        init = (mid,) * self.n
        # shock 初值：給 0（也可改成穩態分佈 sigma/sqrt(1-rho^2)）
        self.theta = 0.0
        self.hist  = [init for _ in range(max(1, self.prm.K))]
        # 若要把 shock 放進觀測（observed_shocks=True），可在此一併返回
        return (tuple(self.hist), float(self.theta)) if self.prm.observed_shocks else tuple(self.hist)

    def step(self, acts: List[int]):
        acts = list(map(int, acts))
        assert len(acts) == self.n, "One action index per firm required."
        p_vec = self.A[np.array(acts, dtype=int)]

        # 利潤（用當期 theta）
        prof = self.static_profits(p_vec, theta=self.theta)

        # 狀態轉移：更新 theta
        if self.prm.sigma > 0.0:
            eps = self.rng.normal()
            self.theta = self.prm.rho * self.theta + self.prm.sigma * eps

        # 更新歷史
        self.hist = self.hist[1:] + [tuple(acts)]

        info = {"p_vec": p_vec.copy(), "theta": float(self.theta)}
        obs  = (tuple(self.hist), float(self.theta)) if self.prm.observed_shocks else tuple(self.hist)
        return obs, prof, info

    def static_profits(self, p_vec, theta=None):
        s = _logit_shares_with_attention(
            p_vec,
            a=self.prm.a, a0=self.prm.a0,
            mu=self.prm.mu, kappa=self.prm.kappa,
            theta=0.0 if theta is None else float(theta)   # <--- 傳入 shock
        )
        return (np.asarray(p_vec, float) - self.prm.c) * s

    def static_profits_batch(self, P: np.ndarray, theta=None):
        out = np.empty_like(P, dtype=float)
        th  = 0.0 if theta is None else float(theta)
        for b in range(P.shape[0]):
            s = _logit_shares_with_attention(
                P[b], a=self.prm.a, a0=self.prm.a0,
                mu=self.prm.mu, kappa=self.prm.kappa,
                theta=th                                 # <--- 傳入 shock
            )
            out[b] = (P[b] - self.prm.c) * s
        return out


    # ---- convenience / diagnostics ----
    def markup_level(self, p_vec: np.ndarray) -> np.ndarray:
        """Level markup: p - c."""
        p_vec = np.asarray(p_vec, float)
        return p_vec - self.prm.c

    def markup_lerner(self, p_vec: np.ndarray) -> np.ndarray:
        """Lerner index: (p - c) / p."""
        p_vec = np.asarray(p_vec, float)
        return (p_vec - self.prm.c) / p_vec

    def markup_table_symmetric_grid(self) -> List[Dict[str, float]]:
        """ Symmetric profile (p,p,...,p) at each grid point; return level & Lerner. """
        rows: List[Dict[str, float]] = []
        for p in self.A:
            m_level = float(p - self.prm.c)
            m_lern  = float((p - self.prm.c) / p)
            rows.append({"p": float(p), "markup_level": m_level, "lerner": m_lern})
        return rows

    def to_dict(self) -> Dict:
        return asdict(self.prm)
