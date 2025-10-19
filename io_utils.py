import os, json, numpy as np
from typing import Tuple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def ckpt_paths(base_dir: str, tag: str, seed: int, agent_idx: int) -> str:
    return os.path.join(base_dir, "checkpoints", tag, f"Q_agent{agent_idx}_seed{seed}.json")

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f: return json.load(f)

def remap_Q_to_new_grid(Q: dict, old_grid: np.ndarray, new_grid: np.ndarray) -> dict:
    """Nearest-neighbor remap of action index dimension when grid changes."""
    if not Q: return Q
    m = len(new_grid)
    newQ = {}
    for s, q in Q.items():
        qnew = np.full(m, -1e12, dtype=float)
        for a_old, val in enumerate(q):
            a_val = old_grid[a_old]
            a_new = int(np.abs(new_grid - a_val).argmin())
            qnew[a_new] = max(qnew[a_new], val)
        newQ[s] = qnew
    return newQ

def tag_from_params(n:int,K:int,grid:tuple,kappa:float,delta:float,mu:float)->str:
    kappa_tag = "inf" if np.isinf(kappa) else str(kappa).replace(".","p")
    grid_tag = "g" + "-".join(str(x) for x in grid)
    return f"n{n}_K{K}_{grid_tag}_kap{kappa_tag}_del{str(delta).replace('.','p')}_mu{str(mu).replace('.','p')}"
