# Platform Attention — Replication Package

This repository replicates and extends Calvano-style algorithmic pricing with:
- **Attention steering** (κ-intensity) that reallocates demand toward low-price products, optionally with **dynamic platform memory** (cushion rule).
- **AR(1) demand shocks** that enter utility directly.
- **Finite-memory Q-learning** agents (state = last **K** joint actions; K = 1 or 2).
- Static **Bertrand (Nash)** and **Joint-Monopoly** benchmarks on a fixed price grid.

> The code reproduces three main results discussed in the paper:
> 1. **Higher profit gain with fewer firms** (Table 2: long-run Lerner index & mean price vs n).  
> 2. **Higher profit gain with longer memory** (by comparing K across otherwise identical runs).  
> 3. **Profit gain is more volatile under shocks** (variance comparison with and without AR(1) shocks).

---

## 🔧 Environment

- **Python** ≥ 3.9  
- Required packages: `numpy`, `pandas`, `matplotlib`, `joblib` (optional, for parallel training), `dataclasses`.

Install dependencies:

```bash
pip install -r requirements.txt
# or manually:
pip install numpy pandas matplotlib joblib
```

---

## 📦 Repository Structure

```
platform_attention/
├── __init__.py
├── agents.py          # Q-learning agents & hyperparameters
├── env.py             # Environment: attention weights, logit shares, AR(1) shocks
├── eval.py            # Static benchmarks: Nash & Monopoly prices, policy loaders
├── io.py              # I/O utilities (ensure_dir, tag_from_params)
├── train.py           # Trainer and training parameters (idempotent)
├── tables.py          # (optional) legacy table writers
├── threshold.py       # (optional) threshold estimation for κ*
├── memory_check.py    # κ-sweep & state-memory verification
├── generate_all.py    # ← One-shot generator for all outputs (tables & volatility)
├── results/           # Auto-generated tables and plots (ignored by git)
└── testing.ipynb      # Example Jupyter notebook for quick testing
```

---

## 🚀 Reproducing All Outputs

Run everything in one shot:

```bash
python generate_all.py --all
```

This will generate the following outputs:

- `results/tables/table2_longrun_markup_meanprice.csv` / `.tex`  
- `results/tables/table3_pB_pM_by_n.csv` / `.tex`  
- `results/tables/profit_gain_volatility.csv` / `.tex`  
- (Optional) κ-sweep threshold plots under `results/` (PDF + PNG)

### Individual Modules

You can also run them separately:

```bash
python generate_all.py --table2        # Long-run markup & mean price by n
python generate_all.py --table3        # Bertrand & Monopoly prices by n
python generate_all.py --volatility    # Profit-gain volatility (shock vs no shock)
python generate_all.py --kappa-sweep   # Optional κ-sweep plots
```

---

## 🧪 Description of Each Output

### **Table 2 – Long-run outcomes**
- Trains Q-learners per n ∈ {2,3,4,5} using fixed primitives.
- Loads the greedy policies, burns into the learned cycle, and averages:
  - **Lerner index** (p−c)/p
  - **Mean price**

### **Table 3 – Static benchmarks**
- For each n, computes the **Bertrand (Nash)** price and the **Joint-Monopoly** price using the same discrete price grid (θ = 0).

### **Volatility comparison**
- Trains once with σ = 0.
- Evaluates the same greedy policies under:
  1. No shocks (σ = 0, ρ = 0)
  2. AR(1) shocks (σ = 0.15, ρ = 0.85)
- Reports the variance of per-period **profit gain relative to Bertrand**.

---

## ⚙️ Key Modeling Details

- **Demand & Attention**  
  Logit shares use inside utility uᵢ = a + θ − pᵢ with temperature parameter μ.  
  Attention weights wᵢ(κ) are normalized to have mean 1.  
  Dynamic platform steering (Johnson-style) boosts recent low-price winners within a small price cushion.

- **State Representation**  
  The state encodes the last K joint actions (and θ if `observed_shocks=True`).

- **Learning Process**  
  Tabular Q-learning with ε-greedy exploration during training; greedy rollout during evaluation.

- **Static Benchmarks**  
  Bertrand and joint-monopoly prices are computed on the same discrete grid for comparability.

- **Shock Process**  
  θₜ = ρ θₜ₋₁ + σ εₜ (AR(1)), where shocks add to utility and affect demand volatility.

---

## 🔁 Reproducibility Notes

- All runs use fixed seeds; `Trainer` is idempotent, so rerunning will reuse cached checkpoints.  
- For quick tests, you can reduce `replications` or `T_max` in `generate_all.py`.  
- All outputs are stored under `results/` to keep the repository clean.

---

## 🧩 Sanity Checks

Run:

```bash
python memory_check.py
```

This prints **state-memory traces** showing that the state encodes the last-K joint actions and generates κ-sweep plots verifying threshold behavior.

---

## 🧠 How to Cite

If you use or extend this package, please cite:

> Khan, A. (2025). *Platform Attention — Replication Package.*  
> GitHub: https://github.com/yourusername/platform_attention

(If archived on Zenodo, replace the URL with the DOI.)

---

## 📝 License

Recommended license: **MIT License**

```
MIT License
Copyright (c) 2025 Asif Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
[...]
```

---

## ✅ Quick Summary

| Output Type | File Path | Description |
|--------------|------------|-------------|
| **Table 2** | `results/tables/table2_longrun_markup_meanprice.csv` | Mean Lerner index and mean price by number of firms |
| **Table 3** | `results/tables/table3_pB_pM_by_n.csv` | Bertrand and monopoly prices by n |
| **Volatility** | `results/tables/profit_gain_volatility.csv` | Profit-gain variance with and without shocks |
| **κ Sweep (optional)** | `results/*.pdf` / `.png` | Threshold plots for steering intensity (κ*) |

---

## 🧭 Versioning

Tag the final replication release before submission:

```bash
git tag -a v1.0 -m "Replication release"
git push origin v1.0
```

---

**Author:** Asif Khan  
**Affiliation:** Purdue University  
**Year:** 2025
