import pandas as pd, os
from .io import ensure_dir

def write_kappa_star_table(df: pd.DataFrame, kstar: float, out_dir: str, tag: str):
    ensure_dir(out_dir)
    path_csv = os.path.join(out_dir, "tables", f"kappa_star_{tag}.csv")
    ensure_dir(os.path.dirname(path_csv))
    df.to_csv(path_csv, index=False)
    # minimal LaTeX table stub
    tex = df.to_latex(index=False, float_format="%.3f", caption=f"Kappa grid and markup — tag={tag}", label=f"tab:kappa_{tag}")
    path_tex = os.path.join(out_dir, "tables", f"kappa_star_{tag}.tex")
    with open(path_tex, "w") as f: f.write(tex)
    return path_csv, path_tex
