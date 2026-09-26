"""TASK 2b: which force sets B's ACCURACY clock -- the individual term, the shared term,
or their sum? Plus a free power-law fit to see which derived exponent is wrong."""
import numpy as np
from timing_extract import load_all2

rows = load_all2()

FORMS = {
    "R_sum   = (1-b)(1-1/V)/n_B + b/V": lambda r: (1 - r["beta"]) * (1 - 1.0 / r["v"]) / r["nb"] + r["beta"] / r["v"],
    "R_ind   = (1-b)(1-1/V)/n_B":       lambda r: (1 - r["beta"]) * (1 - 1.0 / r["v"]) / r["nb"],
    "R_ind0  = (1-1/V)/n_B":            lambda r: (1 - 1.0 / r["v"]) / r["nb"],
    "R_shared= b/V":                    lambda r: max(r["beta"], 1e-3) / r["v"],
    "R_sqrt  = (1-b)/sqrt(n_B V)":      lambda r: (1 - r["beta"]) / np.sqrt(r["nb"] * r["v"]),
}


def score(R, f):
    x = np.array([1.0 / (r["ilr"] * r["d"] * f(r)) for r in R])
    y = np.array([r["t_B_on"] for r in R])
    C = float(np.exp(np.mean(np.log(y) - np.log(x))))
    lr_ = np.log(C * x / y)
    return C, float(np.exp(lr_.std())), float(np.mean(np.abs(lr_) < np.log(1.5)))


for arm, lbl in [(0, "LINEAR"), (1, "NORMALIZED")]:
    R = [r for r in rows if r["norm"] == arm and np.isfinite(r["t_B_on"])]
    print(f"=== {lbl} arm, n={len(R)}: t_B = M* / (eta * d * R) ===")
    for name, f in FORMS.items():
        C, gsd, frac = score(R, f)
        print(f"   {name:36s} M*={C:7.3f}  geo-sd {gsd:.2f}x  {100*frac:3.0f}% within 1.5x")
    # free power law
    Xc = [("log eta", [np.log(r["ilr"]) for r in R]),
          ("log d", [np.log(r["d"]) for r in R]),
          ("log n_B", [np.log(r["nb"]) for r in R]),
          ("log V", [np.log(r["v"]) for r in R]),
          ("log(1-b)", [np.log(max(1 - r["beta"], 0.02)) for r in R]),
          ("log n_A", [np.log(r["na"]) for r in R])]
    X = np.column_stack([np.ones(len(R))] + [np.array(v) for _, v in Xc])
    y = np.log([r["t_B_on"] for r in R])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    derived = {"log eta": -1, "log d": -1, "log n_B": +1, "log V": 0, "log(1-b)": -1,
               "log n_A": 0}
    print(f"   free power-law fit: R2 {r2:.3f}  geo-sd of residual "
          f"{np.exp((y-pred).std()):.2f}x")
    print(f"     {'term':10s} {'fitted':>8s} {'derived':>8s}")
    for i, (nm, _) in enumerate(Xc):
        print(f"     {nm:10s} {coef[i+1]:+8.3f} {derived[nm]:+8.0f}")
    print()
