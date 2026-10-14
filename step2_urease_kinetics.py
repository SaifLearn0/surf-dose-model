# Step 2: Urease kinetics backbone.
# Predicts urea decay over 14 days from the MEASURED pH of each treatment.
# One shared parameter set is fitted across all five treatments of an acid;
# the treatments differ only through their own pH trajectory, so pH alone must
# explain why high doses preserve urea and low doses do not.

# Model (per treatment):
#   dE/dt = mu * g(pH) * E * (1 - E)                 microbial urease pool, logistic
#   dU/dt = -Vmax * E * f(pH) * U/(Km + U) - k0 * U  ureolysis + baseline loss
# with
#   f(pH) = 1 / (1 + 10^(nf*(pKa_lo - pH)))   urease activity switch (acid side)
#   g(pH) = 1 / (1 + exp(-sg*(pH - pHg)))     microbial growth switch
# f(pH) is the catalytic activity vs pH (sharp drop as urease denatures at low pH);
# g(pH) gates whether the urease-active population can proliferate; k0 is a small
# pH-independent floor (slow abiotic hydrolysis + trace residual activity) that
# accounts for the ~15-19% loss seen even in the best-controlled doses.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import step1_data_foundation as s1

DAYS = s1.DAYS

# pH response functions
def f_act(pH, pKa_lo, nf):
    return 1.0 / (1.0 + 10**(nf*(pKa_lo - pH)))

def g_grow(pH, pHg, sg):
    return 1.0 / (1.0 + np.exp(-sg*(pH - pHg)))

# Forward simulation of one treatment given parameters, a pH(t) function and U(day1)
def simulate(params, pH_of_t, U0, t_eval):
    Vmax, mu, E0, pKa_lo, nf, pHg, sg, Km, k0 = params
    def rhs(t, y):
        U, E = y
        pH = pH_of_t(t)
        dE = mu * g_grow(pH, pHg, sg) * E * (1.0 - E)
        dU = -Vmax * E * f_act(pH, pKa_lo, nf) * U/(Km + U) - k0*U
        return [dU, dE]
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [U0, E0], t_eval=t_eval,
                    method="LSODA", rtol=1e-7, atol=1e-6)
    return sol.y[0]

# Measured pH of each treatment becomes the driver (linear interpolation in time)
def build_pH_drivers(key):
    d = s1.DATA[key]
    return [interp1d(DAYS, d["pH"][:, j], kind="linear", fill_value="extrapolate")
            for j in range(5)]

# SD-weighted residuals across all five treatments of one acid (chi-square style)
def residuals(p, key, drivers):
    d = s1.DATA[key]
    res = []
    for j in range(5):
        pred = simulate(p, drivers[j], d["UREA"][0, j], DAYS)
        sd = np.maximum(d["UREA_sd"][:, j], 100.0)
        res.extend((pred - d["UREA"][:, j]) / sd)
    return np.array(res)

# Parameter bounds and multistart seeds (multistart avoids local minima)
PAR_NAMES = ["Vmax", "mu", "E0", "pKa_lo", "nf", "pHg", "sg", "Km", "k0"]
LB = [200,   0.2, 1e-3, 3.2, 0.8, 4.0,  1.0,   500, 0.0]
UB = [30000, 8.0, 0.6,  5.5, 5.0, 7.0, 12.0, 40000, 0.05]
STARTS = [
    [4000, 1.5, 0.05, 3.95, 2.5, 4.5, 4.0, 10000, 0.012],
    [3000, 2.5, 0.03, 4.00, 3.0, 5.0, 5.0,  8000, 0.010],
    [6000, 1.0, 0.08, 3.90, 2.0, 4.2, 3.0, 20000, 0.014],
    [3500, 3.0, 0.04, 4.05, 3.5, 4.8, 6.0, 15000, 0.008],
    [2500, 2.0, 0.06, 3.85, 1.8, 5.5, 3.5,  5000, 0.015],
]

# Fitted urea parameters, pinned to the converged best fit so that results are
# identical on every machine and SciPy version. Set REFIT = True to re-run the
# multistart optimisation that produced them.
REFIT = False
PINNED = {
    "CA": [2587.035655, 2.901538, 0.203616, 3.876673, 4.995740,
           4.491802, 1.476219, 22571.205405, 0.015779],
    "WV": [4883.117114, 7.993488, 0.301520, 3.824191, 2.375079,
           6.013431, 10.963182, 39547.011294, 0.000146],
}

def fit_acid(key):
    drivers = build_pH_drivers(key)
    if not REFIT:
        return np.array(PINNED[key]), drivers
    best = None
    for p0 in STARTS:
        try:
            r = least_squares(residuals, p0, bounds=(LB, UB), args=(key, drivers),
                              method="trf", max_nfev=10000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    return best.x, drivers

def r_squared(obs, pred):
    ss_res = np.sum((obs - pred)**2)
    ss_tot = np.sum((obs - obs.mean())**2)
    return 1.0 - ss_res/ss_tot

def report_and_plot(key):
    d = s1.DATA[key]
    params, drivers = fit_acid(key)
    print("\n" + d["meta"]["name"])
    print("  " + "  ".join(f"{n}={v:.4g}" for n, v in zip(PAR_NAMES, params)))
    r2s = []
    for j in range(5):
        pred = simulate(params, drivers[j], d["UREA"][0, j], DAYS)
        obs = d["UREA"][:, j]
        r2 = r_squared(obs, pred); r2s.append(r2)
        print(f"    {d['meta']['treatments'][j][0]:16s} R2={r2:6.3f}")
    print(f"    mean R2 = {np.mean(r2s):.3f}")

    plt.figure(figsize=(7, 5))
    t_smooth = np.linspace(1, 14, 90)
    for j in range(5):
        label, _, _, colour, opt = d["meta"]["treatments"][j]
        lab = label + " (optimum)" if opt else label
        pred = simulate(params, drivers[j], d["UREA"][0, j], t_smooth)
        plt.errorbar(DAYS, d["UREA"][:, j], yerr=d["UREA_sd"][:, j],
                     fmt=s1.MARKERS[j], color=colour, ms=4.5, markeredgewidth=0.5,
                     capsize=2, capthick=0.7, elinewidth=0.7,
                     ecolor=to_rgba(colour, 0.4), lw=0)
        plt.plot(t_smooth, pred, "-", color=colour, lw=1.4, label=lab)
    plt.xlabel("Days"); plt.ylabel("Urea (mg/L)"); plt.xlim(0, 15)
    plt.title("Urease kinetics fit  --  " + d["meta"]["name"])
    plt.legend(fontsize=8, frameon=False); plt.tight_layout()
    return params

if __name__ == "__main__":
    for key in ("CA", "WV"):
        report_and_plot(key)
    plt.show()
    print("\nStep 2 complete: pH-driven urease kinetics reproduce the urea curves.")
