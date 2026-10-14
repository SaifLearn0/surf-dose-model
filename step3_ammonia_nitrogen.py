# Step 3: Ammonia (TAN) layer and nitrogen budget.
# Builds on Step 2. Adds TAN as a third state variable and reports the
# nitrogen balance between urea lost and ammonia gained.

# TAN dynamics per treatment:
#   dTAN/dt = b0 + kT * E * h(pH) - kv * fNH3(pH) * TAN
# where
#   b0            background ammonia release from non-urea N (creatinine, amino
#                 acids, peptides in real urine); present even at low pH. It is
#                 not fitted: at the highest dose ureolysis is arrested, so the
#                 measured TAN rise in that treatment is the background release.
#   kT * E * h(pH) urea-derived ammonia: only the genuinely ureolytic, higher-pH
#                 activity makes ammonia (h has its own threshold pKaT, distinct
#                 from the urea-loss switch, because low-pH urea loss via k0 is
#                 non-ammoniagenic).
#   kv*fNH3(pH)*TAN  NH3 stripping through the open flasks; fNH3 is the free-NH3
#                 fraction (ammonium pKa 9.25), so it only matters at high pH.

# Nitrogen-budget note: measured TAN gain is far below the urea-N lost
# (apparent yield 0.13-0.35). Volatilisation is ruled out for the low/mid-pH
# doses (no free NH3 there), so most of the "missing" urea-N is non-ammoniagenic
# loss (the k0 route in Step 2), not stripping. The fitted kv is consequently
# small: volatilisation is a minor sink, confined to the high-pH control.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import step1_data_foundation as s1
import step2_urease_kinetics as s2

DAYS = s1.DAYS
UREA_MW = 60.06
N_PER_UREA = 2 * 14.007 / UREA_MW    # mg N per mg urea
pKa_amm = 9.25                       # ammonium/ammonia at 25 C

def fNH3(pH):
    return 1.0 / (1.0 + 10**(pKa_amm - pH))

def h_amm(pH, pKaT, nT):
    return 1.0 / (1.0 + 10**(nT*(pKaT - pH)))

def build_pH_drivers(key):
    d = s1.DATA[key]
    return [interp1d(DAYS, d["pH"][:, j], kind="linear", fill_value="extrapolate")
            for j in range(5)]

# Forward simulation of TAN given fixed Step 2 urea parameters and TAN parameters
def simulate_TAN(urea_par, tan_par, pH_of_t, U0, TAN0, t_eval):
    Vmax, mu, E0, pKa_lo, nf, pHg, sg, Km, k0 = urea_par
    kT, kv, pKaT, nT, b0 = tan_par
    def rhs(t, y):
        U, E, TAN = y
        pH = pH_of_t(t)
        dU = -Vmax*E*s2.f_act(pH, pKa_lo, nf)*U/(Km + U) - k0*U
        dE = mu*s2.g_grow(pH, pHg, sg)*E*(1.0 - E)
        dTAN = b0 + kT*E*h_amm(pH, pKaT, nT) - kv*fNH3(pH)*TAN
        return [dU, dE, dTAN]
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [U0, E0, TAN0], t_eval=t_eval,
                    method="LSODA", rtol=1e-7, atol=1e-6)
    return sol.y[2]

def residuals(tan_par, urea_par, key, drivers):
    d = s1.DATA[key]
    res = []
    for j in range(5):
        tan = simulate_TAN(urea_par, tan_par, drivers[j],
                           d["UREA"][0, j], d["TAN"][0, j], DAYS)
        sd = np.maximum(d["TAN_sd"][:, j], 10.0)
        res.extend((tan - d["TAN"][:, j]) / sd)
    return np.array(res)

TAN_NAMES = ["kT", "kv", "pKaT", "nT", "b0"]
LB = [10, 0, 4.5, 0.5, 0]
UB = [5000, 100, 7.5, 4.0, 120]
STARTS = [[300, 3, 5.8, 1.5, 20], [800, 10, 6.0, 2.0, 15],
          [1500, 20, 5.5, 1.0, 30], [200, 1, 6.5, 2.5, 25],
          [500, 5, 5.9, 1.2, 10]]

def background_release(key):
    # measured directly from the highest-dose treatment, where ureolysis is off
    d = s1.DATA[key]
    return (d["TAN"][-1, 4] - d["TAN"][0, 4]) / (DAYS[-1] - DAYS[0])

# Fitted TAN parameters, pinned to the converged best fit so that results are
# identical on every machine and SciPy version. Set REFIT = True to re-run the
# multistart optimisation that produced them.
REFIT = False
PINNED = {
    "CA": [170.207940, 0.028816, 5.765443, 0.500007, 20.546154],
    "WV": [184.500128, 0.611344, 6.721833, 1.718757, 23.923077],
}

def fit_TAN(key):
    urea_par, _ = s2.fit_acid(key)
    drivers = build_pH_drivers(key)
    if not REFIT:
        return urea_par, np.array(PINNED[key]), drivers
    b0 = background_release(key)
    def resid4(p):
        return residuals(np.array([p[0], p[1], p[2], p[3], b0]), urea_par, key, drivers)
    best = None
    for p0 in STARTS:
        r = least_squares(resid4, p0[:4], bounds=(LB[:4], UB[:4]),
                          method="trf", max_nfev=8000)
        if best is None or r.cost < best.cost:
            best = r
    return urea_par, np.array([best.x[0], best.x[1], best.x[2], best.x[3], b0]), drivers

# Nitrogen budget: urea-N lost vs TAN gained, and the ceiling on NH3 stripping
def nitrogen_budget(key):
    d = s1.DATA[key]
    print("\n" + d["meta"]["name"] + "  --  nitrogen budget (day1 -> day14)")
    print(f"{'dose':16s}{'ureaN_lost':>11s}{'TAN_gain':>9s}{'yield':>7s}"
          f"{'maxNH3frac':>11s}")
    for j, (lab, _, _, _, _) in enumerate(d["meta"]["treatments"]):
        ureaN = (d["UREA"][0, j] - d["UREA"][-1, j]) * N_PER_UREA
        tang = d["TAN"][-1, j] - d["TAN"][0, j]
        yld = tang/ureaN if ureaN > 0 else float("nan")
        frac = fNH3(d["pH"][:, j].max())
        print(f"  {lab:16s}{ureaN:11.0f}{tang:9.0f}{yld:7.2f}{frac:11.4f}")
    print("  yield = TAN gained / urea-N lost (would be ~1 if all loss made ammonia).")
    print("  maxNH3frac = free-NH3 fraction at peak pH (ceiling on volatilisation).")

def report_and_plot(key):
    d = s1.DATA[key]
    urea_par, tan_par, drivers = fit_TAN(key)
    print("\n" + d["meta"]["name"])
    print("  " + "  ".join(f"{n}={v:.4g}" for n, v in zip(TAN_NAMES, tan_par))
          + f"   (b0 measured: {background_release(key):.2f} mg N/L/d)")
    for j in range(5):
        tan = simulate_TAN(urea_par, tan_par, drivers[j],
                           d["UREA"][0, j], d["TAN"][0, j], DAYS)
        obs = d["TAN"][:, j]
        rmse = np.sqrt(np.mean((tan - obs)**2))
        print(f"    {d['meta']['treatments'][j][0]:16s} RMSE={rmse:5.0f} mg/L")

    plt.figure(figsize=(7, 5))
    t_smooth = np.linspace(1, 14, 90)
    for j in range(5):
        label, _, _, colour, opt = d["meta"]["treatments"][j]
        lab = label + " (optimum)" if opt else label
        tan = simulate_TAN(urea_par, tan_par, drivers[j],
                           d["UREA"][0, j], d["TAN"][0, j], t_smooth)
        plt.errorbar(DAYS, d["TAN"][:, j], yerr=d["TAN_sd"][:, j],
                     fmt=s1.MARKERS[j], color=colour, ms=4.5, markeredgewidth=0.5,
                     capsize=2, capthick=0.7, elinewidth=0.7,
                     ecolor=to_rgba(colour, 0.4), lw=0)
        plt.plot(t_smooth, tan, "-", color=colour, lw=1.4, label=lab)
    plt.xlabel("Days"); plt.ylabel("TAN (mg/L)"); plt.xlim(0, 15)
    plt.title("Ammonia (TAN) fit  --  " + d["meta"]["name"])
    plt.legend(fontsize=8, frameon=False); plt.tight_layout()
    return tan_par

if __name__ == "__main__":
    for key in ("CA", "WV"):
        nitrogen_budget(key)
        report_and_plot(key)
    plt.show()
    print("\nStep 3 complete: TAN layer fitted; nitrogen budget reported.")
