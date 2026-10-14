# Step 4 (revised): charge-balance model of pH versus acid dose.
#
# Changes from the original step4_titration.py
#   1. Phosphate set to 23.9 mmol/L (740 mg P/L), Udert et al. (2006) [ref 17].
#      Total inorganic carbon kept at 20 mmol/L, the upper end of values
#      reported for fresh urine (236 mg C/L, Front. Environ. Sci. 2022, 998743).
#   2. All urine constituents, INCLUDING the strong-ion difference (SID), are
#      diluted by the acid volume added (the original code diluted the buffers
#      but not SID).
#   3. Adds the sensitivity analyses reported in Table S3 (Davies activity
#      correction; phosphate; inorganic carbon) and the rebound check in
#      Table S4 (day-14 pH from measured day-14 TAN, added acid conserved).
#
# Only SID is calibrated, once per series, to the measured day-1 control pH.
# Every dosed day-1 pH is a prediction.
#
# Requires step1_data_foundation.py in the same folder.
# Output: Table S3 and Table S4 values printed below; dose-pH figure (manuscript
#         Fig. 6) shown inline and saved as Fig6_dose_pH.png (600 dpi) and .pdf.

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import step1_data_foundation as s1

# ---- constants (25 C) --------------------------------------------------------
PKW = 14.0
PKA = {"NH4": [9.25], "CO3": [6.35, 10.33], "PO4": [2.15, 7.20, 12.35],
       "CIT": [3.13, 4.76, 6.40], "AC": [4.76]}
Z_FIRST = {"NH4": 1, "CO3": 0, "PO4": 0, "CIT": 0, "AC": 0}  # charge of fully protonated form
MW_P, MW_N = 30.974, 14.007

BASE = {"TAN": 503 / MW_N / 1000,     # measured day-1 control TAN (overwritten per series)
        "PO4": 740 / MW_P / 1000,     # 23.9 mmol/L, ref [17]
        "CO3": 0.020}                 # 20 mmol/L, refs [29,30]

# ---- activity (Davies, 25 C) -------------------------------------------------
def gamma(z, I):
    if I <= 0:
        return 1.0
    s = np.sqrt(I)
    return 10 ** (-0.509 * z * z * (s / (1 + s) - 0.3 * I))

def conditional_pKas(species, I):
    """Concentration-based pKa: K' = K * g(HA) / (g(H) * g(A))."""
    out, z = [], Z_FIRST[species]
    for pk in PKA[species]:
        K = 10 ** (-pk) * gamma(z, I) / (gamma(1, I) * gamma(z - 1, I))
        out.append(-np.log10(K))
        z -= 1
    return out

def fractions(H, pKas):
    terms, c = [1.0], 1.0
    for pk in pKas:
        c *= 10 ** (-pk) / H
        terms.append(c)
    t = np.array(terms)
    return t / t.sum()

def anion_charge(H, total, pKas):
    a = fractions(H, pKas)
    return total * sum(i * a[i] for i in range(len(a)))

# ---- charge balance (Eq. 7) --------------------------------------------------
def net_charge(pH, SID, comp, acid, dose, I=0.0):
    g1 = gamma(1, I)
    H = 10 ** (-pH) / g1                       # measured pH treated as activity
    OH = 10 ** (-PKW) / (g1 * g1) / H
    NH4 = fractions(H, conditional_pKas("NH4", I))[0] * comp["TAN"]
    neg = (anion_charge(H, comp["CO3"], conditional_pKas("CO3", I))
           + anion_charge(H, comp["PO4"], conditional_pKas("PO4", I)))
    if acid in ("CIT", "AC") and dose > 0:
        neg += anion_charge(H, dose, conditional_pKas(acid, I))
    return SID + NH4 + H - OH - neg

def solve_pH(SID, comp, acid, dose, I=0.0):
    return brentq(lambda p: net_charge(p, SID, comp, acid, dose, I), 1.0, 13.0, xtol=1e-6)

def calibrate_SID(pH_control, comp, I=0.0):
    return brentq(lambda S: net_charge(pH_control, S, comp, "none", 0.0, I), -1, 1, xtol=1e-10)

def diluted(comp, f):
    return {k: v * f for k, v in comp.items()}

# ---- series runner -----------------------------------------------------------
def run_series(key, PO4=None, CO3=None, I=0.0):
    d = s1.DATA[key]; m = d["meta"]
    acid = "CIT" if key == "CA" else "AC"
    comp = dict(BASE)
    comp["TAN"] = d["TAN"][0, 0] / MW_N / 1000
    if PO4 is not None: comp["PO4"] = PO4
    if CO3 is not None: comp["CO3"] = CO3
    SID = calibrate_SID(d["pH"][0, 0], comp, I)
    rows = []
    for j, (label, mL, pct, *_ ) in enumerate(m["treatments"]):
        if mL == 0:
            continue
        f = 200.0 / (200.0 + mL)
        dose = s1._final_mM(m["stock_M"], mL) / 1000
        c = diluted(comp, f)
        p1_pred = solve_pH(SID * f, c, acid, dose, I)
        # rebound check: measured TAN on day 1 and day 14, acid conserved
        c1 = dict(c); c1["TAN"] = d["TAN"][0, j] / MW_N / 1000
        c14 = dict(c); c14["TAN"] = d["TAN"][-1, j] / MW_N / 1000
        q1 = solve_pH(SID * f, c1, acid, dose, I)
        q14 = solve_pH(SID * f, c14, acid, dose, I)
        rows.append(dict(dose_pct=pct, pH1_meas=d["pH"][0, j], pH1_pred=p1_pred,
                         pH14_meas=d["pH"][-1, j], pH14_calc=q14,
                         rise_meas=d["pH"][-1, j] - d["pH"][0, j], rise_calc=q14 - q1))
    diff = np.array([r["pH1_pred"] - r["pH1_meas"] for r in rows])
    return rows, float(np.sqrt(np.mean(diff ** 2))), float(diff.min()), float(diff.max())

# ---- main --------------------------------------------------------------------
if __name__ == "__main__":
    name = {"CA": "Citric acid", "WV": "Vinegar"}

    print("Baseline: PO4 23.9 mM, TIC 20 mM, ideal solution")
    base = {}
    for key in ("CA", "WV"):
        rows, rmse, lo, hi = run_series(key)
        base[key] = rows
        print(f"  {name[key]:12s} day-1 RMSE {rmse:.2f}  offsets {lo:+.2f} to {hi:+.2f}")

    # Table S3: sensitivity
    cases = [("Baseline (ideal solution; PO4 23.9 mM; TIC 20 mM)", {}),
             ("Davies correction, I = 0.1 M", {"I": 0.1}),
             ("Davies correction, I = 0.2 M", {"I": 0.2}),
             ("Davies correction, I = 0.3 M", {"I": 0.3}),
             ("PO4 18.2 mM", {"PO4": 565 / MW_P / 1000}),
             ("TIC 9 mM", {"CO3": 0.009}),
             ("TIC 2.3 mM", {"CO3": 0.0023})]
    print("\nTable S3  (RMSE: citric acid, vinegar)")
    for label, kw in cases:
        r_ca = run_series("CA", **kw)[1]; r_wv = run_series("WV", **kw)[1]
        print(f"  {label:52s} {r_ca:.2f}  {r_wv:.2f}")

    # Table S4: rebound check, plus share of rise explained across the TIC range
    print("\nTable S4")
    for key in ("CA", "WV"):
        for r in base[key]:
            lab = f"{name[key]} {r['dose_pct']:g}% v/v"
            print(f"  {lab:24s} measured {r['rise_meas']:.2f}  calculated {r['rise_calc']:.2f}")
    print("\nVinegar rebound explained across TIC 2.3-20 mM:")
    for dose in (5, 10):
        shares = []
        for tic in (0.0023, 0.009, 0.020):
            r = [x for x in run_series("WV", CO3=tic)[0] if x["dose_pct"] == dose][0]
            shares.append(100 * r["rise_calc"] / r["rise_meas"])
        print(f"  {dose}% v/v: {min(shares):.0f}-{max(shares):.0f}%")

    # Dose-pH figure: measured vs predicted day-1 pH
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, key, tag in ((axes[0], "WV", "a"), (axes[1], "CA", "b")):
        d = s1.DATA[key]
        x = [0.0] + [r["dose_pct"] for r in base[key]]
        meas = [d["pH"][0, 0]] + [r["pH1_meas"] for r in base[key]]
        pred = [d["pH"][0, 0]] + [r["pH1_pred"] for r in base[key]]
        ax.plot(x, meas, "o-", color="#333333", ms=5, lw=1.2, label="Measured (day 1)")
        ax.plot(x, pred, "s--", color="#c0392b", ms=5, lw=1.2, label="Predicted (charge balance)")
        ax.set_xlabel("Dose (% v/v)"); ax.set_ylabel("pH")
        ax.text(-0.18, 1.02, tag, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.tick_params(direction="in"); ax.set_ylim(2.5, 7.0)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("Fig6_dose_pH.png", dpi=600)    # manuscript Fig. 6 (charge balance)
    fig.savefig("Fig6_dose_pH.pdf")
    plt.show()
    print("\nStep 4 complete: Table S3 and S4 values printed; "
          "Fig6_dose_pH.png and Fig6_dose_pH.pdf saved.")
