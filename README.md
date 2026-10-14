# SURF dose model
     [![DOI](https://zenodo.org/badge/1419442022.svg)](https://doi.org/10.5281/zenodo.23375193)
Model code and data for estimating the acid dose that stabilizes stored urine with citric acid or white vinegar, developed for the stabilized-urine recirculating flush (SURF) concept for mobile urine-diverting toilets.
The model couples urea-hydrolysis kinetics to a charge balance, so that pH is calculated from the acid dose rather than prescribed. It reproduces the 13-day storage experiments and estimates the minimum stabilizing dose (MSD) between the tested doses.

## Contents

| File | What it does | Output |
|---|---|---|
| `step1_data_foundation.py` | Measured pH, urea and TAN (days 1 to 14, mean and SD, n = 3) for both acids and four doses each | Data summary and plots of the measured curves |
| `step2_urease_kinetics.py` | Urea-hydrolysis kinetics driven by the measured pH | Fitted urea parameters and R² |
| `step3_ammonia_nitrogen.py` | TAN formation and nitrogen budget | Fitted TAN parameters and nitrogen balance |
| `step4_titration_v2.py` | Charge-balance model of pH versus acid dose, with sensitivity analysis | Tables S3 and S4 values; `Fig6_dose_pH.png` / `.pdf` |
| `step5_dose_optimisation.py` | Coupled model run over a range of doses; MSD estimate | MSD and OSD values; `Fig7_model_dose.png` / `.pdf` |
| `run_on_colab.ipynb` | Runs all steps in Google Colab | All of the above |

Each step imports the earlier ones, so keep all files in the same folder.

## How to run

**Google Colab:** open `run_on_colab.ipynb` from this repository in Colab and run all cells.

**On your own computer (Python 3.9 or newer):**

```
pip install -r requirements.txt
python step5_dose_optimisation.py
```

Any step can be run on its own in the same way. Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## Fitted parameters

The fitted parameters are fixed in the code (`PINNED` in steps 2 and 3, `CONS_PARAMS` in step 5), so results are identical on every machine. Set `REFIT = True` in steps 2 and 3 to repeat the fitting.

## Applying the model to other urine

The kinetic parameters were fitted to urine from one group of donors stored at 22–25 °C. Temperature is not a model input. To estimate the MSD for other conditions:

1. Replace the measured data in `step1_data_foundation.py` with results from a storage test of the local urine (pH, urea and TAN over the storage period, with an undosed control).
2. Set `REFIT = True` in `step2_urease_kinetics.py` and `step3_ammonia_nitrogen.py`.
3. Update the acid stock concentration in step 1 if a different acid strength is used.
4. Run `step5_dose_optimisation.py`. The MSD criterion is set by `PH_CRIT` in step 5.

Estimates outside the tested dose range, or for urine very different from the original, should be checked with a storage test.

## How to cite

If you use this code, please cite the article and the archived version of this repository:

- [Article citation to be added after publication]
[Ul Rahman, S. (2026). SURF dose model (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23375194]
## License

Code: MIT License (see `LICENSE`).
