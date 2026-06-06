from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent             # this script lives in graphs/
DATA = HERE.parent / "steps" / "ist_combined_all_steps.csv"   # data is in ../steps/
OUT = HERE / "calibration_plots.png"               # save alongside this script

BINS = 15
data = pd.read_csv(DATA)
data["is_ist3"] = (data.source == 1).astype(int)
trial = data[data.source == 1]

panels = [
    (trial[trial.treated == 0], "mu0", "alive_independent", "mu0 (IST-3 control)", 1.0),
    (trial[trial.treated == 1], "mu1", "alive_independent", "mu1 (IST-3 treated)", 1.0),
    (data, "pi_membership", "is_ist3", "membership propensity", 0.5),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (df, pred, obs, title, hi) in zip(axes, panels):
    d = df[[pred, obs]].dropna()
    means = d.groupby(pd.qcut(d[pred], BINS, duplicates="drop"), observed=True).mean()
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(means[pred], means[obs], "o-")
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="predicted", ylabel="observed", title=title)

fig.tight_layout()
fig.savefig(OUT, dpi=120)