from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save files without a display
import matplotlib.pyplot as plt
import seaborn as sns

from equations.ate import trial_ate, transport_ate
from equations.variance import (standard_error, z_score, confidence_interval,
                                 trial_ate_variance, transport_ate_variance)

ROOT = Path(__file__).resolve().parent
STEPS = ROOT / "steps"
GRAPHS = ROOT / "graphs" / "results"

# column names -- must match what pipeline.py wrote into the all-steps CSV
SOURCE          = "source"
MU1, MU0        = "mu1", "mu0"
PROPENSITY      = "e_treatment"
WEIGHT          = "transport_weight"
GAMMA_TRIAL     = "gamma_trial"
GAMMA_TRANSPORT = "gamma_transport"

CONFIDENCE = 0.95

TRIAL_COLOR = "#9c2f26"      # IST-3 (trial)
TRANSPORT_COLOR = "#2d6b61"  # IST-1 (target)


def compute(data):
    """Return the two estimands as dicts of {name, ate, se, lo, hi, color}."""
    z = z_score(CONFIDENCE)

    trial_est = trial_ate(data, GAMMA_TRIAL, SOURCE)
    trial_se = standard_error(trial_ate_variance(data, SOURCE, MU1, MU0, PROPENSITY))
    trial_lo, trial_hi = confidence_interval(trial_est, trial_se, z)

    transport_est = transport_ate(data, GAMMA_TRANSPORT, SOURCE)
    transport_se = standard_error(transport_ate_variance(data, GAMMA_TRANSPORT, SOURCE))
    transport_lo, transport_hi = confidence_interval(transport_est, transport_se, z)

    return [
        {"name": "trial (IST-3)",  "ate": trial_est,     "se": trial_se,
         "lo": trial_lo,     "hi": trial_hi,     "color": TRIAL_COLOR},
        {"name": "transport (IST-1)", "ate": transport_est, "se": transport_se,
         "lo": transport_lo, "hi": transport_hi, "color": TRANSPORT_COLOR},
    ]


def print_table(results):
    pct = round(CONFIDENCE * 100)
    print(f"{'estimand':<18}{'ATE':>9}{'SE':>9}{'  ' + str(pct) + '% CI':>20}")
    for r in results:
        print(f"{r['name']:<18}{r['ate']:>+9.4f}{r['se']:>9.4f}   [{r['lo']:+.4f}, {r['hi']:+.4f}]")


def forest_plot(results, path):
    fig, ax = plt.subplots(figsize=(7.5, 3))
    for y, r in enumerate(results):
        ax.errorbar(r["ate"], y,
                    xerr=[[r["ate"] - r["lo"]], [r["hi"] - r["ate"]]],
                    fmt="o", color=r["color"], ecolor=r["color"],
                    capsize=5, markersize=10, lw=2.5)
    ax.axvline(0, color="0.5", ls="--", lw=1.2, zorder=0)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r["name"] for r in results])
    ax.set_ylim(-0.6, len(results) - 0.4)
    ax.set_xlabel("average treatment effect")
    ax.set_title(f"Trial vs transported effect ({round(CONFIDENCE*100)}% CI)")
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gamma_distributions(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    trial = data[data[SOURCE] == 1]
    sns.histplot(x=trial[GAMMA_TRIAL], bins=40, kde=True, color=TRIAL_COLOR, ax=axes[0])
    axes[0].axvline(trial[GAMMA_TRIAL].mean(), color="black", ls="--", lw=1.5)
    axes[0].set_title("trial gamma  (per trial patient)")
    axes[0].set_xlabel("gamma_trial")

    pop = data[SOURCE].map({1: "trial (IST-3)", 0: "target (IST-1)"})
    gt = data[GAMMA_TRANSPORT]
    lo, hi = gt.quantile([0.01, 0.99])
    sns.histplot(x=gt, hue=pop, bins=40, ax=axes[1], stat="density", common_norm=False,
                 palette={"trial (IST-3)": TRIAL_COLOR, "target (IST-1)": TRANSPORT_COLOR})
    axes[1].set_xlim(lo, hi)
    axes[1].set_title("transport gamma  (density, by population)")
    axes[1].set_xlabel("gamma_transport")

    fig.suptitle("Per-row AIPW contributions (gamma)")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def transport_weights(data, path):
    w = data.loc[data[SOURCE] == 1, WEIGHT]
    ess = (w.sum() ** 2) / (w ** 2).sum()
    design_effect = len(w) / ess

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(x=w, bins=40, color=TRANSPORT_COLOR, ax=ax)
    ax.set_title("Transport weights  w = (1 - pi) / pi  on trial rows")
    ax.set_xlabel("inverse-odds weight")
    ax.text(0.97, 0.95,
            f"n trial = {len(w)}\nESS = {ess:.0f}\ndesign effect = {design_effect:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid", context="talk")
    GRAPHS.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(STEPS / "ist_combined_all_steps.csv")
    results = compute(data)
    print_table(results)

    forest_plot(results, GRAPHS / "forest.png")
    gamma_distributions(data, GRAPHS / "gamma_distributions.png")
    transport_weights(data, GRAPHS / "transport_weights.png")
    print(f"saved 3 graphs to {GRAPHS}")


if __name__ == "__main__":
    main()