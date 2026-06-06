

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "ist_combined_with_membership.csv"
OUTPUT_PNG = "covariate_distributions_redo.png"

CONTINUOUS = ["age", "delay_hours", "sbp", "pi_membership"]
BINARY = ["male", "atrial_fib", "visible_infarct"]
VARIABLES = ["age", "delay_hours", "sbp", "male", "atrial_fib", "visible_infarct", "pi_membership"]

IST3_COLOR = "#da2f20"   # red  = IST-3 (source = 1)
IST1_COLOR = "#2d6b61"   # teal = IST-1 (source = 0)


def plot_continuous(ax, data, col):
    values = data[col].dropna()
    bins = np.linspace(values.min(), values.max(), 31)
    ax.hist(data.loc[data.source == 1, col].dropna(), bins=bins, density=True,
            alpha=0.55, color=IST3_COLOR, label="IST-3")
    ax.hist(data.loc[data.source == 0, col].dropna(), bins=bins, density=True,
            alpha=0.55, color=IST1_COLOR, label="IST-1")
    ax.set_xlabel(col)
    ax.set_ylabel("density")


def plot_binary(ax, data, col):
    prop3 = data.loc[data.source == 1, col].value_counts(normalize=True).reindex([0, 1]).fillna(0)
    prop1 = data.loc[data.source == 0, col].value_counts(normalize=True).reindex([0, 1]).fillna(0)
    x = np.array([0, 1]); w = 0.38
    ax.bar(x - w/2, prop3.values, w, color=IST3_COLOR, label="IST-3")
    ax.bar(x + w/2, prop1.values, w, color=IST1_COLOR, label="IST-1")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0 (no)", "1 (yes)"])
    ax.set_xlabel(col)
    ax.set_ylabel("proportion")


def main():
    data = pd.read_csv(DATA_PATH)
    n_trial = int((data.source == 1).sum())
    n_target = int((data.source == 0).sum())

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    axes = axes.ravel()
    for ax, col in zip(axes, VARIABLES):
        (plot_continuous if col in CONTINUOUS else plot_binary)(ax, data, col)
        ax.set_title(col, fontweight="bold")
        ax.legend()
    for ax in axes[len(VARIABLES):]:      # hide unused cells
        ax.axis("off")

    fig.suptitle(f"Covariate distributions: IST-3 (n={n_trial}) vs IST-1 (n={n_target})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    print(f"saved -> {OUTPUT_PNG}")
    # plt.show()   # uncomment to view interactively


if __name__ == "__main__":
    main()