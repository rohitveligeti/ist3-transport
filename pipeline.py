from pathlib import Path
import inspect
import pandas as pd

from equations import (membership_propensity, transport_weights, outcome_model,
                       treatment_propensity, aipw)
from equations.ate import trial_ate, transport_ate
from equations.variance import (standard_error, z_score, confidence_interval,
                                 trial_ate_variance, transport_ate_variance)

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
STEPS2 = ROOT / "steps"

# the caller owns every column name -- the equations/ functions hold none
COVARIATES = ["age", "sbp", "male", "atrial_fib", "visible_infarct"]
OUTCOME    = "alive_independent"
SOURCE     = "source"
TREATMENT  = "treated"
PI         = "pi_membership"
WEIGHT     = "transport_weight"
MU1, MU0   = "mu1", "mu0"
PROPENSITY = "e_treatment"
GAMMA_TRIAL     = "gamma_trial"
GAMMA_TRANSPORT = "gamma_transport"

STEPS = [
    {"label": "membership propensity",
     "module": membership_propensity, "match": lambda n: True,
     "call": lambda fn, d: fn(d, COVARIATES, SOURCE, PI)},
    {"label": "transport weights",
     "module": transport_weights, "match": lambda n: True,
     "call": lambda fn, d: fn(d, SOURCE, PI, WEIGHT)},
    {"label": "treated outcome model (mu1)",
     "module": outcome_model, "match": lambda n: n.endswith("_treatment_outcome"),
     "call": lambda fn, d: fn(d, COVARIATES, OUTCOME, SOURCE, TREATMENT, MU1)},
    {"label": "control outcome model (mu0)",
     "module": outcome_model, "match": lambda n: n.endswith("_control_outcome"),
     "call": lambda fn, d: fn(d, COVARIATES, OUTCOME, SOURCE, TREATMENT, MU0)},
    {"label": "treatment propensity",
     "module": treatment_propensity, "match": lambda n: True,
     "call": lambda fn, d: fn(d, COVARIATES, SOURCE, TREATMENT, PROPENSITY)},
    {"label": "trial gamma",
     "module": aipw, "match": lambda n: n == "trial_gamma",
     "call": lambda fn, d: fn(d, SOURCE, TREATMENT, OUTCOME, MU1, MU0, PROPENSITY, GAMMA_TRIAL)},
    {"label": "transport gamma",
     "module": aipw, "match": lambda n: n == "transport_gamma",
     "call": lambda fn, d: fn(d, SOURCE, TREATMENT, OUTCOME, MU1, MU0, PROPENSITY, WEIGHT, GAMMA_TRANSPORT)},
]


def options_for(module, match):
    """Public functions actually defined in `module` whose name passes `match`."""
    return [name for name, fn in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_")
            and getattr(fn, "__module__", None) == module.__name__
            and match(name)]


def default_choice(options):
    for o in options:
        if "logistic" in o:
            return o
    return options[0] if options else ""


def run_pipeline(selection, filename, log=print):
    """selection: {step label -> chosen function name}. Runs the steps in order,
    writes the named CSV, reports the estimands, and returns the resulting frame."""
    data = (pd.read_csv(DATASETS / "ist_combined.csv")
              .dropna(subset=COVARIATES + [SOURCE]).reset_index(drop=True))
    log(f"loaded ist_combined.csv -> {len(data)} complete-case rows")
    for step in STEPS:
        name = selection[step["label"]]
        log(f"  {step['label']}: {name}")
        data = step["call"](getattr(step["module"], name), data)

    filename = (filename or "").strip() or "ist_combined_all_steps.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"
    STEPS2.mkdir(exist_ok=True)
    data.to_csv(STEPS2 / filename, index=False)
    log(f"wrote {STEPS2 / filename}  ({data.shape[0]} rows, {data.shape[1]} cols)")
    return data


def _estimands(data):
    z = z_score(0.95)
    te = trial_ate(data, GAMMA_TRIAL, SOURCE)
    tlo, thi = confidence_interval(te, standard_error(
        trial_ate_variance(data, SOURCE, MU1, MU0, PROPENSITY)), z)
    pe = transport_ate(data, GAMMA_TRANSPORT, SOURCE)
    plo, phi = confidence_interval(pe, standard_error(
        transport_ate_variance(data, GAMMA_TRANSPORT, SOURCE)), z)
    return (te, tlo, thi), (pe, plo, phi)


def make_figure(data):
    """One white figure: 3 calibration curves (mu0, mu1, pi) over a forest plot."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 6.6), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.7])

    d = data.copy()
    d["is_ist3"] = (d[SOURCE] == 1).astype(int)
    trial = d[d[SOURCE] == 1]
    panels = [
        (fig.add_subplot(gs[0, 0]), trial[trial[TREATMENT] == 0], MU0, OUTCOME, "mu0 (control)", 1.0),
        (fig.add_subplot(gs[0, 1]), trial[trial[TREATMENT] == 1], MU1, OUTCOME, "mu1 (treated)", 1.0),
        (fig.add_subplot(gs[0, 2]), d, PI, "is_ist3", "membership pi", 0.5),
    ]
    for ax, df, pred, obs, title, hi in panels:
        dd = df[[pred, obs]].dropna()
        means = dd.groupby(pd.qcut(dd[pred], 15, duplicates="drop"), observed=True).mean()
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        ax.plot(means[pred], means[obs], "o-", color="#1f77b4", ms=4)
        ax.set(xlim=(0, hi), ylim=(0, hi), title=title)
        ax.set_xlabel("predicted"); ax.set_ylabel("observed")
        ax.set_facecolor("white")

    (te, tlo, thi), (pe, plo, phi) = _estimands(d)
    axf = fig.add_subplot(gs[1, :])
    for y, (name, est, lo, hi_, c) in enumerate(
            [("trial", te, tlo, thi, "#9c2f26"), ("transport", pe, plo, phi, "#2d6b61")]):
        axf.errorbar(est, y, xerr=[[est - lo], [hi_ - est]], fmt="o",
                     color=c, capsize=5, markersize=9, lw=2)
    axf.axvline(0, color="0.6", ls="--", lw=1)
    axf.set_yticks([0, 1]); axf.set_yticklabels(["trial", "transport"])
    axf.set_ylim(-0.6, 1.6); axf.set_xlabel("ATE (risk difference)")
    axf.set_facecolor("white")
    axf.set_title(f"trial {te:+.3f} [{tlo:+.3f}, {thi:+.3f}]      "
                  f"transport {pe:+.3f} [{plo:+.3f}, {phi:+.3f}]")
    return fig


def build_gui():
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("IST pipeline builder")
    root.configure(bg="white")

    # plain white look, independent of the OS (macOS dark mode won't bleed in)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    for el in (".", "TFrame", "TLabel", "TButton"):
        style.configure(el, background="white", foreground="black")
    style.configure("TCombobox", fieldbackground="white", background="white")
    style.configure("TEntry", fieldbackground="white")

    controls = ttk.Frame(root, padding=10)
    controls.pack(side="top", fill="x")
    plots = tk.Frame(root, bg="white")
    plots.pack(side="top", fill="both", expand=True)

    choices = {}
    for i, step in enumerate(STEPS):
        opts = options_for(step["module"], step["match"])
        ttk.Label(controls, text=step["label"]).grid(row=i, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=default_choice(opts))
        ttk.Combobox(controls, textvariable=var, values=opts, state="readonly",
                     width=34).grid(row=i, column=1, padx=8, pady=2)
        choices[step["label"]] = var

    r = len(STEPS)
    ttk.Label(controls, text="output filename").grid(row=r, column=0, sticky="w", pady=2)
    fname = tk.StringVar(value="ist_combined_all_steps.csv")
    ttk.Entry(controls, textvariable=fname, width=36).grid(row=r, column=1, padx=8, pady=2)

    status = ttk.Label(controls, text="")
    status.grid(row=r + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def on_run():
        status.config(text="running ...")
        root.update_idletasks()
        try:
            selection = {label: var.get() for label, var in choices.items()}
            data = run_pipeline(selection, fname.get(), log=print)
            fig = make_figure(data)
            for w in plots.winfo_children():
                w.destroy()
            canvas = FigureCanvasTkAgg(fig, master=plots)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            status.config(text="done.")
        except Exception as e:
            status.config(text=f"ERROR: {e}")

    ttk.Button(controls, text="Run pipeline", command=on_run).grid(
        row=r + 2, column=0, columnspan=2, pady=8)

    root.mainloop()


if __name__ == "__main__":
    build_gui()