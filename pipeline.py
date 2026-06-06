from pathlib import Path
import pandas as pd

from equations.membership_propensity import logistic_membership_propensity
from equations.transport_weights import inverse_odds_weights
from equations.outcome_model import logistic_treatment_outcome, logistic_control_outcome
from equations.treatment_propensity import logistic_treatment_propensity
from equations.aipw import trial_gamma, transport_gamma

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


def main():
    STEPS2.mkdir(exist_ok=True)
    data = (pd.read_csv(DATASETS / "ist_combined.csv")
              .dropna(subset=COVARIATES + [SOURCE])
              .reset_index(drop=True))
    print(f"loaded ist_combined.csv -> {len(data)} complete-case rows")

    data = logistic_membership_propensity(data, COVARIATES, SOURCE, PI)
    data = inverse_odds_weights(data, SOURCE, PI, WEIGHT)
    data = logistic_treatment_outcome(data, COVARIATES, OUTCOME, SOURCE, TREATMENT, MU1)
    data = logistic_control_outcome(data, COVARIATES, OUTCOME, SOURCE, TREATMENT, MU0)
    data = logistic_treatment_propensity(data, COVARIATES, SOURCE, TREATMENT, PROPENSITY)
    data = trial_gamma(data, SOURCE, TREATMENT, OUTCOME, MU1, MU0, PROPENSITY, GAMMA_TRIAL)
    data = transport_gamma(data, SOURCE, TREATMENT, OUTCOME, MU1, MU0, PROPENSITY, WEIGHT, GAMMA_TRANSPORT)

    out_path = STEPS2 / "ist_combined_all_steps.csv"
    data.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({data.shape[0]} rows, {data.shape[1]} cols)")
    print("columns:", list(data.columns))

    trial = data[SOURCE] == 1
    n_target = int((data[SOURCE] == 0).sum())
    trial_ate = data.loc[trial, GAMMA_TRIAL].mean()
    transport_tate = data[GAMMA_TRANSPORT].sum() / n_target
    print(f"trial ATE = mean({GAMMA_TRIAL}) over trial rows = {trial_ate:+.4f}   (effect in the trial population)")
    print(f"TATE      = sum({GAMMA_TRANSPORT})/n_target     = {transport_tate:+.4f}   (effect for the target population)")


if __name__ == "__main__":
    main()