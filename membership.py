"""
Transportability, step 1: the TRIAL-MEMBERSHIP (participation) model.

Predict whether a patient is in IST-3 (source, S=1) vs IST-1 (target, S=0)
from the shared covariates, using logistic regression. The fitted scores
    pi(X) = P(S = 1 | X)
are the participation probabilities used to build transport weights later.

NOTE: delay_hours is deliberately EXCLUDED here. It nearly separates the two
trials (IST-3 <=6h vs IST-1 <=48h), which forces pi(X) to 0/1 and breaks
positivity. Dropping it means we transport over the covariates that actually
overlap. (delay_hours is still kept as a column in the output for reference.)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

DATA_PATH = "ist_combined.csv"
COVARIATES = ["age", "sbp", "male", "atrial_fib", "visible_infarct"]   # delay_hours dropped


def step1_load(path, covariates):
    data = pd.read_csv(path)
    before = len(data)
    # the membership model needs complete covariates (e.g. IST-1 pilot AF is missing)
    data = data.dropna(subset=covariates + ["source"]).reset_index(drop=True)
    n_trial = int((data["source"] == 1).sum())
    n_target = int((data["source"] == 0).sum())
    print(f"STEP 1  rows {before} -> {len(data)} complete-case  "
          f"(IST-3 trial={n_trial}, IST-1 target={n_target})")
    return data


def step2_fit_membership(data, covariates):
    X = data[covariates]
    S = data["source"]                      # 1 = IST-3 (trial), 0 = IST-1 (target)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, S)
    coefs = dict(zip(covariates, model.named_steps["logisticregression"].coef_[0].round(3)))
    print("STEP 2  logistic membership model fitted (delay_hours excluded)")
    print(f"        standardized coefficients: {coefs}")
    return model


def step3_membership_probability(model, data, covariates):
    pi = pd.Series(model.predict_proba(data[covariates])[:, 1], index=data.index)
    auc = roc_auc_score(data["source"], pi)
    print(f"STEP 3  pi(X) = P(in IST-3):  min={pi.min():.3f}  mean={pi.mean():.3f}  max={pi.max():.3f}")
    print(f"        mean pi among IST-3 = {pi[data.source==1].mean():.3f}   "
          f"among IST-1 = {pi[data.source==0].mean():.3f}")
    print(f"        AUC (how separable the two trials are) = {auc:.3f}")
    if auc > 0.90:
        print("        ** AUC still very high -> overlap problem persists beyond delay **")
    return pi


def main():
    data = step1_load(DATA_PATH, COVARIATES)
    model = step2_fit_membership(data, COVARIATES)
    pi = step3_membership_probability(model, data, COVARIATES)
    data["pi_membership"] = pi
    data.to_csv("ist_combined_with_membership_redo.csv", index=False)
    print("saved -> ist_combined_with_membership_redo.csv")


if __name__ == "__main__":
    main()