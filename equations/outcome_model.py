import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def logistic_treatment_outcome(data, covariates, outcome, source, treatment, prediction):
    """Fit on rows where source==1 and treatment==1; write predictions for all
    rows into the `prediction` column."""
    arm = data[(data[source] == 1) & (data[treatment] == 1)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(arm[covariates], arm[outcome])
    out = data.copy()
    out[prediction] = model.predict_proba(data[covariates])[:, 1]
    return out


def logistic_control_outcome(data, covariates, outcome, source, treatment, prediction):
    """Fit on rows where source==1 and treatment==0; write predictions for all
    rows into the `prediction` column."""
    arm = data[(data[source] == 1) & (data[treatment] == 0)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(arm[covariates], arm[outcome])
    out = data.copy()
    out[prediction] = model.predict_proba(data[covariates])[:, 1]
    return out