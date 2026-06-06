import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def logistic_treatment_propensity(data, covariates, source, treatment, prediction):
    """Fit P(treatment == 1 | X) on rows where source==1; write predictions into
    `prediction` for those rows, NaN elsewhere."""
    trial = data[data[source] == 1]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(trial[covariates], trial[treatment])
    out = data.copy()
    out[prediction] = float("nan")
    out.loc[out[source] == 1, prediction] = model.predict_proba(trial[covariates])[:, 1]
    return out