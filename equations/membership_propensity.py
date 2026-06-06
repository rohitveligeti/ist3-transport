import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def logistic_membership_propensity(data, covariates, source, prediction):
    """Fit P(source == 1 | X) on all rows; write predictions into the `prediction` column."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(data[covariates], data[source])
    out = data.copy()
    out[prediction] = model.predict_proba(data[covariates])[:, 1]
    return out