import pandas as pd
import math
from sklearn.linear_model import LogisticRegression

data_path = "df.sas7bdat"
covariates = ["age", "nihss", "randdelay"]


raw_data = pd.read_sas(data_path)

data = pd.DataFrame()
data["treated"] = (pd.to_numeric(raw_data["itt_treat"], errors="coerce") == 0).astype("Int64")
data["alive_independent"] = (pd.to_numeric(raw_data["aliveind6"], errors="coerce") == 1).astype("Int64")

for covariate in covariates:
    data[covariate] = pd.to_numeric(raw_data[covariate], errors="coerce")

data = data.dropna().reset_index(drop=True)
data["treated"] = data["treated"].astype(int)
data["alive_independent"] = data["alive_independent"].astype(int)

n_treated = int((data["treated"] == 1).sum())
n_control = int((data["treated"] == 0).sum())

# DATA PROCESSING DONE, PROPENSITY MODEL HERE

model = LogisticRegression(max_iter=1000)
model.fit(data[covariates], data["treated"])

prob_treated = model.predict_proba(data[covariates])[:, 1]

propensity = pd.Series(prob_treated, index=data.index)

# horvitz-thompson estimator

is_treated = data["treated"] == 1
is_control = data["treated"] == 0

weight = pd.Series(0.0, index=data.index)
weight[is_treated] = 1 / propensity[is_treated]
weight[is_control] = 1 / (1 - propensity[is_control])

for covariate in covariates:

    # the book uses horvitz-thompson estimator, but claude is telling me to use hajek–I kept the horvitz-thompson estimator here
    #hajek is divided by weights instead of len(data)

    raw_mean_treated = data.loc[is_treated, covariate].mean()
    raw_mean_control = data.loc[is_control, covariate].mean()

    weighted_mean_treated = (weight * data[covariate])[is_treated].sum() / len(data)
    weighted_mean_control = (weight * data[covariate])[is_control].sum() / len(data)

outcome = data["alive_independent"]

mean_outcome_treated = (weight * outcome)[is_treated].sum() / len(data)
mean_outcome_control = (weight * outcome)[is_control].sum() / len(data)

print(mean_outcome_treated, mean_outcome_control)

tau_ht = mean_outcome_treated - mean_outcome_control      
print(f"Tau Horvitz-Thompson IPW: {tau_ht:.3f}")

# The book didnt give me a variance formula/confidence intervals to use for the IPW, so I will end the ipw here

