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


is_treated = data["treated"] == 1
is_control = data["treated"] == 0

# this is the u1/u0 ways of computing the ATE that the book says

mu1_model = LogisticRegression(max_iter=1000)
mu1_model.fit(data.loc[is_treated, covariates], data.loc[is_treated, "alive_independent"])

# μ̂_(0): outcome model fit on CONTROL rows only
mu0_model = LogisticRegression(max_iter=1000)
mu0_model.fit(data.loc[is_control, covariates], data.loc[is_control, "alive_independent"])

# predict the response surfaces for EVERY unit (counterfactuals included)
mu1 = pd.Series(mu1_model.predict_proba(data[covariates])[:, 1], index=data.index)
mu0 = pd.Series(mu0_model.predict_proba(data[covariates])[:, 1], index=data.index)

# ---- propensity model  ê(X)  (predicts TREATMENT) ----
prop_model = LogisticRegression(max_iter=1000)
prop_model.fit(data[covariates], data["treated"])
e = pd.Series(prop_model.predict_proba(data[covariates])[:, 1], index=data.index)
print(f"propensity range: [{e.min():.3f}, {e.max():.3f}]")   # overlap check

W = data["treated"]
Y = data["alive_independent"]

print(W)

gamma_aipw = ( (mu1 - mu0)                       # outcome-model part: μ̂₁ − μ̂₀
             + (W * (Y - mu1) / e)                 # treated residual correction
             - ((1 - W) * (Y - mu0) / (1 - e)) )   # control residual correction

tau_aipw = gamma_aipw.mean()


# calculate variance from the gamma and tau aipw

n = len(data)
V_aipw = ((gamma_aipw - tau_aipw) ** 2).sum() / (n - 1)

print(V_aipw)

# confidence intervals

z = 1.96                              # the 95% normal quantile

se_aipw = math.sqrt(V_aipw / n)       # standard error
ci_lo   = tau_aipw - z * se_aipw
ci_hi   = tau_aipw + z * se_aipw

print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")