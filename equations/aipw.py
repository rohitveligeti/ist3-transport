def trial_gamma(data, source, treatment, outcome, mu1, mu0, propensity, output):
    """
    Per-row trial AIPW score (effect in the trial population), into `output`.
      trial  rows (source == 1):  (mu1 - mu0) + A*(Y-mu1)/e - (1-A)*(Y-mu0)/(1-e)
      target rows (source == 0):  NaN
    The trial ATE is gamma[trial].mean(); its variance is gamma[trial].var() / n_trial.
    """
    out = data.copy()
    trial = out[source] == 1
    A, Y = out[treatment], out[outcome]
    m1, m0, e = out[mu1], out[mu0], out[propensity]

    out[output] = float("nan")
    out.loc[trial, output] = (
        (m1[trial] - m0[trial])
        + A[trial] * (Y[trial] - m1[trial]) / e[trial]
        - (1 - A[trial]) * (Y[trial] - m0[trial]) / (1 - e[trial])
    )
    return out


def transport_gamma(data, source, treatment, outcome, mu1, mu0, propensity, weight, output):
    """
    Per-row transport AIPW contribution (effect for the target population), into `output`.
      target rows (source == 0):  mu1 - mu0
      trial  rows (source == 1):  weight * ( A*(Y-mu1)/e - (1-A)*(Y-mu0)/(1-e) )
    The transported effect is gamma.sum() / n_target; its variance comes from the
    spread of this column.
    """
    out = data.copy()
    trial = out[source] == 1
    target = out[source] == 0
    A, Y = out[treatment], out[outcome]
    m1, m0 = out[mu1], out[mu0]
    e, w = out[propensity], out[weight]

    out[output] = float("nan")
    out.loc[target, output] = m1[target] - m0[target]
    out.loc[trial, output] = w[trial] * (A[trial] * (Y[trial] - m1[trial]) / e[trial]
                                         - (1 - A[trial]) * (Y[trial] - m0[trial]) / (1 - e[trial]))
    return out