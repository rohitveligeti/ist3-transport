def trial_ate(data, gamma, source):
    """Trial ATE = mean of the trial gamma scores over the trial rows.
    (The trial gamma is NaN on target rows.)"""
    trial = data[source] == 1
    n = int(trial.sum())
    return data.loc[trial, gamma].sum() / n


def transport_ate(data, gamma, source):
    """Transported ATE = sum of the transport gamma over all rows, divided by n_target."""
    n_target = int((data[source] == 0).sum())
    return data[gamma].sum() / n_target