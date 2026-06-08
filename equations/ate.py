def trial_ate(data, gamma, source):

    trial = data[source] == 1
    n = int(trial.sum())
    return data.loc[trial, gamma].sum() / n


def transport_ate(data, gamma, source):

    n_target = int((data[source] == 0).sum())
    return data[gamma].sum() / n_target