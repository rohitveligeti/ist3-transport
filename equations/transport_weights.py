def inverse_odds_weights(data, source, membership, weight):

    out = data.copy()
    trial = out[source] == 1
    pi = out[membership]
    out[weight] = float("nan")
    out.loc[trial, weight] = (1 - pi[trial]) / pi[trial]
    return out


def clipped_inverse_odds_weights(data, source, membership, weight, quantile=0.99):
    out = data.copy()
    trial = out[source] == 1
    pi = out[membership]
    w = (1 - pi[trial]) / pi[trial]
    out[weight] = float("nan")
    out.loc[trial, weight] = w.clip(upper=w.quantile(quantile))
    return out