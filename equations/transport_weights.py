def inverse_odds_weights(data, source, membership, weight):

    out = data.copy()
    trial = out[source] == 1
    pi = out[membership]
    out[weight] = float("nan")
    out.loc[trial, weight] = (1 - pi[trial]) / pi[trial]
    return out


