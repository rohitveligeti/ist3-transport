from statistics import NormalDist


# ----- inference toolkit -----

def standard_error(variance):
    """SE = sqrt(variance)."""
    return variance ** 0.5


def z_score(confidence=0.95):
    """Two-sided normal critical value for a confidence level (0.95 -> ~1.95996)."""
    return NormalDist().inv_cdf((1 + confidence) / 2)


def confidence_interval(estimate, se, z):
    """(low, high) = estimate +/- z * se."""
    return estimate - z * se, estimate + z * se


# ----- 2 variance computations -----

def transport_ate_variance(data, gamma, source):

    target = data[source] == 0
    trial = data[source] == 1
    n_target = int(target.sum())
    g = data[gamma]
    ate = g.sum() / n_target
 
    def standard_variance():
        return ((g[target] - ate) ** 2).sum() / n_target ** 2
 
    def trial_group_noise():
        return (g[trial] ** 2).sum() / n_target ** 2
 
    return standard_variance() + trial_group_noise()


def trial_ate_variance(data, source, mu1, mu0, propensity):

    trial = data[data[source] == 1]
    n = len(trial)
    m1, m0, e = trial[mu1], trial[mu0], trial[propensity]

    def standard_variance():
        return (m1 - m0).var()

    def treated_outcome_noise():
        sigma2 = m1 * (1 - m1)
        return (sigma2 / e).mean()

    def control_outcome_noise():
        sigma2 = m0 * (1 - m0)
        return (sigma2 / (1 - e)).mean()

    aipw_variance = standard_variance() + treated_outcome_noise() + control_outcome_noise()
    return aipw_variance / n