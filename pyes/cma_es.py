from numpy import argsort, random, asarray, dot
from .must import must_desc_sequence

def CMAES(mean, cov, n_dimension, objective, n_iter, learn_rate, mu, lam, sigma=1, mean_weights=None, p_sigma=None, p_c=None, random_state=None):
    raise Exception('CMAES NOT IMPLEMENTED')
    if mean_weights is None:
        mean_weights = asarray([1.0 / mu]*mu)
    else:
        if len(mean_weights) != mu:
            raise Exception('length of the weighted average of mu selected points is not equal to {mu}'.format(mu=mu))
        if sum(mean_weights) != 1:
            raise Exception('the sum of the weighted average is not equal to 1')
        must_desc_sequence(mean_weights)

    if p_sigma is None:
        p_sigma = asarray([0]*n_dimension)
    else:
        if len(p_sigma) != n_dimension:
            raise Exception('the length of p_sigma is {len_p_sigma} != {n_dimension}'.format(len_p_sigma=len(p_sigma), n_dimension=n_dimension))

    if p_c is None:
        p_c = asarray([0]*n_dimension)
    else:
        if len(p_c) != n_dimension:
            raise Exception('the length of p_c is {len_p_c} != {n_dimension}'.format(len_p_c=len(p_c), n_dimension=n_dimension))


    rs = random.RandomState(random_state)
    # debug('sigma**2:', sigma**2)
    # debug('cov:', cov)
    # debug('sigma**2*cov:', sigma**2*cov)
    # debug(rs.multivariate_normal(mean, sigma**2*cov, lam))
    population = rs.multivariate_normal(mean, sigma**2*cov, lam)
    mu_weight = 1.0 / sum([w**2 for w in mean_weights])
    # debug('mu_weight:', mu_weight)

    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        # debug('scores:', scores)
        ranks = argsort(argsort(scores))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]

        # debug('selected:', selected)
        # debug('selected:', population[selected])

        # 更新 mean
        mean_copy = mean.copy()
        mean = dot(mean_weights, population[selected])
        # debug('epoch {epoch} mean:'.format(epoch=epoch), mean)


        

    pass