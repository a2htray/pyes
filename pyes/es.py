from .must import must_selection_strategy
from .utils import in_bounds, init_solution, debug
from .strategy import MU_COMMA_LAMBDA
from numpy import argsort, random

# evolution strategy (mu, lambda) algorithm
def _es_comma(n_dimension, objective, bounds, n_iter, learn_rate, mu, lam, random_state=None):
    rs = random.RandomState(random_state)
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    
    population = [init_solution(rs, n_dimension, bound_lows=bounds[0], bound_highs=bounds[1]) for _ in range(lam)]

    debug(population)

    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        children = list()
        
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                debug('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))

            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + rs.randn(n_dimension) * learn_rate
                
                children.append(child)

        population = children
    return [best, best_eval]


# evolution strategy (mu + lambda) algorithm
def _es_plus(n_dimension, objective, bounds, n_iter, learn_rate, mu, lam, random_state=None):
    rs = random.RandomState(random_state)
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)

    population = [init_solution(rs, n_dimension, bound_lows=bounds[0], bound_highs=bounds[1]) for _ in range(lam)]

    debug(population)

    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        children = list()

        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                debug('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))

            # keep the parent
            children.append(population[i])

            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + rs.randn(n_dimension) * learn_rate
                
                children.append(child)

        population = children
    return [best, best_eval]


def ES(comma_or_plus: str, n_dimension, objective, bounds, n_iter, learn_rate, mu, lam, random_state=None):
    must_selection_strategy(comma_or_plus)
    if comma_or_plus == MU_COMMA_LAMBDA:
        return _es_comma(n_dimension, objective, bounds, n_iter, learn_rate, mu, lam, random_state)
    else:
        return _es_plus(n_dimension, objective, bounds, n_iter, learn_rate, mu, lam, random_state)
