# requirements:
#   sklearn 0.24.2
import numpy as np
from loguru import logger
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pyes


objective = pyes.OBJECTIVE_FUNS['square_sum']

def event_on_generation(gen, population, current_best, history_best):
    logger.info('{gen} iter: current_best={current_best}, history_best={history_best}, fitness={fitness}'.format(gen=gen, current_best=current_best, history_best=history_best, fitness=objective(current_best if gen == 0 else history_best)))

best, best_fitness = pyes.MCCNES(
    comma_or_plus=pyes.MU_COMMA_LAMBDA,
    n_dimension=2,
    mean=[12, 12],
    cov=[[5, 0], [0, 5]],
    objective=objective,
    n_iter=10,
    learn_rate=0.01,
    mu=4,
    lam=16,
    random_state=1,
    event_on_genration=event_on_generation,
)

logger.info('best: {best}, best fitness: {best_fitness}'.format(best=best, best_fitness=best_fitness))


