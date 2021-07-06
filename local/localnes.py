import pyes
import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

objective = pyes.OBJECTIVE_FUNS['square_sum']

n = 300
x = np.linspace(-20, 20, n)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)

for i in range(n):
    for j in range(n):
        Z[i, j] = objective([x[i], x[j]])

plt.contourf(X, Y, -Z, 100, cmap=plt.cm.rainbow)
plt.ylim(-20, 20)
plt.xlim(-20, 20)
plt.ion()

sca = None

def event_on_genration(g, xs, current_best, history_best):
    global sca

    logger.info('{g} iter: current_best={current_best}, history_best={history_best}'.format(g=g, current_best=current_best, history_best=history_best))

    if sca is not None:
        sca.remove()

    sca = plt.scatter(xs[:,0], xs[:,1], s=30, c='k')
    plt.pause(0.1)

best, best_fitness = pyes.NES(
    n_dimension=3,
    objective=objective, 
    n_iter=100,
    learn_rate=0.02,
    lam=100,
    random_state=1,
    event_on_genration=event_on_genration)

print('best: {best}, fitness: {fitness}'.format(best=best, fitness=best_fitness))

plt.ioff()
plt.show()