import pyes
import matplotlib.pyplot as plt
import numpy as np

pyes.strategy.IS_DEBUG = False

objective = pyes.OBJECTIVE_FUNS['ackley']

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

def event_on_genration(g, xs):
    global sca

    print('{g} iter:'.format(g=g))
    if sca is not None:
        sca.remove()
    sca = plt.scatter(xs[:,0], xs[:,1], s=30, c='k')
    plt.pause(0.1)
    

print(pyes.ES(
    comma_or_plus=pyes.MU_COMMA_LAMBDA,
    n_dimension=2, 
    objective=objective, 
    bounds=np.asarray([[-20.0, -20.0], [20.0, 20.0]]),
    n_iter=100,
    learn_rate=0.02,
    mu=20,
    lam=100,
    random_state=1,
    event_on_generation=event_on_genration))

plt.ioff()
plt.show()