import warnings
warnings.filterwarnings('ignore')

from .strategy import EVENT_ON_GENERATION
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
from loguru import logger
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

def NES(
    n_dimension: int, 
    objective, 
    n_iter, 
    learn_rate, 
    lam,
    mean,
    sigma,
    cov,
    random_state=None, 
    event_on_genration=EVENT_ON_GENERATION):
    logger.debug({
        'dimension': n_dimension,
        'n_iter': n_iter,
        'learn_rate': learn_rate,
    })
    tf.set_random_seed(random_state)
    best, best_fitness = None, 1e+10
    
    def get_fitness(population): 
        return [-objective(solution) for solution in population]

    mean = tf.Variable(tf.random_normal([n_dimension, ], mean, sigma), dtype=tf.float32)
    cov = tf.Variable(cov, dtype=tf.float32)
    mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    make_population = mvn.sample(lam)

    fitness_input = tf.placeholder(tf.float32, [lam, ])
    prob_output = tf.placeholder(tf.float32, [lam, n_dimension])
    loss = -tf.reduce_mean(mvn.log_prob(prob_output)*fitness_input)         # log prob * fitness
    train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss) # compute and apply gradients for mean and cov

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # training
    for g in range(n_iter):
        population = sess.run(make_population)

        scores = [objective(c) for c in population]
        
        event_on_genration(g, population, population[np.argsort(scores)[0]], best)

        fitness_values = get_fitness(population)
        sess.run(train_op, {fitness_input: fitness_values, prob_output: population}) 

        for i, score in enumerate([objective(x) for x in population]):
            if score < best_fitness:
                best, best_fitness = population[i], score

    return [best, best_fitness]
    
