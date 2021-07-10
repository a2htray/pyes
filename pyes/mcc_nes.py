import warnings
warnings.filterwarnings('ignore')
from .strategy import EVENT_ON_GENERATION
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
from loguru import logger
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

def MCCNES(
    comma_or_plus: str, 
    n_dimension: int,
    mean,
    cov,
    objective, 
    n_iter, 
    learn_rate, 
    mu, 
    lam, 
    random_state=None, 
    event_on_genration=EVENT_ON_GENERATION):
    """
    Parameters
        comma_or_plus: pyes.MU_COMMA_LAMBDA or pyes.MU_PLUS_LAMBDA
        n_dimension: int 个体的维数
        mean: 各维的均值，取决于具体数据集
        cov: 协方差矩阵
        objective: 目标函数
        n_iter: 迭代的次数
        learn_rate: 学习率
        mu: 第代选取的个体数量
        lam: lambda 个体的数量
        random_state: 随机种子
    """
    logger.debug('MCCNES args: {args}'.format(args={
        'comma_or_plus': comma_or_plus,
        'n_dimension': n_dimension,
        'mean': mean,
        'cov': cov,
        'n_iter': n_iter,
        'learn_rate': learn_rate,
        'mu': mu,
        'lam': lam,
        'random_state': random_state,
    }))

    
    tf.set_random_seed(random_state)
    best, best_fitness = None, 1e+10
    
    def get_fitness(population): 
        return [-objective(solution) for solution in population]

    # 均值向量
    mean = tf.Variable(mean, dtype=tf.float32)
    # 协方差矩阵
    cov = tf.Variable(cov, dtype=tf.float32)
    mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    make_population = mvn.sample(lam) # 新解生成器

    # 概率网络，计算适应值权重
    fitness_input = tf.placeholder(tf.float32, [lam, ])
    prob_output = tf.placeholder(tf.float32, [lam, n_dimension])
    loss = -tf.reduce_mean(mvn.log_prob(prob_output) * fitness_input)
    train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss) # 用梯度下降算法优化概率网络

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

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
    
