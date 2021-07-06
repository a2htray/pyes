# requirements:
#   sklearn 0.24.2
import numpy as np
from loguru import logger
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pyes

def encode(x):
    """适应值映射为 boolean 类型
    用于选择特征子集
    """
    # 不使用 numpy 的实现
    # logger.debug([True if (1.0/(1.0+math.exp(-v*2))) > 0.5 else False for v in x])
    return (1.0 / (1.0 + np.exp(-x*2)) > 0.5)

logger.debug(encode(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])))

def DR(encoded_x, n_feature):
    """计算特征维度缩减率"""
    return 1 - 1.0 * sum([1 for v in encoded_x if v]) / n_feature

logger.debug('DR:{DR}'.format(DR=DR([True, True, False], 3)))

def CA(encode_x, data_X, data_y):
    """计算分类准确率"""
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(data_X[:,encode_x], data_y)
    return clf.score(data_X[:,encode_x], data_y)

glass_data = np.genfromtxt('./exampleData/glass.data',delimiter=',')
n_sample, n_feature = glass_data.shape[0], glass_data.shape[1] - 1

logger.debug('number of samples: {n_sample}, number of features: {n_feature}'.format(n_sample=n_sample, n_feature=n_feature))

# 超参
alpha = 0.01 # 高斯扰动
C = 1.0 * np.eye(n_feature) # 分布协方差矩阵

logger.debug('C: {C}'.format(C=C))

logger.debug(glass_data)

np.random.shuffle(glass_data)

data_X = glass_data[:,:-1]
data_y = glass_data[:, -1]

def objective(x):
    """目标函数"""
    encoded_x = encode(x)
    rou = (9+0.99*(1. * n_feature/100)) / 10

    return -(rou * CA(encoded_x, data_X, data_y) + (1-rou)*DR(encoded_x, n_feature))

logger.debug('CA: {CA}'.format(CA=CA([True, True, True, True, False, False, False, False, False, True], data_X, data_y)))

# 各特征均值向量
vector_mean = np.mean(data_X, axis=0)
logger.debug('vector_mean: {vector_mean}'.format(vector_mean=vector_mean))

def event_on_genration(g, xs, current_best, history_best):
    global sca

    logger.info('{g} iter: current_best={current_best}, history_best={history_best}'.format(g=g, current_best=current_best, history_best=history_best))


best, best_fitness = pyes.NES(
    n_dimension=n_feature,
    objective=objective, 
    n_iter=30,
    learn_rate=0.02,
    lam=100,
    random_state=1,
    event_on_genration=event_on_genration)


logger.info('best feature subset: {subset}'.format(subset=encode(best)))
logger.info('best fitness: {fitness}'.format(fitness=best_fitness))