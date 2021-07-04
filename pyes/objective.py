# 定义部分目标函数
import math

def _ackley(xs):
    n = float(len(xs))

    return -20 * math.exp(-0.2 * math.sqrt(1/n * sum([x**2 for x in xs]))) - math.exp(1/n * sum([math.cos(2 * math.pi * x) for x in xs])) + 20 + math.e


OBJECTIVE_FUNS = {
    'ackley': _ackley
}