# 定义部分目标函数
import math

def _ackley(xs):
    n = float(len(xs))

    return -20 * math.exp(-0.2 * math.sqrt(1/n * sum([x**2 for x in xs]))) - math.exp(1/n * sum([math.cos(2 * math.pi * x) for x in xs])) + 20 + math.e


def _bent_cigar(xs):
    return xs[0]**2 + 10**6 * sum([x*2 for x in xs[1:]])


def _square_sum(xs):
    return sum([x**2 for x in xs])

OBJECTIVE_FUNS = {
    'ackley': _ackley,
    'bent_cigar': _bent_cigar,
    'square_sum': _square_sum
}