# 检查方法
from .strategy import MU_COMMA_LAMBDA, MU_PLUS_LAMBDA


def must_selection_strategy(strategy: str):
    """
    确保传入的候选解选取策略的正确性
    """
    if strategy not in [MU_COMMA_LAMBDA, MU_PLUS_LAMBDA]:
        raise Exception('selection strategy must in {strategies}'.format(strategies=','.join([MU_COMMA_LAMBDA, MU_PLUS_LAMBDA])))


def must_desc_sequence(seqs):
    """
    确保序列必须为降序
    """
    n = len(seqs)
    for i in range(n-1):
        if seqs[i] < seqs[i+1]:
            raise Exception('the seqs must be a desc sequence')
