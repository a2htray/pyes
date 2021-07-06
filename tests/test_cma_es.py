import unittest
import pyes
from numpy import asarray

class TestCMAES(unittest.TestCase):
    def test_CMAES0(self):
        print(pyes.CMAES(
            asarray([0, 0]), asarray([[1, 0], [0, 1]]), 2, pyes.OBJECTIVE_FUNS['ackley'], n_iter=1, learn_rate=1, mu=2, lam=8, sigma=1, random_state=1,
        ))

        # print(pyes.CMAES(
        #     asarray([0, 0]), asarray([[1, 0], [0, 1]]), 2, pyes.OBJECTIVE_FUNS['ackley'],  n_iter=1, learn_rate=1, mu=2, lam=4, sigma=1, random_state=1,
        # ))

if __name__ == '__main__':
    unittest.main()
