import unittest
import pyes
from numpy import asarray

class TestCMAES(unittest.TestCase):
    def test_CMAES0(self):
        pyes.strategy.IS_DEBUG = False
        print(pyes.NES(
            n_dimension=2, 
            objective=pyes.OBJECTIVE_FUNS['ackley'], 
            n_iter=100,
            learn_rate=0.02,
            lam=20))

if __name__ == '__main__':
    unittest.main()
