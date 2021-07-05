import unittest
import pyes
from numpy import asarray


class TestES(unittest.TestCase):
    def test_ES0(self):
        bounds = asarray([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
        n_dimension = 3
        n_iter = 1000
        learn_rate = 0.011
        mu = 20
        lam = 100
        random_state = 2
        best, score = pyes.ES(pyes.MU_COMMA_LAMBDA, n_dimension, pyes.OBJECTIVE_FUNS['ackley'], bounds, n_iter, learn_rate, mu, lam, random_state=random_state)
        print('Done!')
        print('f(%s) = %f' % (best, score))

    def test_ES1(self):
        bounds = asarray([[-5.0, -5.0], [5.0, 5.0]])
        n_dimension = 2
        n_iter = 1000
        learn_rate = 0.011
        mu = 20
        lam = 100
        random_state = 2
        best, score = pyes.ES(pyes.MU_PLUS_LAMBDA, n_dimension, pyes.OBJECTIVE_FUNS['ackley'], bounds, n_iter, learn_rate, mu, lam, random_state=random_state)
        print('Done!')
        print('f(%s) = %f' % (best, score))


if __name__ == '__main__':
    unittest.main()
