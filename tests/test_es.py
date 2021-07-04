import unittest
import pyes
from numpy import asarray


class TestES(unittest.TestCase):
    def test_ES0(self):
        bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
        n_iter = 5000
        step_size = 0.15
        mu = 20
        lam = 100
        best, score = pyes.ES(pyes.MU_COMMA_LAMBDA, pyes.OBJECTIVE_FUNS['ackley'], bounds, n_iter, step_size, mu, lam)
        print('Done!')
        print('f(%s) = %f' % (best, score))

    def test_ES1(self):
        bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
        n_iter = 5000
        step_size = 0.15
        mu = 20
        lam = 100
        best, score = pyes.ES(pyes.MU_PLUS_LAMBDA, pyes.OBJECTIVE_FUNS['ackley'], bounds, n_iter, step_size, mu, lam)
        print('Done!')
        print('f(%s) = %f' % (best, score))



if __name__ == '__main__':
    unittest.main()