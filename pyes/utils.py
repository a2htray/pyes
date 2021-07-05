from numpy.random import RandomState
from .strategy import IS_DEBUG

def in_bounds(point, bounds):
	for d in range(len(point)):
		if point[d] < bounds[0, d] or point[d] > bounds[1, d]:
			return False
	return True


def init_solution(rs: RandomState, num_dimensions, bound_lows, bound_highs):
    return init_vector(rs, num_dimensions, bound_lows, bound_highs)


def init_vector(rs: RandomState, n, lows, highs):
    return lows + rs.uniform(0, 1, n) * (highs - lows)

def debug(*args):
    if IS_DEBUG:
        print(*args)