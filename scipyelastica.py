import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping

import elastica as e

np.set_printoptions(3, suppress=True)

# create some test data
L = 10.0
a1, a2 = 0.1, 0.6
E, F = 1.0, 0.1
x0, y0 = 200, 250
alpha, gamma = 0, np.pi / 2
Np = 100
yn = e.gen_test_data(a1, a2, L, E, F, gamma, x0, y0, alpha, Np, ax=plt.gca())


def obj_minimize(p):
    print 'x=[%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f]. ' % tuple(p),
    slen = yn.shape[1]
    ymod = e.model_heavyplanar(p, num_points=Np)
    if ymod is not None and ymod.shape[1] >= slen:
        objfn = (ymod[0:2, 0:slen] - yn[0:2, 0:slen]).flatten()
        objfn = np.sum(objfn ** 2)
        print 'Objective function f(x)=%0.2f' % objfn
        return objfn
    else:
        print 'No solution for objective function.'
        return np.finfo('float64').max


print np.pi / 2
print L, a1, a2, E, F, gamma, x0, y0, alpha
# L, a1, a2, E, F, gamma, x0, y0, alpha = p
param_bounds = ((5.0, 15.0), (0.1, 0.6), (0.5, 1.0), (0.01, 3.0), (0.1, 3.0), (-np.pi, np.pi),
                (0, 512.), (0, 512.), (-np.pi, np.pi))
x0 = [9.0, 0.2, 0.7, 0.1, 0.4, 0, 0, 0, 0]
res = basinhopping(obj_minimize, x0, minimizer_kwargs={'bounds': param_bounds})
print 'x0=[%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f] ' % tuple(res.x),
print 'objective function final: %0.2f' % obj_minimize(res.x)

L, a1, a2, E, F, gamma, x0, y0, alpha = res.x
e.plot_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma, x0, y0, alpha)
plt.savefig('/Users/Fabio/data/lab/figure.svg', format='svg')
plt.show()
