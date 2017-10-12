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
Np = 100
yn = e.gen_test_data(a1, a2, L, E, F, np.pi / 2, 0, 0, Np, ax=plt.gca())


def obj_leastsquares(p, x, y):
    slen = y.size
    ymod = e.model_heavyplanar(p, num_points=Np)
    slen = min(slen, ymod.shape[1])
    objfn = (ymod[0:2, 0:slen] - np.array([x[0:slen], y[0:slen]])).flatten()
    print p, 'objective function %0.2f' % np.sum(objfn ** 2)

    L, a1, a2, E, F, gamma = p
    e.plot_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma)

    return objfn


def obj_minimize(p):
    print 'x=[%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f]. ' % tuple(p),
    slen = yn.shape[1]
    ymod = e.model_heavyplanar(p, num_points=Np)
    if ymod is not None and ymod.shape[1] >= slen:
        objfn = (ymod[0:2, 0:slen] - yn[0:2, 0:slen]).flatten()
        objfn = np.sum(objfn ** 2)
        print 'Objective function f(x)=%0.2f' % objfn

        # L, a1, a2, E, F, gamma = p
        # e.plot_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma)

        return objfn
    else:
        print 'No solution for objective function.'
        return np.finfo('float64').max


print np.pi / 2
print L, a1, a2, E, F

# param_bounds = np.array([[5.0, 0.1, 0.5, 0.01, 0.1, -np.pi], [15.0, 0.6, 1.0, 2.0, 2.0, np.pi]])
# x0 = [5.0, 0.1, 0.9, 0.1, 0.4, 0]
# res = least_squares(obj_leastsquares, x0, bounds=param_bounds, args=yn, ftol=1e-20,verbose=2)
# print res.x, res.success, res.status, res.message
# print 'x0=[%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f]' % tuple(res.x)
# print 'objective function final: %0.2f' % np.sum(obj(res.x, yn[0], yn[1])**2)

param_bounds = ((5.0, 15.0), (0.1, 0.6), (0.5, 1.0), (0.01, 3.0), (0.1, 3.0), (-np.pi, np.pi))
x0 = [9.0, 0.2, 0.7, 0.1, 0.4, 0]
res = basinhopping(obj_minimize, x0, minimizer_kwargs={'bounds': param_bounds})
print 'x0=[%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f] ' % tuple(res.x),
print 'objective function final: %0.2f' % obj_minimize(res.x)

L, a1, a2, E, F, gamma = res.x
e.plot_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma)
plt.savefig('/Users/Fabio/data/lab/figure.svg', format='svg')
plt.show()
