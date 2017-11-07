import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping

import elastica as e

np.set_printoptions(3, suppress=True)
logging.basicConfig(level=logging.DEBUG)

# create some test data
L, a1, a2 = 10.0, 0.1, 0.6
E, F, gamma = 1.0, 0.1, np.pi / 2
x0, y0, theta = 20, 35, np.pi / 6
Np = 100
yn = e.gen_test_data(L, a1, a2, E, F, gamma, x0, y0, theta, Np, ax=plt.gca())

print np.pi / 2
print L, a1, a2, E, F, gamma, x0, y0, theta
param_bounds = ((5.0, 20.0), (0.05, 0.6), (0.5, 1.0),
                (0.01, 2.0), (0.0, 10.0), (-np.pi, np.pi),
                (0, 120.), (0, 120.), (-np.pi, np.pi))
x0 = [9.0, 0.2, 0.7, 0.1, 2, 10, 0, 0, 0]
res = basinhopping(e.obj_minimize, x0, minimizer_kwargs={'bounds': param_bounds, 'args': (yn, Np)})
logging.info('x0=[%f,%f,%f,%f,%f,%f,%f,%f,%f] ' % tuple(res.x))
logging.info('objective function final: %f' % e.obj_minimize(res.x, yn))

L, a1, a2, E, F, gamma, x0, y0, theta = res.x
e.plot_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma, x0, y0, theta)
plt.savefig('/Users/Fabio/data/lab/figure.svg', format='svg')
plt.show()
