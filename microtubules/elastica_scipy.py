import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping

from microtubules import elastica as e

np.set_printoptions(3, suppress=True)
logging.basicConfig(level=logging.DEBUG)

# create some test data
L, a1, a2 = 1.0, 0.1, 0.6
E, J, F, gamma = 0.625, 1.0, 0.1, np.pi / 2
N1, N2, phi, m0 = 1.0, 1.0, 0.0, 0.01
x0, y0, theta = 0, 0, np.pi / 6
theta_e = np.pi / 2

Np = 100
yn = e.gen_test_data(L, a1, a2, E, F, gamma, x0, y0, theta, Np, ax=plt.gca())

print(L, a1, a2, E, F, gamma, x0, y0, theta)
param_bounds = ((5.0, 20.0), (0.05, 0.6), (0.5, 1.0),
                (0.01, 2.0), (0.0, 10.0), (-np.pi, np.pi),
                (0, 120.), (0, 120.), (-np.pi, np.pi))

x0 = [L, a1, a2, E, J, N1, N2, m0, theta_e, x0, y0, phi]
res = basinhopping(e.obj_minimize, x0, minimizer_kwargs={'bounds': param_bounds, 'args': (yn, Np)})
logging.info('x0=[%f,%f,%f,%f,%f,%f,%f,%f,%f] ' % tuple(res.x))
logging.info('objective function final: %f' % e.obj_minimize(res.x, yn))

L, a1, a2, E, F, gamma, x0, y0, theta = res.x
e.plot_planar_elastica(plt.gca(), L, a1, a2, E, F, gamma, x0, y0, theta)
plt.savefig('/Users/Fabio/data/lab/figure.svg', format='svg')
plt.show()
