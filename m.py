import matplotlib.pyplot as plt
import numpy as np
import pymc

import elastica as e
import pymcelastica

R = pymc.MCMC(pymcelastica)  # build the model
R.sample(10000)  # populate and run it
print 'L   ', R.L.value  # print outputs
print 'E    ', R.E.value
print 'F    ', R.F.value
print 'a1   ', R.a1.value
print 'a2   ', R.a2.value

plt.plot(pymcelastica.xo[0], pymcelastica.xo[1], lw=1, c='r')
plt.plot(pymcelastica.ys[0], pymcelastica.ys[1], lw=3)
plt.scatter(pymcelastica.yn[0], pymcelastica.yn[1], c='k', marker='+')

L = R.L.value
E = R.E.value
F = R.F.value
a1 = R.a1.value
a2 = R.a2.value
gamma = R.gamma.value

s = np.linspace(0, L, 100)
r = e.planar_elastica_bvp(s, F=F, E=E)
pol = r.sol
xs, ys = e.eval_planar_elastica(s, pol, a1, a2)

# xs += x0[0]
# ys += x0[1]
xo = pol(s)
plt.plot(xo[0], xo[1], lw=1, c='b', label='%0.1f' % E)
plt.plot(xs, ys, c='b', lw=3)

plt.show()
