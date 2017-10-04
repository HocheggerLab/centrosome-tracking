# import logging
import numpy as np
import pymc
from numpy import cos, sin
from scipy.integrate import solve_bvp

np.set_printoptions(1)


def heavy_planar_bvp(s, E=10.0, F=0.1, gamma=np.pi / 2):
    """

        | x0' |   |  x  |   |    -cos(phi)     |
        | x1' | = |  y  | = |    -sin(phi)     |
        | x2' |   | phi |   |      -k          |
        | x3' |   |  M  |   | F sin(phi+gamma) |

    """

    def f(s, x, p):
        k = x[3] / E
        return [-cos(x[2]), -sin(x[2]), -k, F * sin(x[2] + gamma)]

    # Implement evaluation of the boundary condition residuals:
    def bc(xa, xb, p):
        m0 = p
        out = np.array([xb[0], xb[1], xb[2], xa[3] - m0, xa[1]])
        # print np.array(xa), np.array(xb), m0, '->', out
        return out

    y_a = np.zeros((4, s.size))

    # Now we are ready to run the solver.
    res = solve_bvp(f, bc, s, y_a, p=[1], verbose=1)
    return res


def eval_heavy_planar(s, pol, a1, a2):
    if 0 < a1 < 1 and 0 < a2 < 1:
        l = s[-1]
        l1 = a1 * l
        l2 = a2 * l
        sidx = np.searchsorted(s, [l1, l2])
        x_ = pol(s[sidx[0]:sidx[1]])
        return np.array(x_[0:2])


# create some test data
L = 10.0
a1, a2 = 0.1, 0.5
x0, y0 = 200, 250
s = np.linspace(0, L, 100)
r = heavy_planar_bvp(s, F=0.1, E=1.0)
pol = r.sol
xo = pol(s)
ys = eval_heavy_planar(s, pol, a1, a2)
# xs += x0[0]
# ys += x0[1]
yn = ys + 0.5 * (0.5 - np.random.rand(ys.shape[0], ys.shape[1]))
x = yn[0]

# priors
sig = pymc.Uniform('sig', 0.0, 10.0, value=1.)
L = pymc.Uniform('L', 5.0, 20.0, value=8.0)
E = pymc.Uniform('E', 0.0, 2.0, value=0.1)
F = pymc.Uniform('F', 0.0, 2.0, value=0.05)
a1 = pymc.Uniform('a1', 0.0, 1.0, value=0.1)
a2 = pymc.Uniform('a2', 0.0, 1.0, value=0.5)
gamma = pymc.Uniform('gamma', -np.pi / 2, np.pi / 2, value=0.5)


# model
@pymc.deterministic(plot=True)
def mod_heavyplanar(L=L, E=E, F=F, a1=a1, a2=a2, gamma=gamma):
    _a1, _a2 = min(a1, a2), max(a1, a2)
    a1, a2 = _a1, _a2
    if 0 < a1 < 1 and 0 < a2 < 1 and a2 > a1:
        print a1, a2, L, E, F, gamma
        s = np.linspace(a1 * L, a2 * L, int((a2 - a1) * 100))
        r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        ys = r.sol(s)
        return np.array(ys[0:2])


# likelihood
y = pymc.MvNormal('y', mu=mod_heavyplanar, tau=np.eye(2), value=yn, observed=True)
# y = pymc.Normal('y', mu=mod_heavyplanar, tau=1.0 / sig ** 2, value=yn, observed=True)
