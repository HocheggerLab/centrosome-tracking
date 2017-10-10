import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize

np.set_printoptions(1, suppress=True)


def heavy_planar_bvp(s, E=10.0, F=0.1, gamma=np.pi / 2):
    """

        | x0' |   |  x  |   |    -cos(phi)     |
        | x1' | = |  y  | = |    -sin(phi)     |
        | x2' |   | phi |   |      -k          |
        | x3' |   |  M  |   | F sin(phi+gamma) |

    """

    def f(s, x, p):
        k = x[3] / E
        return [-np.cos(x[2]), -np.sin(x[2]), -k, F * np.sin(x[2] + gamma)]

    # Implement evaluation of the boundary condition residuals:
    def bc(xa, xb, p):
        m0 = p
        out = np.array([xb[0], xb[1], xb[2], xa[3] - m0, xa[1]])
        # print np.array(xa), np.array(xb), m0, '->', out
        return out

    y_a = np.zeros((4, s.size))

    # Now we are ready to run the solver.
    res = solve_bvp(f, bc, s, y_a, p=[1], verbose=0)
    return res


def eval_heavy_planar(s, pol, a1, a2):
    if 0 < a1 < 1 and 0 < a2 < 1:
        l = s[-1]
        l1 = a1 * l
        l2 = a2 * l
        sidx = np.searchsorted(s, [l1, l2])
        x_ = pol(s[sidx[0]:sidx[1]])
        return np.array(x_[0:2])


def print_heavyplanar(ax, L, a1, a2, E, F, gamma):
    s = np.linspace(0, L, Np)
    r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
    pol = r.sol
    xs, ys = eval_heavy_planar(s, pol, a1, a2)

    # xs += x0[0]
    # ys += x0[1]
    xo = pol(s)
    ax.plot(xo[0], xo[1], lw=1, c='b', label='%0.1f' % E)
    ax.plot(xs, ys, c='b', lw=3)


# create some test data
L = 10.0
a1, a2 = 0.1, 0.6
e, f = 1.0, 0.1

x0, y0 = 200, 250
Np = 100

s = np.linspace(0, L, Np)
r = heavy_planar_bvp(s, F=f, E=e)
pol = r.sol
xo = pol(s)
ys = eval_heavy_planar(s, pol, a1, a2)
# xs += x0[0]
# ys += x0[1]
yn = np.array([ys[0], ys[1]])


# yn[1] += 0.5 * (0.5 - np.random.rand(ys.shape[1]))
# x = yn[0]


def mod_heavyplanar(p):
    L, a1, a2, E, F, gamma = p
    if 0 < a1 < 1 and 0 < a2 < 1:
        _a1, _a2 = min(a1, a2), max(a1, a2)
        a1, a2 = _a1, _a2
        s1 = int(a1 * Np)
        s2 = int(a2 * Np)
        s = np.linspace(0, L, Np)
        r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        ys = pol(s)

        slen = yn.shape[1]
        if slen > s2 - s1:
            slen = s2 - s1
        # objfn = np.sum(np.dot((ys[0:2, s1:s2] - yn[0:2, 0:slen]) ** 2, yn[0, 0:slen]))
        objfn = np.sum((ys[0:2, s1:s2] - yn[0:2, 0:slen]) ** 2)
        print p, 'objective function %0.2f' % objfn
        # print_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma)

        return objfn


def mod_heavyplanar_x(p):
    L, a1, a2, E, F, gamma = p
    if 0 < a1 < 1 and 0 < a2 < 1:
        _a1, _a2 = min(a1, a2), max(a1, a2)
        a1, a2 = _a1, _a2
        # print a1, a2, L, E, F, gamma
        # s = np.linspace(a1 * L, a2 * L, int((a2 - a1) * 100))
        s1 = int(a1 * Np)
        s2 = int(a2 * Np)
        s = np.linspace(0, L, Np)
        r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        ys = pol(s)

        print 'min x %0.2f' % min(ys[0, s1:s2])
        return min(ys[0, s1:s2])


print np.pi / 2
print L, a1, a2, e, f
plt.plot(xo[0], xo[1], lw=1, c='r')
plt.plot(ys[0], ys[1], lw=3)
plt.scatter(yn[0], yn[1], c='k', marker='+')

# param_bounds = np.array([[5.0, 0.1, 0.5, 0.01, 0.1, -np.pi], [15.0, 0.6, 1.0, 3.0, 3.0, np.pi]]).T
param_bounds = ((5.0, 15.0), (0.1, 0.6), (0.5, 1.0), (0.01, 3.0), (0.1, 3.0), (-np.pi, np.pi))
x0 = [9.0, 0.2, 0.7, 0.1, 0.4, 0]
# res = minimize(mod_heavyplanar, x0, method='SLSQP', bounds=param_bounds)
cons = ({'type': 'ineq',
         'fun': mod_heavyplanar_x})
res = minimize(mod_heavyplanar, x0, method='BFGS',
               bounds=param_bounds, )
print res.x, res.success, res.message
print 'x0=[%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f]' % tuple(res.x)

L, a1, a2, E, F, gamma = res.x
print_heavyplanar(plt.gca(), L, a1, a2, E, F, gamma)
plt.show()
