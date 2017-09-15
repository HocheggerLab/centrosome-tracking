import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numpy import cos, sin
from scipy.integrate import solve_bvp
from scipy.optimize import curve_fit
from scipy.special import ellipeinc, ellipj

from label import labelLines

np.set_printoptions(1)


class HeavyPlanarElastica():
    """

        | x0' |   |  x  |   |    -cos(phi)     |
        | x1' | = |  y  | = |    -sin(phi)     |
        | x2' |   | phi |   |      -k          |
        | x3' |   |  M  |   | F sin(phi+gamma) |

    """

    def __init__(self):
        self.L = 3.0
        self.E = 10.0
        self.F = 0.1
        self.gamma = 0.1
        self.endY = 1
        self.m0 = None

    def f(self, s, x, p):
        k = x[3] / self.E
        return [-cos(x[2]), -sin(x[2]), -k, self.F * sin(x[2] + self.gamma)]

    # Implement evaluation of the boundary condition residuals:
    def bc(self, xa, xb, m0):
        # p = m0
        out = np.array([xb[0], xb[1], xb[2], xa[3] - m0, xa[1] - self.endY])
        # print  np.array(xa), np.array(xb), m0, '->', out
        return out

    def update_ode(self):
        s = np.linspace(0, self.L, 4)
        y_a = np.zeros((4, s.size))

        # Now we are ready to run the solver.
        self.res = solve_bvp(self.f, self.bc, s, y_a, p=[1], verbose=0)
        self.m0 = self.res.p[0]


# def heavy_planar_bvp(s, E=10.0, F=0.1, gamma=0):
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
    res = solve_bvp(f, bc, s, y_a, p=[1], verbose=0)
    return res


def eval_heavy_planar(s, pol, a1, a2):
    if 0 < a1 < 1 and 0 < a2 < 1:
        ns = s.size
        l = s[-1]
        l1 = a1 * l
        l2 = a2 * l
        sidx = np.searchsorted(s, [l1, l2])
        x_ = pol(s[sidx[0]:sidx[1]])
        return x_[0], x_[1]


def heavy_planar(x, L, e, f, gamma):
    print 'heavy_planar --> ', L, e, f, gamma
    # if x1 < 0 or x2 < 0:
    #     raise ValueError('x1 & x2 must be positive')
    # xa = min(x1, x2)
    # xb = max(x1, x2)
    # if not np.all((xa < x) * (x < xb)):
    #     raise ValueError('some values of x not in interval')

    s = np.linspace(0, L, 100)
    r = heavy_planar_bvp(s, F=f, E=e, gamma=gamma)
    s1 = np.max(r.sol.solve(y=np.min(x))[0])
    s2 = np.max(r.sol.solve(y=np.max(x))[0])
    print 's1=%0.2f s2=%0.2f' % (s1, s2)
    sx = list()
    for _x in x:
        _sx = r.sol.solve(y=_x)[0]
        # print  _x, _sx
        logging.info(_sx)
        sx.append(np.max(_sx))
    ys = r.sol(sx)
    return ys[1]


def pole_clamp(P, alpha):
    def F(m):
        sn, cn, dn, ph = ellipj(np.sqrt(P / m), m)
        return [2 * ph - alpha]

    m = scipy.optimize.linearmixing(F, [0.9], verbose=True)
    u0 = np.sqrt(P / m)

    s = np.linspace(0, 1, num=100)

    u = u0 * (1 - s)
    _, _, dn0, am_u0 = ellipj(u0, m)
    _, _, dn, am_u = ellipj(u, m)
    x = s * (m - 2) / m - 2 / (m * u0) * (ellipeinc(am_u, m) * np.sign(dn) - ellipeinc(am_u0, m) * np.sign(dn0))
    y = 2 / (m * u0) * (dn - dn0)
    return x, y


L = 10.0
s = np.linspace(0, L, 100)
e = 10.0
f = 0.1
# for f in np.arange(-0.9, 0.9, 0.1):
#     r = heavy_planar(s, F=f, E=e)
#     ss = s[40:90]
#     x, y = r.sol(ss)[0], r.sol(ss)[1]
#
#     plt.plot(x, y, label='%0.1f' % f)

x0 = np.array([10.0, 13.0, 0, 0]).reshape((4, 1))
r = heavy_planar_bvp(s, F=f, E=e)
pol = r.sol
xo = pol(s)
xs, ys = eval_heavy_planar(s, pol, 0.1, 0.5)
# xs += x0[0]
# ys += x0[1]
plt.plot(xo[0], xo[1], lw=1, label='%0.1f' % f)
plt.plot(xs, ys, lw=2)
yn = ys + 0.5 * (0.5 - np.random.rand(ys.size))
plt.scatter(xs, yn, s=2, c='k')

# model fit
# def heavy_planar(x, L, e, f, gamma):
popt, pcov = curve_fit(heavy_planar, xs, ys, bounds=(0, [15, 10, 5, np.pi]))
print popt
print pcov

# plot solution
L, e, f, gamma = popt
r = heavy_planar_bvp(s, F=f, E=e)
s = np.linspace(0, L, 100)
r = heavy_planar_bvp(s, F=f, E=e)
pol = r.sol
xo = pol(s)
# xs, ys = eval_heavy_planar(s, pol, 0.1, 0.5)
plt.plot(xo[0], xo[1], lw=1, c='r', label='fit-%0.1f' % f)

ax = plt.gca()
labelLines(ax.get_lines(), fontsize=6, align=False)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# ax.set_xlim([min(xlim[0], ylim[0]), max(xlim[1], ylim[1])])
# ax.set_ylim([min(xlim[0], ylim[0]), max(xlim[1], ylim[1])])
# plt.axes().set_aspect('equal')
plt.show()


# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
#
#
# # define the data to be fit with some noise
# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise
# plt.plot(xdata, ydata, 'b-', label='data')
#
# # Fit for the parameters a, b, c of the function func
# popt, pcov = curve_fit(func, xdata, ydata)
# plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')
#
# # Constrain the optimization to the region of 0 < a < 3, 0 < b < 2 and 0 < c < 1:
# popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 2., 1.]))
# plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
