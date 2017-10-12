import numpy as np
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


def gen_test_data(a1, a2, L, e, f, gamma, x0, y0, num_points=100, ax=None):
    s = np.linspace(0, L, num_points)
    r = heavy_planar_bvp(s, F=f, E=e)
    pol = r.sol
    xo = pol(s)
    ys = eval_heavy_planar(s, pol, a1, a2)
    # xs += x0[0]
    # ys += x0[1]
    yn = np.array([ys[0], ys[1]])
    yn[1] += 0.5 * (0.5 - np.random.rand(ys.shape[1]))

    if ax is not None:
        ax.plot(xo[0], xo[1], lw=6, c='r')
        ax.plot(ys[0], ys[1], lw=10, c='g')
        ax.scatter(yn[0], yn[1], c='g', marker='<')

    return yn


def plot_heavyplanar(ax, L, a1, a2, E, F, gamma, num_points=100, plot_options=None):
    s = np.linspace(0, L, num_points)
    r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
    pol = r.sol
    xs, ys = eval_heavy_planar(s, pol, a1, a2)
    # xs += x0[0]
    # ys += x0[1]
    xo = pol(s)
    ax.plot(xo[0], xo[1], lw=1, c='b', label='%0.1f' % E)
    ax.plot(xs, ys, c='b', lw=3)
    ax.scatter(xs, ys, c='k', marker='+')


def model_heavyplanar(p, num_points=100):
    L, a1, a2, E, F, gamma = p
    if 0 < a1 < 1 and 0 < a2 < 1:
        _a1, _a2 = min(a1, a2), max(a1, a2)
        a1, a2 = _a1, _a2
        s1 = int(a1 * num_points)
        s2 = int(a2 * num_points)
        s = np.linspace(0, L, num_points)
        r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        ys = pol(s)

        return ys[0:2, s1:s2]
