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


def gen_test_data(a1, a2, L, e, f, gamma, x0, y0, theta, num_points=100, ax=None):
    s = np.linspace(0, L, num_points)
    r = heavy_planar_bvp(s, F=f, E=e, gamma=gamma)
    pol = r.sol
    xo = pol(s)[0:2, :]
    ys = eval_heavy_planar(s, pol, a1, a2)[0:2, :]

    # add noise
    yn = ys.copy()
    yn[1] += 0.5 * (0.5 - np.random.rand(ys.shape[1]))

    # deal with rotations and translations
    sinth, costh = np.sin(theta), np.cos(theta)
    M = np.array([[costh, -sinth], [sinth, costh]])
    ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
    xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

    if ax is not None:
        ax.plot(xo[0], xo[1], lw=6, c='r', zorder=2)
        ax.plot(ys[0], ys[1], lw=10, c='g', zorder=1)
        ax.scatter(yn[0], yn[1], c='g', marker='<', zorder=3)

    return yn


def plot_heavyplanar(ax, L, a1, a2, E, F, gamma, x0, y0, theta, num_points=100):
    s = np.linspace(0, L, num_points)
    r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
    pol = r.sol
    xo = pol(s)[0:2, :]
    xs, ys = eval_heavy_planar(s, pol, a1, a2)[0:2, :]
    ys = np.array([xs, ys])

    # deal with rotations and translations
    sinth, costh = np.sin(theta), np.cos(theta)
    M = np.array([[costh, -sinth], [sinth, costh]])
    ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
    xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

    ax.plot(xo[0], xo[1], lw=1, c='b', label='%0.1f' % E, zorder=4)
    ax.plot(ys[0], ys[1], lw=3, c='b', zorder=4)
    ax.scatter(ys[0], ys[1], c='k', marker='+', zorder=5)


def model_heavyplanar(p, num_points=100):
    L, a1, a2, E, F, gamma, x0, y0, theta = p
    if 0 < a1 < 1 and 0 < a2 < 1:
        _a1, _a2 = min(a1, a2), max(a1, a2)
        a1, a2 = _a1, _a2
        s1 = int(a1 * num_points)
        s2 = int(a2 * num_points)
        s = np.linspace(0, L, num_points)
        r = heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        ys = pol(s)[0:2, :]

        # deal with rotations and translations
        sinth, costh = np.sin(theta), np.cos(theta)
        M = np.array([[costh, -sinth], [sinth, costh]])
        ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))

        return ys[0:2, s1:s2]
