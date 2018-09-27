import logging

import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_bvp

np.set_printoptions(1)
logger = logging.getLogger(__name__)


def planar_elastica_bvp(s, E=1.0, J=1.0, N1=1.0, N2=1.0, m0=0.1,
                        theta_end=np.pi / 2, endX=0, endY=0):
    """

        | x0' |   |  x'    |   |     cos(theta)     |
        | x1' | = |  y'    | = |     sin(theta)     |
        | x2' |   | theta' |   |       M /(EJ)      |

        M -N1 y + N2 x + m0 = 0

        Variables: x, y, theta
        Parameters: N1, N2, m0
        Constants: E, J
        Boundary value conditions:
            (x(0), y(0), theta(0)) = (0, 0, 0)
            (x(L), y(L), theta(L)) = (endX, endY, theta_end)
    """

    def f(s, x, p):
        _n1, _n2, _m0 = p
        M = _n1 * x[1] - _n2 * x[0] - _m0
        return [cos(x[2]), sin(x[2]), M / E / J]

    # Implement evaluation of the boundary condition residuals:
    def bc(xa, xb, p):
        _n1, _n2, _m0 = p
        out = np.array([xa[0], xa[1],
                        xb[0] - endX,
                        xb[1] - endY,
                        xa[2],
                        xb[2] - theta_end])
        # print (np.array(xa), np.array(xb), p, '->', out)
        return out

    def fn_jac(s, x, p):
        _n1, _n2, _m0 = p
        sz = len(x[0])
        EJ = E * J
        zero = np.repeat(0, sz)
        dFdy = np.array([[zero, zero, -sin(x[2])],
                         [zero, zero, cos(x[2])],
                         [np.repeat(-_n2 / EJ, sz), np.repeat(_n1 / EJ, sz), zero]])
        dFdp = np.array([[zero, zero, zero],
                         [zero, zero, zero],
                         [x[1] / EJ, -x[0] / EJ, np.repeat(1 / EJ, sz)]])
        return dFdy, dFdp

    y_a = np.array([sin(s), cos(s), s ** 2])
    # logger.debug('planar_elastica_bvp called with params\r\n'
    #              'E=%0.2f, J=%0.2f, N1=%0.2f, N2=%0.2f, m0=%0.2f\r\n'
    #              'theta_end=%0.2f, endX=%0.2f, endY=%0.2f' % (E, J, N1, N2, m0, theta_end, endX, endY))

    # Now we are ready to run the solver.
    # res = solve_bvp(f, bc, s, y_a, p=[N1, N2, m0], fun_jac=fn_jac, verbose=1)
    res = solve_bvp(f, bc, s, y_a, p=[N1, N2, m0], verbose=1)
    return res


def eval_planar_elastica(s, pol, a1, a2):
    if 0 < a1 < 1 and 0 < a2 < 1:
        l = s[-1]
        l1 = a1 * l
        l2 = a2 * l
        sidx = np.searchsorted(s, [l1, l2])
        x_ = pol(s[sidx[0]:sidx[1]])
        return np.array(x_[0:2])


def gen_test_data(L, a1, a2, E, J, N1, N2, m0, theta_e, x0=0, y0=0, phi=0, num_points=100, sigma=0.1, ax=None):
    s = np.linspace(0, L, num_points)
    r = planar_elastica_bvp(s, E=E, J=J, N1=N1, N2=N2, m0=m0, theta_end=theta_e)
    pol = r.sol
    xo = pol(s)[0:2, :]
    ys = eval_planar_elastica(s, pol, a1, a2)[0:2, :]

    # deal with rotations and translations
    sinth, costh = np.sin(phi), np.cos(phi)
    M = np.array([[costh, -sinth], [sinth, costh]])
    ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
    xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

    # add noise
    yn = ys.copy()
    yn[1] += 0.5 * (0.5 - np.random.rand(ys.shape[1]))

    if ax is not None:
        ax.plot(xo[0], xo[1], lw=6, c='r', zorder=2)
        ax.plot(ys[0], ys[1], lw=10, c='g', zorder=1)
        ax.scatter(yn[0], yn[1], c='g', marker='<', zorder=3)

    return yn


def plot_planar_elastica(ax, L, a1, a2, E, J, N1, N2, m0, theta_e, endX, endY, x0=0, y0=0, phi=0, num_points=100,
                         alpha=0.5):
    s = np.linspace(0, L, num_points)
    r = planar_elastica_bvp(s, E=E, J=J, N1=N1, N2=N2, m0=m0, theta_end=theta_e, endX=endX, endY=endY)
    pol = r.sol
    xo = pol(s)[0:2, :]
    xs, ys = eval_planar_elastica(s, pol, a1, a2)[0:2, :]
    ys = np.array([xs, ys])

    # deal with rotations and translations
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    M = np.array([[cosphi, -sinphi], [sinphi, cosphi]])
    ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
    xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

    ax.plot(xo[0], xo[1], lw=1, c='b', alpha=alpha, label='%0.1f' % E, zorder=4)
    ax.plot(ys[0], ys[1], lw=3, c='b', alpha=alpha, zorder=4)
    ax.scatter(ys[0], ys[1], c='k', marker='+', alpha=alpha, zorder=5)


def model_planar_elastica(p, num_points=100):
    L, a1, a2, E, J, N1, N2, m0, theta_e, x0, y0, phi = p
    if 0 < a1 < 1 and 0 < a2 < 1:
        _a1, _a2 = min(a1, a2), max(a1, a2)
        a1, a2 = _a1, _a2
        s1 = int(a1 * num_points)
        s2 = int(a2 * num_points)
        s = np.linspace(0, L, num_points)
        r = planar_elastica_bvp(s, E=E, J=J, N1=N1, N2=N2, m0=m0, theta_end=theta_e)
        pol = r.sol
        ys = pol(s)[0:2, :]

        # deal with rotations and translations
        sinth, costh = np.sin(phi), np.cos(phi)
        M = np.array([[costh, -sinth], [sinth, costh]])
        ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))

        return ys[0:2, s1:s2]


def obj_minimize(p, yn, Np=100):
    slen = yn.shape[1]
    ymod = model_planar_elastica(p, num_points=Np)
    if ymod is not None and ymod.shape[1] >= slen:
        objfn = (ymod[0:2, 0:slen] - yn[0:2, 0:slen]).flatten()
        objfn = np.sum(objfn ** 2)
        logging.debug(
            'x=[%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f]. Obj f(x)=%0.3f' % (
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], objfn))
        return objfn
    else:
        logging.debug('No solution for objective function.')
        return np.finfo('float64').max
