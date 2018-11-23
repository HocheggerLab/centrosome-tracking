import logging

import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_bvp
from lmfit import Parameters

np.set_printoptions(1)
logger = logging.getLogger(__name__)


def planar_elastica_bvp_numeric(s, E=1.0, J=1.0, N1=1.0, N2=1.0, m0=0.1,
                                theta_end=np.pi / 2, endX=0.0, endY=0.0):
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
        out = np.array([xa[0], xa[1], xa[2],
                        xb[0] - endX,
                        xb[1] - endY,
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
    logger.debug('planar_elastica_bvp_numeric called with params:'
                 'L=%0.2e, E=%0.2e, J=%0.2e, N1=%0.2e, N2=%0.2e, m0=%0.2e '
                 'theta_end=%0.2e, endX=%0.2e, endY=%0.2e' % (s[-1], E, J, N1, N2, m0, theta_end, endX, endY))

    # Now we are ready to run the solver.
    res = solve_bvp(f, bc, s, y_a, p=[N1, N2, m0], fun_jac=fn_jac, verbose=2)
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
    r = planar_elastica_bvp_numeric(s, E=E, J=J, N1=N1, N2=N2, m0=m0, theta_end=theta_e)
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


def model_planar_elastica(p, num_points=100):
    L, a1, a2, E, J, N1, N2, m0, theta_e, x0, y0, phi = p
    if 0 < a1 < 1 and 0 < a2 < 1:
        _a1, _a2 = min(a1, a2), max(a1, a2)
        a1, a2 = _a1, _a2
        s1 = int(a1 * num_points)
        s2 = int(a2 * num_points)
        s = np.linspace(0, L, num_points)
        r = planar_elastica_bvp_numeric(s, E=E, J=J, N1=N1, N2=N2, m0=m0, theta_end=theta_e)
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


class PlanarElastica():
    def __init__(self, L=1.0, E=0.625, J=1.0, N1=1.0, N2=1.0,
                 x0=0.0, y0=0.0, dx=0.4, dy=0.1,
                 phi=0.0, theta=3 * np.pi / 2):
        self.L = L
        self.E = E
        self.J = J
        self.B = E * J  # flexural rigidity

        self._N1 = None
        self._N2 = None
        self.N1 = N1
        self.N2 = N2

        self.x0 = x0
        self.y0 = y0
        self.phi = phi

        self._endX = 0.0
        self._endY = 0.0
        self.endX = dx
        self.endY = dy

        self.m0 = 0.01
        self.theta0 = theta
        self.res = None

    def __str__(self):
        str = 'Fiber:\r\n' \
              '{:8} {:8} {:8} | {:9} {:9} {:9} {:9} | {:8} {:8} {:8} | {:8} {:8} {:8}\r\n' \
            .format('L', 'J', 'E', ' x0', ' y0', ' endX', ' endY', 'N1', 'N2', 'm0', 'phi', 'theta_e', 'lambda')
        str += '{:0.2e} {:0.2e} {:0.2e} | {: .2e} {: .2e} {: .2e} {: .2e} | {:0.2e} {:0.2e} {:0.2e} | ' \
               '{:0.2e} {:0.2e} {:0.2e}' \
            .format(self.L, self.J, self.E,
                    self.x0, self.y0, self.endX, self.endY,
                    self.N1, self.N2, self.m0,
                    self.phi, self.theta0, self.lambda_const())
        return str

    def lambda_const(self):
        F = np.sqrt(self.N1 ** 2 + self.N2 ** 2)
        return F / self.B

    @property
    def endX(self):
        return self._endX

    @endX.setter
    def endX(self, value):
        self._endX = value
        self.L = np.sqrt(self.endX ** 2 + self.endY ** 2)
        logging.debug('setting endX: endX={:02.2f} endY={:02.2f} L={:02.2f}'.format(self._endX, self._endY, self.L))

    @property
    def endY(self):
        return self._endY

    @endY.setter
    def endY(self, value):
        self._endY = value
        self.L = np.sqrt(self.endX ** 2 + self.endY ** 2)
        logging.debug('setting endY: endX={:02.2f} endY={:02.2f} L={:02.2f}'.format(self._endX, self._endY, self.L))

    @property
    def N1(self):
        return self._N1 * self.B / self.L ** 2

    @N1.setter
    def N1(self, value):
        self._N1 = value / self.B * self.L ** 2

    @property
    def N2(self):
        return self._N2 * self.B / self.L ** 2

    @N2.setter
    def N2(self, value):
        self._N2 = value / self.B * self.L ** 2

    def asdict(self):
        return {'L': self.L, 'E': self.E, 'J': self.J,
                'N1': self.N1, 'N2': self.N2, 'x0': self.x0, 'y0': self.y0,
                'xe': self.endX, 'ye': self.endY, 'm0': self.m0,
                'phi': self.phi, 'theta0': self.theta0}

    @property
    def parameters(self):
        params = Parameters()
        params.add('L', value=self.L)
        params.add('E', value=self.E)
        params.add('J', value=self.J, vary=False)
        params.add('N1', value=self.N1)
        params.add('N2', value=self.N2)
        params.add('x0', value=self.x0)
        params.add('y0', value=self.y0)
        params.add('xe', value=self.endX)
        params.add('ye', value=self.endY)
        params.add('m0', value=self.m0)
        params.add('phi', value=self.phi)
        params.add('theta', value=self.theta0)

        return params

    @parameters.setter
    def parameters(self, value):
        if type(value) != Parameters:
            raise Exception('value is not a lmfit parameters object.')
        vals = value.valuesdict()

        self.L = vals['L']
        self.E = vals['E']
        self.J = vals['J']
        self.B = self.E * self.J

        self.N1 = vals['N1']
        self.N2 = vals['N2']

        self.x0 = vals['x0']
        self.y0 = vals['y0']
        self.phi = vals['phi']

        self.endX = vals['xe']
        self.endY = vals['ye']

        self.m0 = vals['m0']
        self.theta0 = vals['theta']
        self.phi = vals['phi']

    def update_ode(self):
        s = np.linspace(0, 1, 1000)
        x0, y0, endX, endY = self.normalized_variables()

        # Now we are ready to run the solver.
        self.res = planar_elastica_bvp_numeric(s, E=self.E, J=self.J, N1=self._N1, N2=self._N2, m0=self.m0,
                                               theta_end=self.theta0, endX=endX, endY=endY)

    def normalized_variables(self):
        return (self.x0 / self.L,
                self.y0 / self.L,
                self.endX / self.L,
                self.endY / self.L)

    def _eval(self, num_points=1000):
        a1 = 0.3
        a2 = 0.8
        logging.debug(
            'evaluating elastica with:\r\n'
            '{:6} {:4} {:4} | '
            '{:8} {:8} {:8} {:8} {:4} | '
            '{:6} {:5} {:5} | '
            '{:5} {:5} {:4}\r\n'
            .format('L', 'a1', 'a2', 'E', 'J', 'N1', 'N2', 'm0', 'theta_e', 'endX', 'endY', 'x0', 'y0', 'phi') +
            '{:< 6.2f} {:04.2f} {:04.2f} | '
            '{:04.2e} {:04.2e} {:04.2e} {:04.2e} {:04.2f} | '
            '{:< 6.2f} {:< 5.2f} {:< 5.2f} | '
            '{: 5.2f} {: 5.2f} {:04.2f}'
            .format(self.L, a1, a2,
                    self.E, self.J, self.N1, self.N2, self.m0,
                    self.theta0, self.endX, self.endY,
                    self.x0, self.y0, self.phi))

        self.update_ode()
        # if not self.res.success: raise Exception('numerical solution did not converge.')
        s = np.linspace(0, 1, num_points)
        r = self.res
        pol = r.sol

        xo = pol(s)[0:2, :]
        xs, ys = eval_planar_elastica(s, pol, a1, a2)[0:2, :]
        ys = np.array([xs, ys])

        # deal with rotations and translations
        x0, y0, _, _ = self.normalized_variables()
        sinphi, cosphi = np.sin(self.phi), np.cos(self.phi)
        M = np.array([[cosphi, -sinphi], [sinphi, cosphi]])
        ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
        xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

        self.curve_x = xo[0] * self.L
        self.curve_y = xo[1] * self.L
        self.curve_subset_x = ys[0] * self.L
        self.curve_subset_y = ys[1] * self.L

        # if np.pi < self.theta0 <= 2 * np.pi:  # TODO: check if this transformation is correct
        #     self.curve_y *= -1
        #     self.curve_subset_y *= -1

    def plot(self, ax, alpha=0.5):
        self._eval(num_points=1000)
        ax.plot(self.curve_x, self.curve_y, lw=1, c='r', alpha=alpha, label='%0.1e' % (self.E * self.J), zorder=4)
        ax.plot(self.curve_subset_x, self.curve_subset_y, lw=3, c='r', alpha=alpha, zorder=4)
        # ax.scatter(self.curve_subset_x, self.curve_subset_y, c='k', marker='+', alpha=alpha, zorder=5)

        L = np.sqrt(np.diff(self.curve_x) ** 2 + np.diff(self.curve_y) ** 2).sum()

        logger.debug(
            'first ({:04.2f},{:04.2f}) '
            'last ({:04.2f},{:04.2f}) '
            'length {:04.2f}'.format(self.curve_x[0], self.curve_y[0],
                                     self.curve_x[-1], self.curve_y[-1],
                                     L))
