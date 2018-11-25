import logging

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters

from ._numerical import eval_planar_elastica, planar_elastica_bvp_numeric

np.set_printoptions(1)
logger = logging.getLogger(__name__)


class PlanarElastica():
    def __init__(self, L=1.0, E=0.625, J=1.0, F=1.0,
                 x0=0.0, y0=0.0, m0=0.01,
                 phi=0.0, theta=3 * np.pi / 2):
        self._L = L
        self.E = E
        self.J = J
        self.B = E * J  # flexural rigidity

        self._F = 0.0
        self._F = F

        self.x0 = x0
        self.y0 = y0
        self.phi = phi

        self._endX = 0.0
        self._endY = 0.0

        self.m0 = m0
        self.theta0 = theta
        self.res = None
        self.curve_x = np.array([0])
        self.curve_y = np.array([0])

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

    @property
    def endY(self):
        return self._endY

    @endX.setter
    def endX(self, value):
        self._endX = value
        self.L = np.sqrt(self.endX ** 2 + self.endY ** 2)
        logging.debug('setting endX: endX={:02.2f} endY={:02.2f} L={:02.2f}'.format(self._endX, self._endY, self.L))

    @endY.setter
    def endY(self, value):
        self._endY = value
        self.L = np.sqrt(self.endX ** 2 + self.endY ** 2)
        logging.debug('setting endY: endX={:02.2f} endY={:02.2f} L={:02.2f}'.format(self._endX, self._endY, self.L))

    @property
    def F(self):
        return self._F * self.B / self.L ** 2

    @F.setter
    def F(self, value):
        self._F = value / self.B * self.L ** 2

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        _F = self.F
        self._L = value
        self._F = _F / self.B * self.L ** 2

    @property
    def parameters(self):
        params = Parameters()
        params.add('L', value=self.L)
        params.add('E', value=self.E)
        params.add('J', value=self.J, vary=False)
        params.add('F', value=self.F)
        params.add('x0', value=self.x0)
        params.add('y0', value=self.y0)
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

        self.F = vals['F']

        self.x0 = vals['x0']
        self.y0 = vals['y0']
        self.phi = vals['phi']

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


class PlanarElasticaDrawObject_xy(plt.Artist):

    def __init__(self, axes, elasticaModel, xini=0.0, yini=0.0, radius=1.0):
        plt.Artist.__init__(self)
        self.ini_angle_pick = False
        self.end_point_pick = False
        self.end_angle_pick = False
        self.fiber = elasticaModel

        self.dx = self.fiber.endX
        self.dy = self.fiber.endY
        self.x0 = xini
        self.y0 = -yini

        self.fiber.x0 = xini
        self.fiber.y0 = -yini
        logger.debug('creating PlanarElasticaDrawObject_xy with fiber model {:s}'.format(str(elasticaModel)))

        self.r = radius

        self.iax = 1.0  # x coordinate for the angle picker point at the initial point
        self.iay = 0.0
        self._theta_i = np.arctan2(self.iay, self.iax)

        self.dax = np.cos(self.fiber.theta0)  # x coordinate for the angle picker point at the end point
        self.day = np.sin(self.fiber.theta0)
        self._theta_e = self.fiber.theta0
        self.ax = axes

        self.ax.set_xlabel('$x [\mu m]$')
        self.ax.set_ylabel('$y [\mu m]$')

        self._initial_plot()
        self.update_picker_point()
        self._connect()

    def __str__(self):
        return "PlanarElastica draw object: x0=%0.2e y0=%0.2e  xe=%0.2e ye=%0.2e dx=%0.2f dy=%0.2f theta_end=%0.2f \r\n" \
               "%s" % (self.x0, self.y0, self.Xe, self.Ye, self.dx, self.dy, self._theta_e, str(self.fiber))

    @property
    def Xe(self):
        out = self._x0 + self.dx
        return out

    @property
    def Ye(self):
        out = self._y0 + self.dy
        return out

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @x0.setter
    def x0(self, value):
        self._x0 = value
        logging.debug('x0 setter: %0.2f  xt= %0.2f+%0.2f = %0.2f fib.L = %0.2f' % (
            value, self._x0, self.dx, self.Xe, self.fiber.L))

    @y0.setter
    def y0(self, value):
        self._y0 = value
        logging.debug('y0 setter: %0.2f  yt= %0.2f+%0.2f = %0.2f fib.L = %0.2f' % (
            value, self._y0, self.dy, self.Ye, self.fiber.L))

    def on_pick(self, event):
        logging.debug(event.artist)
        if self.pick_end_point == event.artist and not self.end_angle_pick:
            self.end_point_pick = True
            mevent = event.mouseevent
            logging.debug('fiber pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))
            self._Ye = mevent.ydata

        if self.ini_angle_circle == event.artist and not self.ini_angle_pick:
            self.ini_angle_pick = True
            mevent = event.mouseevent
            logging.debug('initial angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))

        if self.end_angle_circle == event.artist and not self.end_point_pick:
            self.end_angle_pick = True
            mevent = event.mouseevent
            logging.debug('final angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))

    def on_motion(self, event):
        if event.xdata is None: return

        if self.end_point_pick:
            # deal with rotations and translations
            sinth, costh = np.sin(self._theta_i), np.cos(self._theta_i)
            M = np.array([[costh, -sinth], [sinth, costh]])
            p = np.matmul(M, [event.xdata - self.x0, event.ydata - self.y0])

            self.dx = p[0]
            self.dy = p[1]

        if self.ini_angle_pick:
            self.iax = event.xdata - self.x0
            self.iay = event.ydata - self.y0

        if self.end_angle_pick:
            # deal with rotations and translations
            sinth, costh = np.sin(self._theta_i), np.cos(self._theta_i)
            M = np.array([[costh, -sinth], [sinth, costh]])
            p = np.matmul(M, [event.xdata - self.x0, event.ydata - self.y0])

            self.dax = p[0]
            self.day = p[1]

        self.update_picker_point()
        return

    def on_release(self, event):
        logging.info('on_release {} {} {}'.format(self.end_point_pick, self.ini_angle_pick, self.end_angle_pick))
        if not (self.end_point_pick or self.ini_angle_pick or self.end_angle_pick): return

        self._Xe = event.xdata - self.x0
        self._Ye = event.ydata - self.y0

        logging.debug('x0={: .2f} y0={: .2f} xe={: .2f} ye={: .2f}'.format(self.x0, self.y0, self.Xe, self.Ye))

        if self.ini_angle_pick:
            self.iax = event.xdata - self.x0
            self.iay = event.ydata - self.x0
            logging.debug(
                'ini_angle_pick dx={: .2f} dy={: .2f} theta={: .2f}'.format(self.iax, self.iay, self._theta_i))

        if self.end_angle_pick:
            # deal with rotations and translations
            sinth, costh = -np.sin(self._theta_i), np.cos(self._theta_i)
            M = np.array([[costh, -sinth], [sinth, costh]])
            p = np.matmul(M, [self.Xe, self.Ye]).reshape((2, 1))

            self.dax = event.xdata - p[0][0]
            self.day = event.ydata - p[1][0]

            self.fiber.L = np.sqrt(self.fiber.endX ** 2 + self.fiber.endY ** 2)

            logging.debug(
                'end_angle_pick dx={: .2f} dy={: .2f} theta={: .2f}'.format(self.dax, self.day, self._theta_e))

        self.update_plot()
        self.update_picker_point()
        self.end_point_pick = False
        self.ini_angle_pick = False
        self.end_angle_pick = False
        logging.debug('dx={: .2f} dy={: .2f} theta={: .2f}'.format(self.dax, self.day, self._theta_e))

    def update_plot(self):
        self.fiber.x0 = self.x0
        self.fiber.y0 = self.y0
        self.fiber.endX = self.dx
        self.fiber.endY = self.dy
        self.fiber.theta0 = self._theta_e
        self.fiber.phi = -self._theta_i
        self.fiber.plot(self.ax)
        self.ax.figure.canvas.draw()

    def update_picker_point(self):
        # update initial picker point
        self._theta_i = np.arctan2(self.iay, self.iax)
        thi = np.arctan2(-np.tan(self._theta_i), 1)
        if self.iax < 0:
            thi += np.pi
        irx = np.cos(thi) * self.r
        iry = np.sin(thi) * self.r
        self.ini_angle_circle.center = (self.x0, self.y0)
        self.ini_angle_circle.radius = self.r
        self.ini_angle_point.center = (self.x0 + irx, self.y0 - iry)
        self.ini_angle_point.radius = self.r / 5.0
        self.ini_angle_line.set_data((self.x0, self.x0 + irx), (self.y0, self.y0 - iry))

        # update final picker point
        self._theta_e = np.arctan2(self.day, self.dax)
        the = np.arctan2(-np.tan(self._theta_e), 1)
        if self.dax < 0:
            the += np.pi
        drx = np.cos(the) * self.r
        dry = np.sin(the) * self.r

        # deal with rotations and translations
        sinth, costh = -np.sin(thi), np.cos(thi)
        M = np.array([[costh, -sinth], [sinth, costh]])
        endp = np.matmul(M, [self.dx, self.dy]) + np.array([self.x0, self.y0])
        enda = np.matmul(M, [self.dx + drx, self.dy - dry]) + np.array([self.x0, self.y0])

        self.pick_end_point.center = (endp[0], endp[1])
        self.pick_end_point.radius = self.r / 5.0
        self.end_angle_circle.center = (endp[0], endp[1])
        self.end_angle_circle.radius = self.r
        self.end_angle_point.center = (enda[0], enda[1])
        self.end_angle_point.radius = self.r / 5.0
        self.end_angle_line.set_data((endp[0], enda[0]), (endp[1], enda[1]))

        self.ax.figure.canvas.draw()

    def _initial_plot(self):
        drx = np.cos(self._theta_e) * self.r
        dry = np.sin(self._theta_e) * self.r
        self.pick_end_point = plt.Circle((self.Xe, self.Ye), radius=self.r / 5.0, fc='y', picker=5)
        self.end_angle_circle = plt.Circle((self.Xe, self.Ye), radius=self.r, fc='none', ec='k', lw=1, linestyle='--',
                                           picker=5)
        self.end_angle_point = plt.Circle((self.Xe + drx, self.Ye - dry), radius=self.r / 5.0, fc='w', picker=5)
        self.end_angle_line = plt.Line2D((self.Xe, self.Xe + drx), (self.Ye, self.Ye - dry), color='k')

        irx = np.cos(self._theta_i) * self.r
        iry = np.sin(self._theta_i) * self.r
        self.ini_angle_point = plt.Circle((self.x0 + irx, self.y0 - iry), radius=self.r / 5.0, fc='w', picker=5)
        self.ini_angle_line = plt.Line2D((self.x0, self.x0 + irx), (self.y0, self.y0 - iry), color='k')
        self.ini_angle_circle = plt.Circle((self.x0, self.y0), radius=self.r, fc='none', ec='k', lw=1, linestyle='--',
                                           picker=5)

        self.ax.add_artist(self.pick_end_point)
        self.ax.add_artist(self.end_angle_circle)
        self.ax.add_artist(self.end_angle_point)
        self.ax.add_artist(self.end_angle_line)
        self.ax.add_artist(self.ini_angle_circle)
        self.ax.add_artist(self.ini_angle_point)
        self.ax.add_artist(self.ini_angle_line)
        self.update_plot()

    def _connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def _disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
