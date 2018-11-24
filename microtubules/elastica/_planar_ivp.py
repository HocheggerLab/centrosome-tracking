from lmfit import Parameters
import matplotlib.pyplot as plt
from planar import Affine, Vec2

from ._numerical import *

logger = logging.getLogger(__name__)
np.set_printoptions(1)


class PlanarElasticaIVP():

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
              '{:8} {:8} {:8} {:8} | {:9} {:9} {:9} {:9} | {:8} {:8} | {:8} {:8} {:8}\r\n' \
            .format('L', 'J', 'E', 'B', ' x0', ' y0', ' endX', ' endY', 'F', 'm0', 'phi', 'theta_e', 'lambda')
        str += '{:0.2e} {:0.2e} {:0.2e} {:0.2e} | {: .2e} {: .2e} {: .2e} {: .2e} | {:0.2e} {:0.2e} | ' \
               '{:0.2e} {:0.2e} {:0.2e}' \
            .format(self.L, self.J, self.E, self.B,
                    self.x0, self.y0, self.endX, self.endY,
                    self.F, self.m0,
                    self.phi, self.theta0, self.lambda_const())
        return str

    def lambda_const(self):
        return self.F / self.B

    @property
    def endX(self):
        return self._endX

    @property
    def endY(self):
        return self._endY

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

    def asdict(self):
        return {'L': self.L, 'E': self.E, 'J': self.J,
                'F': self.F, 'x0': self.x0, 'y0': self.y0,
                'xe': self.endX, 'ye': self.endY, 'm0': self.m0,
                'phi': self.phi, 'theta0': self.theta0}

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

    def eval(self, num_points=1000):
        logger.debug(
            'evaluating elastica with:\r\n'
            '{:6} {:8} {:8} {:8} {:8} {:5} | '
            '{:6} {:5} {:5} | '
            '{:5} {:5} {:4}\r\n'
            .format('L', 'E', 'J', 'B', 'F', 'm0',
                    'theta_e', 'endX', 'endY',
                    'x0', 'y0', 'phi') +
            '{:< 6.2f} {:04.2e} {:04.2e} {:04.2e} {:04.2e} {: 4.2f} | '
            '{:< 6.2f} {:< 5.2f} {:< 5.2f} | '
            '{: 5.2f} {: 5.2f} {:04.2f}'
            .format(self.L, self.E, self.J, self.B, self.F, self.m0,
                    self.theta0, self.endX, self.endY,
                    self.x0, self.y0, self.phi))

        if self.F < 0:
            self.F *= -1
            self.theta0 += 2 * np.pi

        # create params for solving the ODE
        s = np.linspace(0, 1, num_points)
        self._w = np.sqrt(self.F * self.L ** 2 / self.B)
        self._k0 = self.m0 * self.L / self.B
        logger.debug('w={:04.2e} k0={:04.2e}'.format(self._w, self._k0))

        # Now we are ready to run the solver.
        r = planar_elastica_ivp_numeric(s, w=self._w, g=self.theta0, k0=self._k0, alpha=np.pi / 2)
        self.res = r
        xo = r.y[0:2, :]

        # deal with rotations and translations
        x0, y0 = self.x0 / self.L, self.y0 / self.L
        sinphi, cosphi = np.sin(self.phi), np.cos(self.phi)
        M = np.array([[cosphi, -sinphi], [sinphi, cosphi]])
        xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

        self.curve_x = xo[0] * self.L
        self.curve_y = xo[1] * self.L
        self._endX = xo[0, -1] * self.L
        self._endY = xo[1, -1] * self.L

    def plot(self, ax, alpha=0.5, num_points=100):
        # self.eval(num_points=num_points)
        if np.isnan(self.curve_x).any() or np.isnan(self.curve_y).any():
            raise Exception("NaNs in computed curves, can't plot")

        ax.plot(self.curve_x, self.curve_y, lw=1, c='r', alpha=alpha, label='%0.1e' % (self.E * self.J), zorder=4)

        L = np.sqrt(np.diff(self.curve_x) ** 2 + np.diff(self.curve_y) ** 2).sum()

        logger.debug(
            'first ({:04.2f},{:04.2f}) '
            'last ({:04.2f},{:04.2f}) '
            'length {:04.2f}'.format(self.curve_x[0], self.curve_y[0],
                                     self.curve_x[-1], self.curve_y[-1],
                                     L))


class PlanarElasticaIVPArtist(PlanarElasticaIVP, plt.Artist):

    def __init__(self, axes, radius=1.0,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, theta=3 * np.pi / 2):
        plt.Artist.__init__(self)
        PlanarElasticaIVP.__init__(self, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, theta=theta)
        self.ini_angle_pick = False
        self.end_angle_pick = False

        self.r = radius

        self.phi = np.arctan2(1.0, 0.0)

        self.ini_pt = Vec2(self.x0, self.y0)
        self.end_pt = Vec2(self.endX, self.endY)
        # coordinates for the angle picker points at initial and end points
        self.phi_angle_pt = Vec2(np.cos(self.phi), np.sin(self.phi))
        self.theta_angle_pt = Vec2(np.cos(self.theta0), np.sin(self.theta0))

        self.ax = axes

        self.ax.set_xlabel('$x [\mu m]$')
        self.ax.set_ylabel('$y [\mu m]$')

        self.t = Affine.translation(Vec2(x0, y0))
        self.R = Affine.rotation(np.rad2deg(self.phi))
        self.Rt = self.R * self.t

        self._initial_plot()
        self.update_picker_point()
        self._connect()

    def __str__(self):
        return "PlanarElastica draw object: %s" % (str(PlanarElasticaIVP))

    def on_pick(self, event):
        logging.debug(event.artist)
        if self.ini_angle_circle == event.artist and not self.ini_angle_pick:
            self.ini_angle_pick = True
            mevent = event.mouseevent
            logging.debug('initial angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))

        if self.end_picker_perimeter == event.artist and not self.end_angle_pick:
            self.end_angle_pick = True
            mevent = event.mouseevent
            logging.debug('final angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))

    def on_motion(self, event):
        if event.xdata is None: return

        if self.ini_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.phi_angle_pt = xmouse - self.ini_pt

        if self.end_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.theta_angle_pt = (xmouse - self.end_pt) * self.R.__invert__()

        self.update_picker_point()
        return

    def on_release(self, event):
        logging.info('on_release {} {}'.format(self.ini_angle_pick, self.end_angle_pick))
        if not (self.ini_angle_pick or self.end_angle_pick): return

        logging.debug('x0={: .2f} y0={: .2f} xe={: .2f} ye={: .2f}'.format(self.x0, self.y0, self.endX, self.endY))

        if self.ini_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.phi_angle_pt = xmouse - self.ini_pt
            logging.debug('ini_angle_pick')

        if self.end_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.theta_angle_pt = (xmouse - self.end_pt) * self.R.__invert__()

            logging.debug('end_angle_pick')

        self.eval()
        self.plot(self.ax)
        self.update_picker_point()
        self.ini_angle_pick = False
        self.end_angle_pick = False
        logging.debug('angle_pt={:s} theta={: .2f}'.format(str(self.theta_angle_pt), self.theta0))

    def update_picker_point(self):
        # update initial picker point
        self.ini_pt = Vec2(self.x0, self.y0)
        self.phi = np.arctan2(self.phi_angle_pt.y, self.phi_angle_pt.x)
        self.phi_angle_pt = Vec2(np.cos(self.phi) * self.r, np.sin(self.phi) * self.r)
        phi_ang_pt = self.phi_angle_pt + self.ini_pt

        self.ini_angle_circle.center = (self.x0, self.y0)
        self.ini_angle_circle.radius = self.r
        self.ini_angle_point.center = (phi_ang_pt.x, phi_ang_pt.y)
        self.ini_angle_point.radius = self.r / 5.0
        self.ini_angle_line.set_data((self.x0, phi_ang_pt.x), (self.y0, phi_ang_pt.y))

        # update final picker point
        self.theta0 = np.arctan2(self.theta_angle_pt.y, self.theta_angle_pt.x)  # +np.pi

        # deal with rotations and translations
        self.end_pt = Vec2(self.endX, self.endY)
        pc = self.end_pt
        pick_e = Vec2(np.cos(self.theta0) * self.r, np.sin(self.theta0) * self.r)
        pe = pick_e * self.R * Affine.translation(self.end_pt)

        self.end_pick_point.center = (pc.x, pc.y)
        self.end_picker_perimeter.center = (pc.x, pc.y)
        self.end_angle_point.center = (pe.x, pe.y)
        self.end_angle_line.set_data((pe.x, pc.x), (pe.y, pc.y))

        self.end_picker_perimeter.radius = self.r
        self.end_pick_point.radius = self.r / 5.0
        self.end_angle_point.radius = self.r / 5.0

        self.ax.figure.canvas.draw()

    def _initial_plot(self):
        drx = np.cos(self.theta0) * self.r
        dry = np.sin(self.theta0) * self.r
        self.end_pick_point = plt.Circle((self.endX, self.endY), radius=self.r / 10.0, fc='b')
        self.end_picker_perimeter = plt.Circle((self.endX, self.endY), radius=self.r, fc='none', ec='k', lw=1,
                                               linestyle='--', picker=5)
        self.end_angle_point = plt.Circle((self.endX + drx, self.endY - dry), radius=self.r / 5.0, fc='w', picker=5)
        self.end_angle_line = plt.Line2D((self.endX, self.endX + drx), (self.endY, self.endY - dry), color='k')

        irx = np.cos(self.phi) * self.r
        iry = np.sin(self.phi) * self.r
        self.ini_angle_point = plt.Circle((self.x0 + irx, self.y0 - iry), radius=self.r / 5.0, fc='w', picker=5)
        self.ini_angle_line = plt.Line2D((self.x0, self.x0 + irx), (self.y0, self.y0 - iry), color='k')
        self.ini_angle_circle = plt.Circle((self.x0, self.y0), radius=self.r, fc='none', ec='k', lw=1, linestyle='--',
                                           picker=5)

        self.ax.add_artist(self.end_pick_point)
        self.ax.add_artist(self.end_picker_perimeter)
        self.ax.add_artist(self.end_angle_point)
        self.ax.add_artist(self.end_angle_line)
        self.ax.add_artist(self.ini_angle_circle)
        self.ax.add_artist(self.ini_angle_point)
        self.ax.add_artist(self.ini_angle_line)
        self.eval()
        self.plot(self.ax)

    def _hide_picker(self):
        self.end_pick_point.set_visible(False)
        self.end_picker_perimeter.set_visible(False)
        self.end_angle_point.set_visible(False)
        self.end_angle_line.set_visible(False)

        self.ini_angle_line.set_visible(False)
        self.ini_angle_point.set_visible(False)
        self.ini_angle_circle.set_visible(False)

    def _show_picker(self):
        self.end_pick_point.set_visible(True)
        self.end_picker_perimeter.set_visible(True)
        self.end_angle_point.set_visible(True)
        self.end_angle_line.set_visible(True)

        self.ini_angle_line.set_visible(True)
        self.ini_angle_point.set_visible(True)
        self.ini_angle_circle.set_visible(True)

    def _connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def _disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
