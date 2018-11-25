import matplotlib.pyplot as plt
from planar import Affine, Vec2

from ._numerical import *
from ._planar import PlanarElastica

logger = logging.getLogger(__name__)
np.set_printoptions(1)


class PlanarElasticaIVP(PlanarElastica):
    """
          Solves an initial value problem on a dimensionless elastica model

    """

    def __init__(self, w=1.0, k0=1.0, alpha=np.pi / 2,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, phi=0.0, theta=3 * np.pi / 2):
        PlanarElastica.__init__(self, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, phi=phi, theta=theta)
        self._w = w
        self._k0 = k0
        self.alpha = alpha
        self.w_was_calculated = False
        self.k0_was_calculated = False

    def __str__(self):
        str = 'PlanarElasticaIVP:\r\n' \
              '{:8} {:8} {:8} {:8} | {:9} {:9} {:9} {:9} | {:8} {:8} | {:8} {:8} {:8}\r\n' \
            .format('L', 'J', 'E', 'B', ' x0', ' y0', ' endX', ' endY', 'F', 'm0', 'phi', 'theta_e', 'lambda')
        str += '{:0.2e} {:0.2e} {:0.2e} {:0.2e} | {: .2e} {: .2e} {: .2e} {: .2e} | {:0.2e} {:0.2e} | ' \
               '{:0.2e} {:0.2e} {:0.2e}' \
            .format(self.L, self.J, self.E, self.B,
                    self.x0, self.y0, self.endX, self.endY,
                    self.F, self.m0,
                    self.phi, self.theta0, self.lambda_const())
        return str

    @property
    def w(self):
        return self._w

    @property
    def k0(self):
        return self._k0

    @w.setter
    def w(self, value):
        if value is not None:
            self.w_was_calculated = False
            self._w = value
        else:
            self.w_was_calculated = True
            self._w = np.sqrt(self.F * self.L ** 2 / self.B)

    @k0.setter
    def k0(self, value):
        if value is not None:
            self.k0_was_calculated = False
            self._k0 = value
        else:
            self.k0_was_calculated = True
            self._k0 = self.m0 * self.L / self.B

    @property
    def parameters(self):
        params = super().parameters
        params.add('w', value=self.w)
        params.add('k0', value=self.k0)
        params.add('alpha', value=self.alpha)

        return params

    @parameters.setter
    def parameters(self, value):
        PlanarElastica.parameters = value
        vals = value.valuesdict()

        self._w = vals['w']
        self._k0 = vals['k0']
        self.alpha = vals['alpha']
        self.w_was_calculated = False
        self.k0_was_calculated = False

    def eval(self, num_points=1000, log=True):
        super().eval(num_points=num_points, log=False)

        # create params for solving the ODE
        s = np.linspace(0, 1, num_points)
        logger.debug('w={:04.2e} k0={:04.2e}'.format(self.w, self.k0))

        # Now we are ready to run the solver.
        r = planar_elastica_ivp_numeric(s, w=self.w, g=self.theta0, k0=self.k0, alpha=self.alpha)
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


class PlanarElasticaIVPArtist(PlanarElasticaIVP, plt.Artist):

    def __init__(self, axes, radius=1.0,
                 w=1.0, k0=1.0, alpha=np.pi / 2,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, phi=0.0, theta=3 * np.pi / 2):
        plt.Artist.__init__(self)
        PlanarElasticaIVP.__init__(self, w=w, k0=k0, alpha=alpha, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, theta=theta)
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

        self.ax.set_xlabel('$x [\mu m]$')
        self.ax.set_ylabel('$y [\mu m]$')

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
