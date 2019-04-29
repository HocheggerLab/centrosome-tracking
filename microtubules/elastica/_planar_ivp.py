from planar import Affine, Vec2
from sympy import *
from pint import UnitRegistry
from lmfit import Parameters
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from ._numerical import *
from ._planar import PlanarElastica

logger = logging.getLogger(__name__)
np.set_printoptions(1)
ur = UnitRegistry()


class PlanarElasticaIVP(PlanarElastica):
    """
          Solves an initial value problem on a dimensionless elastica model

    """

    def __init__(self, w=1.0, k0=1.0, alpha=np.pi / 2,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, phi=0.0, theta=3 * np.pi / 2):
        PlanarElastica.__init__(self, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, phi=phi, theta=theta)
        self.phi = phi
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
        params = Parameters()
        params.add('L', value=self.L)
        params.add('x0', value=self.x0)
        params.add('y0', value=self.y0)
        params.add('phi', value=self.phi)

        params.add('w', value=self.w)
        params.add('k0', value=self.k0)
        params.add('alpha', value=self.alpha)

        return params

    @parameters.setter
    def parameters(self, value):
        vals = value.valuesdict()

        self.L = vals['L']
        self.x0 = vals['x0']
        self.y0 = vals['y0']
        self.phi = vals['phi']

        self._w = vals['w']
        self._k0 = vals['k0']
        self.alpha = vals['alpha']
        self.w_was_calculated = False
        self.k0_was_calculated = False

    def eval(self, num_points=1000):
        super().eval(num_points=num_points)

        # create params for solving the ODE
        s = np.linspace(0, 1, num_points)
        # logger.debug('w={:04.2e} k0={:04.2e}'.format(self.w, self.k0))

        # Now we are ready to run the solver.
        r = planar_elastica_ivp_numeric(s, w=self.w, gamma=self.theta0, k0=self.k0, alpha=self.alpha)
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
        self.angle_phi = r.y[2, :]
        self.curvature_k = r.y[3, :]

    def force(self, EI=10e-24 * ur.N * ur.m ** 2, num_fibers=10):
        I = ur.pi / 4 * ((12 * ur.nm) ** 4 - (8 * ur.nm) ** 4)
        I.ito(ur.um ** 4)

        # is_force_dominant = sin(self.alpha / 2) ** 2 + (self._k0 / 2 / self.w) ** 2 < 1
        # print('The solution %s force dominant.' % ('is' if is_force_dominant else 'is not'))
        # print('I = %s' % str(I))

        M = EI * self.curvature_k * self.L / ur.um
        M.ito(ur.N * ur.m)

        w, F, L, E, _I, B = symbols('w F L E I B')
        N = Symbol('N', integer=True)
        eq = Eq(w ** 2, F * L ** 2 / (B * N ** 2))
        sol_EI = solve(eq.subs({L: self.L, w: self.w}), B)[0]
        # logger.debug('EI = ' + str(EI) + ' = ' + str(sol_EI))

        slope = sol_EI.subs({F: 1, N: 1}) * ur.um ** 2
        slope.ito(ur.m ** 2)
        # logger.debug('slope = ' + str(slope))

        F = EI * num_fibers ** 2 / slope
        F.ito(ur.pN)
        return F.m

    def forces(self, EI=10e-24 * ur.N * ur.m ** 2, num_fibers=10):
        I = ur.pi / 4 * ((12 * ur.nm) ** 4 - (8 * ur.nm) ** 4)
        I.ito(ur.um ** 4)

        is_force_dominant = sin(self.alpha / 2) ** 2 + (self._k0 / 2 / self.w) ** 2 < 1
        print('The solution %s force dominant.' % ('is' if is_force_dominant else 'is not'))
        print('I = %s' % str(I))

        fig = plt.figure()
        ax = fig.gca()
        M = EI * self.curvature_k * self.L / ur.um
        M.ito(ur.N * ur.m)
        sct = ax.scatter(self.curve_x, self.curve_y, s=2, c=M, vmin=min(M), vmax=max(M), cmap='PiYG')
        fig.colorbar(sct, ax=ax)

        w, F, L, E, _I, B = symbols('w F L E I B')
        N = Symbol('N', integer=True)
        eq = Eq(w ** 2, F * L ** 2 / (B * N ** 2))
        sol_EI = solve(eq.subs({L: self.L, w: self.w}), B)[0]
        print('number of fibers = %d' % num_fibers)
        print('EI = ' + str(EI) + ' = ' + str(sol_EI))

        fig = plt.figure()
        ax = fig.gca()
        norm = mpl.colors.Normalize(vmin=0, vmax=num_fibers)
        color = cm.ScalarMappable(norm=norm, cmap='PiYG')
        slope = sol_EI.subs({F: 1, N: 1}) * ur.um ** 2
        slope.ito(ur.m ** 2)
        print('slope = ' + str(slope))

        f_ = np.linspace(0, 40, num=10 ** 3) * ur.pN
        for n in range(1, num_fibers + 2):
            _EI = slope * f_.to(ur.N) / n ** 2
            _EI.ito(ur.N * ur.m ** 2)
            ax.plot(f_, _EI, c=color.to_rgba(n), label=n)
            if n == 10:
                _F = EI * n ** 2 / slope
                _F.ito(ur.pN)
                ax.axvline(_F.magnitude, color=color.to_rgba(n), linestyle='--')
                print('F = ' + str(_F))

        ax.legend()
        ax.axhline(EI.to(ur.N * ur.m ** 2).magnitude, color='k', linestyle='--')
        ax.set_xlabel('F [pN]')
        ax.set_ylabel('EI [Nm^2]')
        ax.set_yscale('log')
        locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.yaxis.set_minor_locator(locmin)

        plt.show(block=False)


class PlanarElasticaIVPArtist(PlanarElasticaIVP, plt.Artist):

    def __init__(self, axes, radius=1.0,
                 w=1.0, k0=1.0, alpha=np.pi / 2,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, phi=0.0, theta=3 * np.pi / 2,
                 callback=None):
        plt.Artist.__init__(self)
        PlanarElasticaIVP.__init__(self, w=w, k0=k0, alpha=alpha, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, theta=theta)
        self.ini_angle_pick = False
        self.end_angle_pick = False
        self.selected = False

        self.r = radius
        self.phi = phi

        self._ini_pt = Vec2(self.x0, self.y0)
        self._end_pt = Vec2(self.endX, self.endY)
        # rigid transformations based on phi
        self.t = Affine.translation(Vec2(x0, y0))

        self.ax = axes

        self.callback = callback

        self._initial_plot()
        self.update_picker_point()
        self._connect()

    @property
    def theta0(self):
        return self._theta0

    @theta0.setter
    def theta0(self, value):
        self._theta0 = value
        self.theta_angle_pt = Vec2(np.cos(value), np.sin(value))

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value
        # vectors and rigid transformations based on phi
        self.phi_angle_pt = Vec2(np.cos(self._phi), np.sin(self._phi))
        self.R = Affine.rotation(np.rad2deg(self._phi))
        logger.debug(self.R.is_degenerate)
        self.R.__invert__()
        # print("coudn't invert R for phi=%0.4f"%value)

    @property
    def Rt(self):
        return self.R * self.t

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        self._x0 = value
        self._ini_pt = Vec2(self.x0, self.y0)
        self.ini_angle_circle.center = (self.x0, self.y0)

    @property
    def y0(self):
        return self._y0

    @y0.setter
    def y0(self, value):
        self._y0 = value
        self._ini_pt = Vec2(self.x0, self.y0)
        self.ini_angle_circle.center = (self.x0, self.y0)

    @property
    def picked(self):
        return self.ini_angle_pick or self.end_angle_pick

    def __str__(self):
        return "PlanarElastica draw object: %s" % (super().__str__())

    def select(self):
        self.selected = True
        self._show_picker()

    def unselect(self):
        self.selected = False
        self._hide_picker()

    def on_pick(self, event):
        if not self.selected: return
        if type(event.artist) == plt.Line2D: return
        logger.debug('on pick %s' % event.artist)
        # if self.curve == event.artist and not (self.ini_angle_pick or self.end_angle_pick):
        #     self.select()
        if self.ini_angle_circle == event.artist and not self.ini_angle_pick:
            self.ini_angle_pick = True
            mevent = event.mouseevent
            logger.debug('initial angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                         (mevent.xdata, mevent.ydata, self.x0, self.y0))

        if self.end_picker_perimeter == event.artist and not self.end_angle_pick:
            self.end_angle_pick = True
            mevent = event.mouseevent
            logger.debug('final angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                         (mevent.xdata, mevent.ydata, self.x0, self.y0))

    def on_motion(self, event):
        if not self.selected: return
        if not (self.ini_angle_pick or self.end_angle_pick): return
        if event.xdata is None: return

        if self.ini_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.phi_angle_pt = xmouse - self._ini_pt

        if self.end_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.theta_angle_pt = (xmouse - self._end_pt) * self.R.__invert__()

        self.update_picker_point()
        self.ax.figure.canvas.draw()
        return

    def on_release(self, event):
        if not self.selected: return
        logger.info('on_release {} {}'.format(self.ini_angle_pick, self.end_angle_pick))
        if not (self.ini_angle_pick or self.end_angle_pick): return

        logger.debug('x0={: .2f} y0={: .2f} xe={: .2f} ye={: .2f}'.format(self.x0, self.y0, self.endX, self.endY))

        if self.ini_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.phi_angle_pt = xmouse - self._ini_pt
            self.phi = np.arctan2(self.phi_angle_pt.y, self.phi_angle_pt.x)
            self.phi_angle_pt = Vec2(np.cos(self.phi) * self.r, np.sin(self.phi) * self.r)
            logger.debug('ini_angle_release')

        if self.end_angle_pick:
            xmouse = Vec2(event.xdata, event.ydata)
            self.theta_angle_pt = (xmouse - self._end_pt) * self.R.__invert__()
            logger.debug('end_angle_release')

        self.eval()
        self.plot(self.ax)
        self.update_picker_point()
        self.ini_angle_pick = False
        self.end_angle_pick = False
        logger.debug('angle_pt={:s} theta={: .2f}'.format(str(self.theta_angle_pt), self.theta0))

        if self.callback is not None: self.callback()
        self.ax.figure.canvas.draw()

    def update_picker_point(self):
        # update initial picker point
        phi_ang_pt = self.phi_angle_pt + self._ini_pt

        self.ini_angle_circle.radius = self.r
        self.ini_angle_point.center = (phi_ang_pt.x, phi_ang_pt.y)
        self.ini_angle_line.set_data((self.x0, phi_ang_pt.x), (self.y0, phi_ang_pt.y))

        # update final picker point
        self.theta0 = np.arctan2(self.theta_angle_pt.y, self.theta_angle_pt.x)  # +np.pi

        # deal with rotations and translations
        self._end_pt = Vec2(self.endX, self.endY)
        pc = self._end_pt
        pick_e = Vec2(np.cos(self.theta0) * self.r, np.sin(self.theta0) * self.r)
        pe = pick_e * self.R * Affine.translation(self._end_pt)

        self.end_pick_point.center = (pc.x, pc.y)
        self.end_picker_perimeter.center = (pc.x, pc.y)
        self.end_angle_point.center = (pe.x, pe.y)
        self.end_angle_line.set_data((pe.x, pc.x), (pe.y, pc.y))

    def _initial_plot(self):
        ps = 0.2
        drx = np.cos(self.theta0) * self.r
        dry = np.sin(self.theta0) * self.r
        self.end_pick_point = plt.Circle((self.endX, self.endY), radius=ps, fc='b', zorder=10)
        self.end_picker_perimeter = plt.Circle((self.endX, self.endY), picker=1, radius=1, fc='none', ec='k', lw=0.5,
                                               linestyle='--', zorder=10)
        self.end_angle_point = plt.Circle((self.endX + drx, self.endY - dry), picker=ps, radius=ps, fc='w')
        self.end_angle_line = plt.Line2D((self.endX, self.endX + drx), (self.endY, self.endY - dry), color='k')

        irx = np.cos(self.phi) * self.r
        iry = np.sin(self.phi) * self.r
        self.ini_angle_point = plt.Circle((self.x0 + irx, self.y0 - iry), picker=ps, radius=ps, fc='w', zorder=10)
        self.ini_angle_line = plt.Line2D((self.x0, self.x0 + irx), (self.y0, self.y0 - iry), color='k')
        self.ini_angle_circle = plt.Circle((self.x0, self.y0), picker=1, radius=1, fc='none', ec='k', lw=0.5,
                                           linestyle='--', zorder=10)

        self.ax.add_artist(self.end_pick_point)
        self.ax.add_artist(self.end_picker_perimeter)
        self.ax.add_artist(self.end_angle_point)
        self.ax.add_artist(self.end_angle_line)
        self.ax.add_artist(self.ini_angle_circle)
        self.ax.add_artist(self.ini_angle_point)
        self.ax.add_artist(self.ini_angle_line)

        self.ax.set_xlabel('$x [\mu m]$')
        self.ax.set_ylabel('$y [\mu m]$')

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
        self.update_picker_point()
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
