import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_bvp

np.set_printoptions(1)


class HeavyPlanarElastica():
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


class HeavyPlanarElasticaDrawObject_xy(plt.Artist):
    def __init__(self, axes, elasticaModel, xini=0.0, yini=0.0):
        plt.Artist.__init__(self)
        self.picking = False
        self.fiber = elasticaModel
        self.fiber_line = None

        self.Xoffset = xini
        self._Yoffset = -yini
        self.Xe = 1
        self._Ye = 1
        self.ax = axes

        self.ax.set_xlabel('$x [\mu m]$')
        self.ax.set_ylabel('$y [\mu m]$')

        self._initial_plot()
        self._connect()

    @property
    def Ye(self):
        out = self._Ye + self._Yoffset
        print 'Ye getter: %0.2f = -(%0.2f) + %0.2f' % (out, self._Ye, self._Yoffset)
        return out

    @property
    def Yoffset(self):
        print 'Yoffset getter: %0.2f' % self._Yoffset
        return self._Yoffset

    @Yoffset.setter
    def Yoffset(self, value):
        print 'Yoffset setter'
        self._Yoffset = -value

    def on_pick(self, event):
        if self.pick_end_point == event.artist:
            self.picking = True
            mevent = event.mouseevent
            print('fiber pick: x=%d, y=%d, xdata=%f, ydata=%f' % (mevent.x, mevent.y, mevent.xdata, mevent.ydata))
            self._Ye = mevent.ydata

    def on_motion(self, event):
        if not self.picking: return
        # print('fiber motion: picking=%f, event.xdata=%f, event.ydata=%f' % (self.picking, event.xdata, event.ydata))
        self._Ye = event.ydata
        self._update_picker_point()
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        if not self.picking: return
        print('fiber release.')
        self.picking = False
        self._Ye = -event.ydata
        self.fiber.endY = self._Ye - self.Yoffset
        self.update_plot()

    def _update_ode_plot(self):
        self.fiber.update_ode()

        s_plot = np.linspace(0, self.fiber.L, 100)
        x_plot = self.fiber.res.sol(s_plot)[0] + self.Xoffset
        y_plot = -self.fiber.res.sol(s_plot)[1] - self.Yoffset

        self.Xe = x_plot[0]
        self._Ye = y_plot[0]
        print 'xe = %0.2f, ye = %0.2f, m0 = %0.2f' % (self.Xe, self._Ye, self.fiber.m0)

        if self.fiber_line is None:
            self.fiber_line, = self.ax.plot(x_plot, y_plot, label='fiber')
        else:
            self.fiber_line.set_data(x_plot, y_plot)

            # self.ax.set_aspect('equal', 'datalim')

    def _update_picker_point(self):
        self.pick_end_point.center = (self.Xe, self._Ye)
        self.ax.draw_artist(self.pick_end_point)

    def update_plot(self):
        self._update_ode_plot()
        self._update_picker_point()
        self.ax.figure.canvas.draw()

    def _initial_plot(self):
        self._update_ode_plot()

        self.pick_end_point = plt.Circle((self.Xe, self._Ye), radius=0.05, fc='r', picker=5)
        self.ax.add_artist(self.pick_end_point)

    def _connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def _disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)


class Aster():
    def __init__(self, axes):
        self.picking = False
        self.ax = axes
        self.fibers = []

        self._whmax = None  # width height
        self.Xi = 0.0
        self.Yi = 0.0
        self.pick_ini_point = plt.Circle((self.Xi, self.Yi), radius=0.05, fc='b', picker=5)
        self.ax.add_artist(self.pick_ini_point)

        self._connect()

    def add_fiber(self, fiber):
        hpe_xy = HeavyPlanarElasticaDrawObject_xy(self.ax, fiber, xini=self.Xi, yini=self.Yi)
        self.fibers.append(hpe_xy)
        self._reframe()

    def _update_picker_point(self):
        self.pick_ini_point.center = (self.Xi, self.Yi)
        self.ax.draw_artist(self.pick_ini_point)
        self.ax.figure.canvas.draw()

    def _reframe(self):
        maxx, maxy = 0, 0
        for fib in self.fibers:
            maxx = np.sqrt(fib.Xe ** 2) if np.sqrt(fib.Xe ** 2) > maxx else maxx
            maxy = np.sqrt(fib.Ye ** 2) if np.sqrt(fib.Ye ** 2) > maxy else maxy
        maxx *= 1.2
        maxy *= 1.2

        self._whmax = np.max([maxx, maxy, self._whmax])
        xi, xe = -self._whmax + self.Xi, self._whmax + self.Xi
        yi, ye = -self._whmax + self.Yi, self._whmax + self.Yi
        print 'reframe: maxx=%0.2f maxy=%0.2f' % (maxx, maxy)
        print 'reframe: xi=%0.2f yi=%0.2f xe=%0.2f ye=%0.2f' % (xi, yi, xe, ye)
        self.ax.set_xlim([xi, xe])
        self.ax.set_ylim([yi, ye])
        self.ax.set_aspect('equal')
        self.ax.figure.canvas.draw()

    def _connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def _disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_pick(self, event):
        if self.pick_ini_point == event.artist:
            self.picking = True
            mevent = event.mouseevent
            print('aster pick: x=%d, y=%d, xdata=%f, ydata=%f' % (mevent.x, mevent.y, mevent.xdata, mevent.ydata))
            self.Xi = mevent.xdata
            self.Yi = mevent.ydata

    def on_motion(self, event):
        if not self.picking: return
        # print('aster motion: x=%f, y=%f, xdata=%f, ydata=%f' % (event.x, event.y, event.xdata, event.ydata))
        self.Xi = event.xdata
        self.Yi = event.ydata
        self._update_picker_point()

    def on_release(self, event):
        self._reframe()
        if not self.picking: return
        print('aster release.')
        self.picking = False
        self.Xi = event.xdata
        self.Yi = event.ydata
        for fib in self.fibers:
            fib.Xoffset = event.xdata
            fib.Yoffset = event.ydata
            fib.update_plot()
        self._update_picker_point()


if __name__ == '__main__':
    fiber = HeavyPlanarElastica()
    fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    ax = fig.gca()

    centrosome = Aster(ax)
    centrosome.add_fiber(fiber)

    plt.show()
