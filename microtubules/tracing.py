import logging

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np

from microtubules import elastica as e

np.set_printoptions(1)

coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)


class PlanarElasticaDrawObject_xy(plt.Artist):
    r = 0.1

    def __init__(self, axes, elasticaModel, xini=0.0, yini=0.0):
        plt.Artist.__init__(self)
        self.end_point_pick = False
        self.end_angle_pick = False
        self.fiber = elasticaModel
        self.fiber_line = None

        self._x0 = xini
        self._y0 = -yini
        self.dx = 0.1
        self.dy = 0.6
        self._Theta_e = np.pi / 2
        self.ax = axes

        self.ax.set_xlabel('$x [\mu m]$')
        self.ax.set_ylabel('$y [\mu m]$')

        self._initial_plot()
        self._connect()

    def __str__(self):
        return "PlanarElastica draw object: xe=%0.2f ye=%0.2f dx=%0.2f dy=%0.2f theta_end=%0.2f \r\n" \
               "%s" % (self.Xe, self.Ye, self.dx, self.dy, self._Theta_e, str(self.fiber))

    @property
    def Xe(self):
        out = self._x0 + self.dx
        # logging.debug('Xe getter: %0.2f = -(%0.2f) + %0.2f' % (out, self._Xe, self.Xoffset))
        return out

    @property
    def Ye(self):
        out = self._y0 + self.dy
        # logging.debug('Ye getter: %0.2f = -(%0.2f) + %0.2f' % (out, self._Ye, self.Yoffset))
        return out

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @x0.setter
    def x0(self, value):
        # self._Xe += value
        # self.fiber.endX -= value
        self.fiber.x0 = value
        self._x0 = value
        logging.debug('x0 setter: %0.2f  xt= %0.2f+%0.2f = %0.2f' % (value, self._x0, self.dx, self.Xe))

    @y0.setter
    def y0(self, value):
        # self._Ye += value
        # self.fiber.endY  -= value
        self.fiber.y0 = value
        self._y0 = value
        logging.debug('y0 setter: %0.2f  yt= %0.2f+%0.2f = %0.2f' % (value, self._y0, self.dy, self.Ye))

    def on_pick(self, event):
        logging.debug(event.artist)
        if self.pick_end_point == event.artist:
            self.end_point_pick = True
            mevent = event.mouseevent
            logging.debug('fiber pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))
            self._Ye = mevent.ydata

        if self.angle_circle == event.artist:
            self.end_angle_pick = True
            mevent = event.mouseevent
            logging.debug('angle pick: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                          (mevent.xdata, mevent.ydata, self.x0, self.y0))

    def on_motion(self, event):
        if self.end_point_pick:
            # logging.debug('fiber motion: event.xdata=%f, event.ydata=%f' % (event.xdata, event.ydata))
            self.dx = event.xdata - self.x0
            self.dy = event.ydata - self.y0

        if self.end_angle_pick:
            dax = event.xdata - self.Xe
            day = event.ydata - self.Ye
            self._Theta_e = np.arctan2(day, dax)

        self.update_picker_point()
        return

    def on_release(self, event):
        if not (self.end_point_pick or self.end_angle_pick): return
        logging.debug('fiber release: xdata=%f, ydata=%f, x0=%f, y0=%f ' %
                      (event.xdata, event.ydata, self.x0, self.y0))
        self.end_point_pick = False
        self.end_angle_pick = False
        self._Xe = event.xdata - self.x0
        self._Ye = event.ydata - self.y0
        dax = event.xdata - self.Xe
        day = event.ydata - self.Ye
        self._Theta_e = np.arctan2(day, dax)

        self.update_picker_point()
        self.update_plot()

    def update_plot(self):
        self.fiber.endX = self.dx
        self.fiber.endY = self.dy
        self.fiber.theta0 = self._Theta_e
        self.fiber.update_ode()
        self.fiber.plot(self.ax)
        self.ax.figure.canvas.draw()

    def update_picker_point(self):
        # update picker point
        thetae = np.arctan2(-np.tan(self._Theta_e), 1)
        drx = np.cos(thetae) * self.r
        dry = np.sin(thetae) * self.r
        self.pick_end_point.center = (self.Xe, self.Ye)
        self.angle_circle.center = (self.Xe, self.Ye)
        self.angle_point.center = (self.Xe + drx, self.Ye - dry)
        self.angle_line.set_data((self.Xe, self.Xe + drx), (self.Ye, self.Ye - dry))
        self.ax.draw_artist(self.pick_end_point)
        self.ax.figure.canvas.draw()

    def _initial_plot(self):
        drx = np.cos(self._Theta_e) * self.r
        dry = np.sin(self._Theta_e) * self.r
        self.pick_end_point = plt.Circle((self.Xe, self.Ye), radius=0.05, fc='r', picker=5)
        self.angle_circle = plt.Circle((self.Xe, self.Ye), radius=self.r, fc='none', ec='k', lw=1, linestyle='--',
                                       picker=5)
        self.angle_point = plt.Circle((self.Xe + drx, self.Ye - dry), radius=0.01, fc='k', picker=5)
        self.angle_line = plt.Line2D((self.Xe, self.Xe + drx), (self.Ye, self.Ye - dry), color='k')
        self.ax.add_artist(self.pick_end_point)
        self.ax.add_artist(self.angle_circle)
        self.ax.add_artist(self.angle_point)
        self.ax.add_artist(self.angle_line)
        self.update_plot()

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

        self._whmax = 0.0  # width height
        self.xa = 0.0
        self.ya = 0.0

        self._init_graph()
        self._connect()

    def _init_graph(self):
        self.pick_ini_point = plt.Circle((self.xa, self.ya), radius=0.05, fc='b', picker=5)
        self.ax.add_artist(self.pick_ini_point)

    def add_fiber(self, fiber):
        hpe_xy = PlanarElasticaDrawObject_xy(self.ax, fiber, xini=self.xa, yini=self.ya)
        self.fibers.append(hpe_xy)
        self._reframe()

    def _update_picker_point(self):
        self.pick_ini_point.center = (self.xa, self.ya)
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
        xi, xe = -self._whmax + self.xa, self._whmax + self.xa
        yi, ye = -self._whmax + self.ya, self._whmax + self.ya
        # print 'reframe: maxx=%0.2f maxy=%0.2f' % (maxx, maxy)
        # print 'reframe: xi=%0.2f yi=%0.2f xe=%0.2f ye=%0.2f' % (xi, yi, xe, ye)
        self.ax.set_xlim([xi, xe])
        self.ax.set_ylim([yi, ye])
        self.ax.set_aspect('equal')
        self.ax.figure.canvas.draw()

    def _connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkeyboard = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def _disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
        self.ax.figure.canvas.mpl_disconnect(self.cidkeyboard)

    def on_key(self, event):
        print('press', event.key)
        if event.key == 'r':
            # visible = xl.get_visible()
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.cla()
            self._init_graph()
            for f in self.fibers:
                f._initial_plot()
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.figure.canvas.draw()
        elif event.key == 'w':
            for f in self.fibers:
                print(f)
        elif event.key == 'y':
            logging.debug('creating pymc model')
            import pymc3 as pm
            f = self.fibers[0]
            with pm.Model() as model:
                theta_e = pm.Normal(name="theta_end", mu=f.fiber.theta0, sd=np.pi / 20)
                trace = pm.sample(10, tune=1000, )
            print(trace)
            print(trace.get_values('theta_end'))
            for te in trace.get_values('theta_end'):
                f.fiber.theta0 = te
                f.fiber.update_ode()
                f.fiber.plot(self.ax)
                f.ax.figure.canvas.draw()

    def on_pick(self, event):
        if self.pick_ini_point == event.artist:
            self.picking = True
            mevent = event.mouseevent
            logging.debug('aster pick: xdata=%f, ydata=%f' % (mevent.xdata, mevent.ydata))
            for fib in self.fibers:
                fib.x0 = event.xdata
                fib.y0 = event.ydata
                fib.dx = fib.Xe - event.xdata
                fib.dy = fib.Ye - event.ydata
                fib.update_picker_point()
            self.xa = mevent.xdata
            self.ya = mevent.ydata

    def on_motion(self, event):
        if not self.picking: return
        # logging.debug('aster motion: xdata=%f, ydata=%f' % (event.xdata, event.ydata))
        for fib in self.fibers:
            fib.x0 = event.xdata
            fib.y0 = event.ydata
            fib.update_picker_point()
        self.xa = event.xdata
        self.ya = event.ydata
        self._update_picker_point()

    def on_release(self, event):
        self._reframe()
        if not self.picking: return
        logging.debug('aster release: xdata=%f, ydata=%f' % (event.xdata, event.ydata))
        self.picking = False
        for fib in self.fibers:
            fib.x0 = event.xdata
            fib.y0 = event.ydata
            fib.update_plot()
            fib.update_picker_point()
        self.xa = event.xdata
        self.ya = event.ydata
        self._update_picker_point()


if __name__ == '__main__':
    fiber = e.PlanarElastica()
    fig = plt.figure()
    ax = fig.gca()

    centrosome = Aster(ax)
    centrosome.add_fiber(fiber)

    plt.show()
