import logging

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

from microtubules import elastica as e

np.set_printoptions(1)
coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger('elastica')
logging.getLogger('matplotlib').setLevel(level=logging.INFO)


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

        self.end_point_pick = False
        self.ini_angle_pick = False
        self.end_angle_pick = False
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


class Aster:
    def __init__(self, axes, x0=0.0, y0=0.0):
        self.picking = False
        self.ax = axes
        # self.fig = axes.get_figure()
        self.fibers = []

        self._whmax = 0.0  # width height
        self.xa = x0
        self.ya = y0

        # fig_size = self.fig.get_size_inches()
        # self.r = min(fig_size) / 2.5
        ax_size_x = np.diff(self.ax.get_xticks())[0]
        self.r = ax_size_x

        self._init_graph()
        self._connect()

    def _init_graph(self):
        self.pick_ini_point = plt.Circle((self.xa, self.ya), radius=self.r / 5.0, fc='w', picker=5)
        self.ax.add_artist(self.pick_ini_point)

    def add_fiber(self, fiber):
        hpe_xy = PlanarElasticaDrawObject_xy(self.ax, fiber, xini=self.xa, yini=-self.ya, radius=self.r)

        self.fibers.append(hpe_xy)
        self._reframe()

    def _update_picker_point(self):
        self.pick_ini_point.center = (self.xa, self.ya)
        self.pick_ini_point.radius = (self.r / 5.0)
        for f in self.fibers:
            f.r = self.r
            f.update_picker_point()
        self.ax.figure.canvas.draw()

    def _reframe(self):
        return
        maxx, maxy = 0, 0
        for fib in self.fibers:
            maxx = np.sqrt(fib.Xe ** 2) if np.sqrt(fib.Xe ** 2) > maxx else maxx
            maxy = np.sqrt(fib.Ye ** 2) if np.sqrt(fib.Ye ** 2) > maxy else maxy
        maxx *= 1.5
        maxy *= 1.5

        self._whmax = np.max([maxx, maxy, self._whmax])
        xi, xe = -self._whmax + self.xa, self._whmax + self.xa
        yi, ye = -self._whmax + self.ya, self._whmax + self.ya

        self.ax.set_xlim([xi, xe])
        self.ax.set_ylim([yi, ye])
        self.ax.set_aspect('equal')

        # computing radius for picker elements
        print(self.ax.get_xticks())
        ax_size_x = np.diff(self.ax.get_xticks())[0]
        self.r = ax_size_x / 2.0
        for f in self.fibers:
            f.r = ax_size_x
            f.update_picker_point()

        self._update_picker_point()

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
            self.ax.lines = []
            self.ax.collections = []
            for f in self.fibers:
                f.update_plot()
            self.ax.figure.canvas.draw()
        elif event.key == 'w':
            for f in self.fibers:
                print(f)
        elif event.key == 'y':
            logging.debug('varying length parameter')
            f = self.fibers[0].fiber
            # plotting interval
            dmin = np.sqrt((f.endX - f.x0) ** 2 + (f.endY - f.y0) ** 2)
            for l in np.linspace(dmin, dmin * 2.0, num=7):
                f.L = l
                f.update_ode()
                f.plot(self.ax)
                self.ax.figure.canvas.draw()
        elif event.key == 'd':
            f = self.fibers[0].fiber
            logger.debug(f.get_line_integral_over_image())
            f.paint_line_integral_over_image(self.ax)
            self.ax.figure.canvas.draw()
        elif event.key == 'f':
            f = self.fibers[0].fiber
            # logger.debug(f.get_line_integral_over_image())
            result = f.fit()
            logger.debug(result.params)
            f.parameters = result.params
            f.update_ode()
            f.plot(self.ax)
            self.ax.figure.canvas.draw()

    def on_pick(self, event):
        if self.pick_ini_point == event.artist:
            self.picking = True
            mevent = event.mouseevent
            logging.debug('aster pick: xdata=%f, ydata=%f' % (mevent.xdata, mevent.ydata))
            self.xa = mevent.xdata
            self.ya = mevent.ydata

    def on_motion(self, event):
        if not self.picking: return
        # logging.debug('aster motion: xdata=%f, ydata=%f' % (event.xdata, event.ydata))
        # for fib in self.fibers:
        #     fib.x0 = event.xdata
        #     fib.y0 = event.ydata
        #     fib.update_picker_point()
        self.xa = event.xdata
        self.ya = event.ydata
        self._update_picker_point()

    def on_release(self, event):
        self._reframe()
        if not self.picking: return
        logging.debug('aster release: xdata=%f, ydata=%f' % (event.xdata, event.ydata))
        self.picking = False
        for i, fib in enumerate(self.fibers):
            fib.x0 = event.xdata
            fib.y0 = event.ydata
            logging.debug(
                'fiber {}: x0={: .2f} y0={: .2f} xe={: .2f} ye={: .2f}'.format(i, fib.x0, fib.y0, fib.Xe, fib.Ye))

        self.xa = event.xdata
        self.ya = event.ydata
        self._update_picker_point()


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')

    ax.set_facecolor('b')

    with tf.TiffFile(
            '/Users/Fabio/data/mt-bending/srrf/Stream to disk_XY0_Z0_T000_C0-2.tiff - SRRF 50 frames.tif') as tif:
        if tif.is_imagej is not None:
            sizeT, channels = tif.imagej_metadata['images'], tif.pages[0].imagedepth
            sizeZ, sizeX, sizeY = 1, tif.pages[0].imagewidth, tif.pages[0].imagelength
            print('N of frames=%d channels=%d, sizeZ=%d, sizeX=%d, sizeY=%d' % \
                  (sizeT, channels, sizeZ, sizeX, sizeY))
            res = 4.5

            sequence = tif.asarray().reshape([sizeT, channels, sizeX, sizeY])
            # print(sequence.shape)
            img = tif.pages[10].asarray()

            ax.imshow(img, extent=(0, sizeX / res, 0, sizeY / res))
            # ax.set_xlim([0, sizeX / res])
            # ax.set_ylim([0, sizeY / res])
    ax.set_xlim([40, 70])
    ax.set_ylim([40, 70])

    F = 30 / 2.0 * np.sqrt(2) * 1e-12
    fiber = e.PlanarImageMinimizer(L=1.2, E=1e9, J=1e-20, N1=2.5e-22, N2=2.5e-22,
                                   dx=10.0, dy=0.0, theta=2 * np.pi / 3, image=tif.pages[10])
    # fiber = e.PlanarElastica(L=1.2, E=1e9, J=1e-20, N1=2.5e-22, N2=2.5e-22,
    #                          dx=10.0, dy=0.0, theta=1.01)
    # fiber = e.PlanarElastica()
    centrosome = Aster(ax, x0=56.5, y0=60)
    # centrosome = Aster(ax)
    centrosome.add_fiber(fiber)

    plt.show()
