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


class Aster:
    def __init__(self, axes, x0=0.0, y0=0.0):
        self.picking = False
        self.ax = axes
        self.fibers = []

        self._whmax = 0.0  # width height
        self.xa = x0
        self.ya = y0

        ax_size_x = np.diff(self.ax.get_xticks())[0]
        self.r = ax_size_x

        self._init_graph()
        self._connect()

    def _init_graph(self):
        self.pick_ini_point = plt.Circle((self.xa, self.ya), radius=self.r / 5.0, fc='w', picker=5)
        self.ax.add_artist(self.pick_ini_point)

    def _hide_picker(self):
        self.pick_ini_point.set_visible(False)

    def _show_picker(self):
        self.pick_ini_point.set_visible(True)

    def add_fiber(self, fiber):
        fiber.x0 = self.xa
        fiber.y0 = self.ya
        self.fibers.append(fiber)
        self._reframe()

    def _update_picker_point(self):
        self.pick_ini_point.center = (self.xa, self.ya)
        self.pick_ini_point.radius = (self.r / 5.0)
        self.ax.figure.canvas.draw()

    def _reframe(self):
        # computing radius for picker elements
        print(self.ax.get_xticks())
        ax_size_x = np.diff(self.ax.get_xticks())[0]
        self.r = ax_size_x / 2.0
        for f in self.fibers:
            f.r = ax_size_x
            f.update_picker_point()

        return
        maxx, maxy = 0, 0
        for fib in self.fibers:
            maxx = np.abs(fib.endX) if np.abs(fib.endX) > maxx else maxx
            maxy = np.abs(fib.endY) if np.abs(fib.endY) > maxy else maxy
        maxx *= 1.5
        maxy *= 1.5

        self._whmax = np.max([maxx, maxy, self._whmax])
        xi, xe = -self._whmax + self.xa, self._whmax + self.xa
        yi, ye = -self._whmax + self.ya, self._whmax + self.ya

        self.ax.set_xlim([xi, xe])
        self.ax.set_ylim([yi, ye])
        self.ax.set_aspect('equal')

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
                f.plot(self.ax)
                f._show_picker()
            self._show_picker()
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
                f.eval()
                f.plot(self.ax)
                self.ax.figure.canvas.draw()
        elif event.key == 'd':
            f = self.fibers[0]
            logger.debug(f.get_line_integral_over_image())
            f.paint_line_integral_over_image(self.ax)
            self.ax.figure.canvas.draw()
        elif event.key == 'f':
            f = self.fibers[0]
            result = f.fit()
            logger.debug(result.params)

            f.parameters = result.params
            f.eval()
            f.plot(self.ax)

            f._hide_picker()
            self._hide_picker()

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
        self.xa = event.xdata
        self.ya = event.ydata
        self._update_picker_point()

    def on_release(self, event):
        if not self.picking: return
        logging.debug('aster release: xdata=%f, ydata=%f' % (event.xdata, event.ydata))
        self.picking = False
        self.xa = event.xdata
        self.ya = event.ydata
        for i, f in enumerate(self.fibers):
            f.r = self.r
            f.x0 = self.xa
            f.y0 = self.ya
            f.update_picker_point()

        self._update_picker_point()
        self._reframe()


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

    F = 30 * np.sqrt(2) * 1e-12
    x0, y0 = 56.5, 60
    fiber = e.PlanarImageMinimizerIVP(ax, L=10.2, E=1e9, J=1e-8, F=F, m0=0.01, x0=x0, y0=y0,
                                      theta=2 * np.pi / 3, image=tif.pages[10])
    # fiber = e.PlanarElasticaIVPArtist(ax, L=10.2, E=1e9, J=1e-8, F=5e-1, theta=np.pi / 3)
    centrosome = Aster(ax, x0=x0, y0=y0)
    centrosome.add_fiber(fiber)

    plt.show()
