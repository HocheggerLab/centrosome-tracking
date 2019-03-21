import ast
import configparser
import os
import logging

from planar import Vec2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from lmfit import Parameters

from tools.draggable import DraggableCircle
from microtubules import elastica as e

log = logging.getLogger(__name__)


class Aster:
    """
        Implements an aster where microtubules are placed.
        All the length variables are in micrometers
    """
    centrosome_radius = 0.2

    def __init__(self, axes: plt.Axes, image: tifffile.TiffPage, x0=0.0, y0=0.0, filename=None):
        self.ax = axes
        self._img = image
        self.fibers = []
        self.selected_fiber = None

        self.pt = Vec2(x0, y0)

        crcl = plt.Circle((x0, y0), radius=self.centrosome_radius, fc='y', picker=self.centrosome_radius, zorder=100)
        axes.add_artist(crcl)
        self._o = DraggableCircle(crcl, on_pick_callback=self._cleanup,
                                  on_release_callback=self._update_fibers)  # o for origin
        self._o.connect()
        self._connect()

    def add_fiber(self, fiber):
        assert type(fiber) == e.PlanarImageMinimizerIVP, 'argument is not a fiber that we like'

        fiber.callback = self._update_fibers
        fiber.x0 = self.pt.x
        fiber.y0 = self.pt.y
        fiber.unselect()
        self.fibers.append(fiber)

    def _cleanup(self):
        log.debug('cleanup')
        for f in self.fibers:
            f.unselect()

    def _update_fibers(self):
        log.debug('update fibers')
        x0, y0 = self._o.circle.center
        self.pt = Vec2(x0, y0)

        self.clear()
        f: e.PlanarImageMinimizerIVP
        for f in self.fibers:
            f.x0 = x0
            f.y0 = y0
            f.update_picker_point()
            f.eval()
            f.plot(self.ax)
        return

    def clear(self):
        self.ax.lines = []
        self.ax.collections = []

    def save(self, path):
        with open(path, 'w') as configfile:
            config = configparser.RawConfigParser()
            config.add_section('General')
            config.set('General', 'Version', 'v0.1')

            section = 'Aster'
            config.add_section(section)
            config.set(section, 'fiber number', len(self.fibers))
            config.set(section, 'unit', 'micrometer')
            config.set(section, 'center', (self.pt.x, self.pt.y))

            f: e.PlanarImageMinimizerIVP
            for i, f in enumerate(self.fibers):
                section = 'Fiber %d' % i
                config.add_section(section)
                config.set(section, 'center', (f.x0, f.y0))
                config.set(section, 'fitting_pt', f.fit_pt)
                config.set(section, 'parameters', f.parameters.dumps())

            log.info('saving aster into %s' % path)
            config.write(configfile)

    @staticmethod
    def from_file(filename, ax, img):
        if os.path.isfile(filename):
            with open(filename, 'r') as configfile:
                log.info('loading aster from %s' % filename)
                config = configparser.ConfigParser()
                config.read_file(configfile)

                section = 'Aster'
                if config.has_section(section):
                    fnumber = config.getint(section, 'fiber number')
                    unit = config.get(section, 'unit')
                    assert unit == 'micrometer', 'units are currently accepted in micrometers only'
                    x0, y0 = ast.literal_eval(config.get(section, 'center'))

                    aster = Aster(ax, img, x0=x0, y0=y0)
                    for i in range(fnumber):
                        section = 'Fiber %d' % i
                        if config.has_section(section):
                            x0, y0 = ast.literal_eval(config.get(section, 'center'))
                            params = Parameters().loads(config.get(section, 'parameters'))
                            fiber = e.PlanarImageMinimizerIVP(ax, x0=x0, y0=y0, image=img)
                            fiber.fit_pt = ast.literal_eval(config.get(section, 'fitting_pt'))
                            fiber.parameters = params
                            fiber.unselect()
                            aster.add_fiber(fiber)
                    aster._update_fibers()
                    return aster

    def on_pick(self, event):
        f: e.PlanarImageMinimizerIVP
        if np.any([f.picked for f in self.fibers]): return
        if type(event.artist) == plt.Line2D:
            if self.selected_fiber is not None and self.selected_fiber.curve == event.artist: return
            for k, f in enumerate(self.fibers):
                if f.curve == event.artist:
                    self.selected_fiber = f
                    f.select()
                    print('fiber %d, index %d' % (k, self.fibers.index(f)))
                else:
                    f.unselect()
            self.ax.figure.canvas.draw()

    def _connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)

    def _disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
