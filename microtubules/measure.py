import os
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from microtubules.elastica import Aster
import tools.data
from microtubules import elastica as e

np.set_printoptions(1)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)
log = logging.getLogger(__name__)


# log.setLevel(logging.DEBUG)


def on_key(event):
    log.debug('press %s' % event.key)
    if event.key == 'w':
        for f in c1.fibers:
            f.parameters.pretty_print()
    elif event.key == 'a':
        x0, y0 = c1.pt
        fiber = e.PlanarImageMinimizerIVP(ax, w=1.0, k0=-1.0, alpha=np.pi / 6,
                                          m0=0.0, x0=x0, y0=y0, L=10,
                                          phi=1 * np.pi / 3, image=tub)

        c1.add_fiber(fiber)
        c1.ax.figure.canvas.draw()
    elif event.key == 'd':
        f = c1.selected_fiber
        log.debug(f.get_line_integral_over_image())
        f.paint_line_integral_over_image(c1.ax)
        c1.ax.figure.canvas.draw()

    elif event.key == 'f':
        log.info('data fitting for single fiber')
        f: e.PlanarImageMinimizerIVP = c1.selected_fiber
        result = f.fit()
        f.parameters = result.params

        c1.clear()
        for f in c1.fibers:
            f.eval()
            f.plot(c1.ax)
        c1.ax.figure.canvas.draw()

    elif event.key == 'i':
        log.info('retrieving info for selected fiber')
        f: e.PlanarImageMinimizerIVP = c1.selected_fiber
        f.parameters.pretty_print()
        f.plot_fit()
        f.forces()

    elif event.key == 'a':
        log.info('data fitting process taking place. This can take long!')
        f: e.PlanarImageMinimizerIVP
        for f in c1.fibers:
            result = f.fit()
            f.parameters = result.params

        c1.clear()
        for f in c1.fibers:
            f.eval()
            f.plot(c1.ax)

        c1.ax.figure.canvas.draw()


if __name__ == '__main__':
    fig = plt.figure(dpi=100)
    ax = fig.gca()
    ax.set_aspect('equal')

    file = '/Users/Fabio/data/lab/SRRF images for Fabio/15-2-19/15-2-19 Best Images/U2OS CDK1as +1NM+STLC PFA in CBS SRRF composite - Capture 11.tif'
    imgs, frames, channels, dt, res, (sizeZ, sizeX, sizeY) = tools.data.image_reader(file)
    act, tub, pact, dapi = imgs
    # act = color.gray2rgb(act) * sp.colors.alexa_594
    # tub = color.gray2rgb(tub) * sp.colors.alexa_488
    # dapi = color.gray2rgb(dapi) * sp.colors.hoechst_33342
    # img = act * 0.1 + tub * 0.1 + dapi * 0.1
    res = 4.5 * 4 if res is None else res
    ax.imshow(tub, extent=(0, sizeX / res, 0, sizeY / res), cmap=cm.gray, interpolation='none')

    x0, y0 = sizeX / res / 2, sizeY / res / 2

    cfg_file = file[:-3] + 'cfg'
    if os.path.exists(cfg_file):
        log.info('loading aster from configuration file')
        c1 = Aster.from_file(cfg_file, ax, tub)
    else:
        c1 = Aster(ax, imgs[1], x0=x0, y0=y0)
        fiber = e.PlanarImageMinimizerIVP(ax, w=1.0, k0=-1.0, alpha=np.pi / 6,
                                          m0=0.0, x0=x0, y0=y0, L=10,
                                          phi=1 * np.pi / 3, image=tub)

        c1.add_fiber(fiber)

    ax.figure.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    c1.save(cfg_file)
