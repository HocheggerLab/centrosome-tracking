import os
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm

from microtubules.elastica import Aster
import tools.data
from microtubules import elastica as e

np.set_printoptions(1)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 50)

# Type 2/TrueType fonts.
matplotlib.rcParams.update({'pdf.fonttype': 42})
matplotlib.rcParams.update({'ps.fonttype': 42})
matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif': ['Arial']})


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

    elif event.key == '+':
        log.info('adding another fitting point to selected fibre')
        f: e.PlanarImageMinimizerIVP = c1.selected_fiber
        f.fit_pt = f.fit_pt + [(c1.pt.x + 2, c1.pt.y)]

    elif event.key == 'l':
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

    elif event.key == 'r':
        log.info('report of fiber parameters')
        f: e.PlanarImageMinimizerIVP
        pr = pd.DataFrame([f.parameters.valuesdict() for f in c1.fibers])
        print(pr)


if __name__ == '__main__':
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    ax.set_aspect('equal')

    file = '/Users/Fabio/data/lab/SRRF images for Fabio/15-2-19/15-2-19 Best Images/U2OS CDK1as +1NM+STLC PFA in CBS SRRF composite - Capture 11.tif'
    imgs, frames, channels, dt, res, (sizeZ, sizeX, sizeY) = tools.data.image_reader(file)
    act, tub, pact, dapi = imgs
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
    # fig.savefig('big_picture.pdf')

    # fig = plt.figure(dpi=200)
    # ax = fig.gca()
    # ax.set_aspect('equal')
    # act = color.gray2rgb(img_as_ubyte(act)) * sp.colors.red
    # tub = color.gray2rgb(img_as_ubyte(tub)) * sp.colors.alexa_488
    # dapi = color.gray2rgb(img_as_ubyte(dapi)) * sp.colors.hoechst_33342
    # img = act * 0.004 + tub * 0.1 + dapi * 0.05
    # ax.imshow(img, extent=(0, sizeX / res, 0, sizeY / res), interpolation='none')
    ax.imshow(tub, extent=(0, sizeX / res, 0, sizeY / res), cmap=cm.gray, interpolation='none')
    c1.render(ax=ax)
    ax.set_xlim([20, 45])
    ax.set_ylim([22, 47])
    fig.savefig('net_forces.pdf')

    plt.show()
