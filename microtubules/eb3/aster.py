import os
import configparser
import ast

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import plot_special_tools as sp
from tools.draggable import DraggableCircle


def read_aster_config(fname):
    _sters = list()
    if os.path.isfile(fname):
        with open(fname, 'r') as configfile:
            config = configparser.ConfigParser()
            config.read_file(configfile)

            section = 'Asters'
            if config.has_section(section):
                number = config.getint(section, 'number')
                unit = config.get(section, 'unit')

                if unit != 'micrometer': raise Exception(
                    'units are currently accepted in micrometers only for configuration file.')

            for i in range(number):
                section = 'Aster %d' % i
                if config.has_section(section):
                    c_pt = ast.literal_eval(config.get(section, 'center'))
                    circle = plt.Circle(xy=c_pt, radius=2, fc='g', picker=5)
                    _sters.append(DraggableCircle(circle))

    return [a.circle.center for a in _sters]


def write_aster_config(fname, asters):
    with open(fname, 'w') as configfile:
        config = configparser.RawConfigParser()
        config.add_section('General')
        config.set('General', 'Version', 'v0.1')

        section = 'Asters'
        config.add_section(section)
        config.set(section, 'number', len(asters))
        config.set(section, 'unit', 'micrometer')

        for i, a in enumerate(asters):
            # if type(a) != DraggableCircle: continue

            section = 'Aster %d' % i
            config.add_section(section)
            config.set(section, 'center', a)

        config.write(configfile)


def select_asters(image_path):
    def on_key(event):
        print('press', event.key)
        if event.key == 'c':
            ci = DraggableCircle(plt.Circle(xy=orig, radius=2, fc='g', picker=5))
            asters.append(ci)
            ax.add_artist(ci.circle)
            ci.connect()
            fig.canvas.draw()

    images, pix_per_um, dt, n_frames = sp.load_tiff(image_path)
    w, h = images[0].shape[0], images[0].shape[1]

    fig = plt.figure()
    ax = fig.gca()
    ext = [0, w / pix_per_um, h / pix_per_um, 0]
    ax.imshow(np.max(images, axis=0), interpolation='none', extent=ext, cmap=cm.gray)
    orig = (w / 2 / pix_per_um, h / 2 / pix_per_um)

    ci = DraggableCircle(plt.Circle(xy=orig, radius=2, fc='g', picker=5))
    asters = [ci]
    ax.add_artist(ci.circle)
    ci.connect()

    cidkeyboard = fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.draw()
    plt.show()

    return [a.circle.center for a in asters]
