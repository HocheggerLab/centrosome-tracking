import os
import configparser
import ast

import matplotlib.pyplot as plt

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
