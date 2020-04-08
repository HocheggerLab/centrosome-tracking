import logging
import matplotlib.cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

logger = logging.getLogger(__name__)

#  register some color palletes for use with microscopy images
green = {'red': [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0]],
         'green': [[0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0]],
         'blue': [[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]}

matplotlib.cm.register_cmap(name='uscope_green',
                            cmap=LinearSegmentedColormap('uscope_green', segmentdata=green, N=256))

red = {'red': [[0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0]],
       'green': [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0]],
       'blue': [[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]]}

matplotlib.cm.register_cmap(name='uscope_red',
                            cmap=LinearSegmentedColormap('uscope_red', segmentdata=red, N=256))

blue = {'red': [[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]],
        'green': [[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]],
        'blue': [[0.0, 0.0, 0.0],
                 [1.0, 1.0, 1.0]]}

matplotlib.cm.register_cmap(name='uscope_blue',
                            cmap=LinearSegmentedColormap('uscope_blue', segmentdata=blue, N=256))

magenta = {'red': [[0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0]],
           'green': [[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]],
           'blue': [[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0]]}

matplotlib.cm.register_cmap(name='uscope_magenta',
                            cmap=LinearSegmentedColormap('uscope_magenta', segmentdata=magenta, N=256))
