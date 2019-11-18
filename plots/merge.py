import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg

logger = logging.getLogger(__name__)


def merge(images, ax, um_per_pix=1,
          cmaps=None, xlim_um=None, ylim_um=None,
          order=None, merge=None):
    logger.debug("Making montage of image.")
    w_um, h_um = [s * um_per_pix for s in images[0].shape]
    n_channels = len(images)
    order = range(n_channels) if order is None else order
    cmaps = ['gray'] * n_channels if cmaps is None else cmaps

    ax.cla()
    a = ax.imshow(images[0], resample=False)
    shape = a.make_image(matplotlib.backends.backend_agg, unsampled=True)[0].shape
    img = np.zeros(shape, dtype=np.int64)
    for i in merge:
        ax.cla()
        j = order.index(i)
        o = order[j]
        print("selecting merge index %d order index %d cmap %s" % (i, j, cmaps[j]))
        a = ax.imshow(images[o], cmap=cmaps[o], resample=False)
        img += a.make_image(matplotlib.backends.backend_agg, unsampled=True)[0]

    ax.cla()
    ax.imshow(img, extent=[0, w_um, h_um, 0], resample=False)
    ax.invert_yaxis()

    if xlim_um is not None and ylim_um is not None:
        ax.set(xlim=xlim_um, ylim=ylim_um)

    ax.set_axis_off()
    plt.subplots_adjust(hspace=0, wspace=0.01, left=0, right=1, top=1, bottom=0)
