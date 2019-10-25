import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg

logger = logging.getLogger(__name__)


def montage(images, ch_names=None, um_per_pix=1,
            cmaps=None, xlim_um=None, ylim_um=None,
            order=None, merge=None):
    def plotimg(img, ch, **kwargs):
        ax = plt.gca()
        att = [1.3, 0.3, 0.6]
        ch = ch.iloc[0]
        if ch == "merge":
            ax.cla()
            a = ax.imshow(img.iloc[0], resample=False)
            shape = a.make_image(matplotlib.backends.backend_agg, unsampled=True)[0].shape
            img = np.zeros(shape, dtype=np.int64)
            for i in merge:
                ax.cla()
                a = ax.imshow(images[i], cmap=cmaps[order[i]], vmax=images[i].max() * att[i], resample=False)
                img += a.make_image(matplotlib.backends.backend_agg, unsampled=True)[0]

            ax.cla()
            ax.imshow(img, extent=[0, w_um, h_um, 0], resample=False)
            ax.invert_yaxis()
        else:
            img = img.iloc[0]
            ax.imshow(img, extent=[0, w_um, h_um, 0], cmap=plt.cm.binary, resample=False, vmax=img.max() * att[ch])

        if xlim_um is not None and ylim_um is not None:
            x0, y0 = xlim_um[0], ylim_um[0]
            ax.plot([x0 + 2, x0 + 12], [y0 + 2, y0 + 2], c='w', lw=1)
            ax.text(x0 + 5, y0 + 3.5, '10 um', color='w', fontdict={'size': 7})
        ax.set_axis_off()

    logger.debug("Making montage of image.")
    w_um, h_um = [s * um_per_pix for s in images[0].shape]
    n_channels = len(images)
    order = range(n_channels) if order is None else order
    ch_names = range(n_channels) if ch_names is None else ch_names
    cmaps = ['gray'] * n_channels if cmaps is None else cmaps

    im_df = pd.DataFrame()
    for i, (o, title) in enumerate(zip(order, ch_names)):
        d = pd.DataFrame(data={
            'channel': [i],
            'name': [title],
            'image': [images[o]],
        })
        im_df = im_df.append(d, ignore_index=True, sort=False)
    im_df = im_df.append(pd.DataFrame(data={
        'channel': ["merge"],
        'name': ["Merge"],
        'image': [images[0]],
    }), ignore_index=True, sort=False)

    _s = 1.0
    g = sns.FacetGrid(im_df, col="name", height=_s)
    g = (g.map(plotimg, "image", "channel")
         .set_titles("{col_name}")
         .add_legend()
         )
    if xlim_um is not None and ylim_um is not None:
        g.set(xlim=xlim_um, ylim=ylim_um)

    # g.fig.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.01, left=0, right=1, top=1, bottom=0)

    return g
