import logging

import seaborn as sns
import numpy as np

import plot_special_tools as sp

log = logging.getLogger(__name__)

pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]


class Rotation:
    def __init__(self, data):
        self.data = data

    def angle_over_time(self, ax):
        df = self.data
        color = sns.color_palette()[1]

        sns.lineplot(data=df,
                     x='frame', y='th0',
                     estimator=np.nanmean, err_style=None,
                     lw=2, alpha=1, c=color, ax=ax)
        sns.lineplot(data=df,
                     x='frame', y='th0',
                     units='particle', estimator=None,
                     lw=0.2, alpha=1, c=color, ax=ax)
        sns.scatterplot(data=df,
                        x='frame', y='th0',
                        units='particle', estimator=None,
                        alpha=0.5, c=color, ax=ax)
        ax.text(0, 0, 'Ns=%d' % df['particle'].nunique(), backgroundcolor='yellow')
