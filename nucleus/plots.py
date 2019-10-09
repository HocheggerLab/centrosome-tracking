import logging

import seaborn as sns
import numpy as np

import tools.plot_tools as sp

logger = logging.getLogger(__name__)

pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]


class Rotation:
    def __init__(self, data):
        self.data = data
        self.data.loc[:, 'indv'] = self.data['nucleus'].map(str) + '|' + self.data['particle'].map(str)
        l = len(data['nucleus'].unique())
        logger.info("%d %s to plot" % (l, "nucleus" if l == 1 else "nuclei"))
        logger.info(sorted(data['nucleus'].unique()))

    @staticmethod
    def _lineplot(df, y_col, ax):
        color = sns.color_palette()[1]
        nuc_order = sorted(df['nucleus'].unique())
        sns.lineplot(data=df, x='frame', y=y_col,
                     # hue='nucleus', hue_order=nuc_order,
                     estimator=np.nanmean, err_style=None,
                     lw=2, alpha=1, color=color, ax=ax)
        sns.lineplot(data=df, x='frame', y=y_col,
                     # hue='nucleus', hue_order=nuc_order,
                     units='indv', estimator=None,
                     lw=0.2, alpha=1, color=color, ax=ax)
        sns.scatterplot(data=df, x='frame', y=y_col,
                        # hue='nucleus', hue_order=nuc_order,
                        units='indv', estimator=None,
                        alpha=0.5, color=color, ax=ax)
        ax.text(0.5, 0, 'Ns=%d' % df['particle'].nunique(), backgroundcolor='yellow')

    def angle_over_time(self, ax):
        self._lineplot(self.data, 'th', ax)

    def angle_corrected_over_time(self, ax):
        self._lineplot(self.data, 'th+', ax)

    def angle_displacement_over_time(self, ax):
        self._lineplot(self.data, 'th_dev', ax)

    def angular_speed_over_time(self, ax):
        self._lineplot(self.data, 'omega', ax)

    def radius_over_time(self, ax):
        self._lineplot(self.data, 'r', ax)

    def radius_change_over_time(self, ax):
        self.data.loc[:, "dr/dt"] = self.data["r"].diff()
        self._lineplot(self.data, 'dr/dt', ax)

    def msd_over_time(self, ax):
        # sp.set_axis_size(4, 2, ax=ax)
        sns.lineplot(data=self.data, x='frame', y='msd', estimator=np.nanmean, ax=ax)
        sns.lineplot(data=self.data, x='frame', y='msd',
                     estimator=None, units='indv',
                     lw=0.2, alpha=1, ax=ax)
        ax.text(0.5, 0, 'Ns=%d' % self.data['particle'].nunique(), backgroundcolor='yellow')
        ax.set_title('MSD of particles in nucleus')

    def msd_change_over_time(self, ax):
        # sp.set_axis_size(4, 2, ax=ax)
        self.data.loc[:, "dmsd/dt"] = self.data["msd"].diff()
        sns.lineplot(data=self.data, x='frame', y='dmsd/dt', estimator=np.nanmean, ax=ax)
        sns.lineplot(data=self.data, x='frame', y='dmsd/dt',
                     estimator=None, units='indv',
                     lw=0.2, alpha=1, ax=ax)
        ax.text(0.5, 0, 'Ns=%d' % self.data['particle'].nunique(), backgroundcolor='yellow')
        ax.set_title('MSD of particles in nucleus')
