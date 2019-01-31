import logging

import seaborn as sns
import pandas as pd
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from imagej_pandas import ImagejPandas
import plot_special_tools as sp

log = logging.getLogger(__name__)

pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]
_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.5, 'lw': 0.1}
_dpi = 300


class Tracks():
    def __init__(self, data):
        self.data = data

    def pSTLC(self, ax, centered=False):
        df = self.data.get_condition(['1_P.C.'], centered=centered)
        # sns.lineplot(data=df,
        #              x='Time', y='DistCentr', hue='condition',
        #              estimator=np.nanmean, lw=3, ax=ax)
        sns.lineplot(data=df,
                     x='Time', y='DistCentr', hue='condition',
                     units='indiv', estimator=None,
                     lw=0.2, alpha=1, legend=False, ax=ax)
        ax.axvline(linewidth=1, linestyle='--', color='k', zorder=50)
        ax.set_xlabel('time [min]')
        ax.set_ylabel('distance [um]')
        ax.legend(title=None, loc='upper left')


class MSD():
    msd_ylim = [0, 420]

    def __init__(self, data):
        self.data = data

    def pSTLC(self, ax, centered=False):
        df, cond = self.data.get_condition(['1_P.C.'], centered=centered)

    def msd_vs_congression(self, ax):
        _conds = ['1_N.C.', '1_P.C.',
                  '2_CDK1_DA', '2_CDK1_DC', '1_Bleb', 'hset', 'kif25', 'hset+kif25',
                  '2_Kines1', '2_CDK1_DK', '1_DIC', '1_Dynei', '1_CENPF', '1_BICD2',
                  '1_No10+', '1_MCAK', '1_chTOG',
                  '1_CyDT', '1_FAKI', '1_ASUND']

        markers = ['o', 'o',
                   '^', '<', '>', 's', 'X', 'v',
                   's', 'X', 'v', '^', '<', '>',
                   'p', 'P', 'X',
                   'p', 'P', 'X']

        df, conds = self.data.get_condition(_conds)
        colors = [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]
        colors.extend([sp.SUSSEX_GRAPE] * 6)
        colors.extend([sp.SUSSEX_FLINT] * 6)
        colors.extend([sp.SUSSEX_FUSCHIA_PINK] * 3)
        colors.extend([sp.SUSSEX_TURQUOISE] * 3)
        # colortuple = dict(zip(conds, colors))

        df_ = df[df['Time'] <= 100]
        df_msd = ImagejPandas.msd_particles(df_).set_index('Frame').sort_index()
        dfcg = sp._compute_congression(df).set_index('Time').sort_index()
        df_msd_final = pd.DataFrame()
        for _, dfmsd in df_msd.groupby(ImagejPandas.CENTROSOME_INDIV_INDEX):
            cnd = dfmsd.iloc[0]['condition']
            cgr = dfcg[dfcg['condition'] == cnd].iloc[-1]['congress']
            df_it = pd.DataFrame([[dfmsd.iloc[-1]['msd'], cgr, cnd]],
                                 columns=['msd', 'cgr', 'condition'])
            df_msd_final = df_msd_final.append(df_it)

        # inneficient as cgr is repeated for every sample
        df_msd_final = df_msd_final.groupby('condition').mean().reset_index()
        for cnd, m, _color in zip(conds, markers, colors):
            p = df_msd_final[df_msd_final['condition'] == cnd]
            ax.scatter(p['cgr'], p['msd'], c=_color, s=200, label=cnd, marker=m, zorder=1000)
