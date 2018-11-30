import logging

import coloredlogs
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy.stats

import stats
import mechanics as m
import data as hdata
import parameters
import plot_special_tools as sp
import tools.plots as pl
import tools.data as data

logger = logging.getLogger(__name__)
logger.info(font_manager.OSXInstalledFonts())
logger.info(font_manager.OSXFontDirectories)

plt.style.use('bmh')
# print(matplotlib.rcParams.keys())
# Type 2/TrueType fonts.
matplotlib.rcParams.update({'pdf.fonttype': 42})
matplotlib.rcParams.update({'ps.fonttype': 42})

matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif': ['Arial']})

matplotlib.rcParams.update({'axes.titlesize': 8})
matplotlib.rcParams.update({'axes.labelsize': 7})
matplotlib.rcParams.update({'xtick.labelsize': 7})
matplotlib.rcParams.update({'ytick.labelsize': 7})
matplotlib.rcParams.update({'legend.fontsize': 7})

pd.set_option('display.width', 320)
coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)

_dpi = parameters.dpi


def fig_1(msd):
    trk_plots = pl.Tracks(msd)

    logger.info('doing figure 1')
    with PdfPages(parameters.data_dir + 'out/figure1.pdf') as pdf:
        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        trk_plots.pSTLC(ax, centered=True)
        ax.set_xlim([-150, 150])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sns.xkcd_rgb["grey"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(3.02, 1.5, ax=ax)

        dfh = hdata.harry_tidy_format()
        dfh = dfh[dfh['condition'].isin(['unperterbed', '+1nmpp1+stlc-washout-STLC'])]
        dfh = dfh[dfh['state'].isin(['sep'])]
        dd = dfh.groupby(['condition', 'indiv']).apply(m.avg_speed).rename('avg_speed').reset_index()
        dd = dd.dropna()

        dfs = msd.get_condition(['1_P.C.'])
        dfs.drop(
            columns=['NuclX', 'NuclY', 'CNx', 'CNy', 'Dist', 'Speed', 'Acc',
                     'Centrosome', 'DistCentr', 'SpeedCentr', 'AccCentr',
                     'CellX', 'CellY', 'DistCell', 'SpdCell', 'AccCell', 'NuclBound', 'CellBound'], inplace=True)
        dfs = dfs.rename(columns={'CentX': 'x', 'CentY': 'y'})
        print(dfs.columns)
        kwargs = {'frame': 'Frame', 'time': 'Time', 'centrosome_label_col': 'CentrLabel'}
        dfs.loc[:, 'indiv'] = dfs['run'] + '|' + dfs['Nuclei'].map(str)
        mua = dfs.groupby(['condition', 'indiv']).apply(m.avg_speed, **kwargs).rename('avg_speed').reset_index()
        print(mua)
        print(mua.columns)
        dd = dd.append(mua)
        order = ['unperterbed', '+1nmpp1+stlc-washout-STLC', '+STLC']

        sns.boxplot('condition', 'avg_speed', data=dd, ax=ax, linewidth=0.5, width=0.4, showfliers=False, dodge=False,
                    order=order)

        for i, artist in enumerate(ax.artists):
            artist.set_facecolor('None')
        sns.swarmplot('condition', 'avg_speed', data=dd, ax=ax, size=2, zorder=0, order=order)

        maxy = ax.get_ylim()[1]
        ypos = np.flip(ax.yaxis.get_major_locator().tick_values(maxy, maxy * 0.8))
        dy = np.diff(ypos)[0] * -1
        k = 0
        for i, c1 in enumerate(order):
            d1 = dd[dd['condition'] == c1]['avg_speed']
            for j, c2 in enumerate(order):
                if i < j:
                    d2 = dd[dd['condition'] == c2]['avg_speed']
                    st, p = scipy.stats.ttest_ind(d1, d2)
                    ypos = maxy - dy * k
                    print(i, j, c1, c2, ypos, p)
                    ax.plot([i, j], [ypos, ypos], lw=0.75, color='k')
                    ax.text(j, ypos, stats.star_system(p), ha='right', va='bottom')
                    k += 1

        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight')

        # msd_plots = pl.MSD(data)
        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        msd = msd.get_condition(['1_P.C.'])
        msd = m.get_msd(msd, group='trk',
                        frame='Frame', time='Time', x='CentX', y='CentY')
        msd = msd.loc[msd['msd'].notna(), :]
        msd = m._msd_tag(msd, time='Time', centrosome_label='CentrLabel')

        sns.lineplot(x='Time', y='msd', data=msd,
                     style='msd_cat', style_order=['displacing more', 'displacing less'],
                     hue='msd_cat', hue_order=['displacing more', 'displacing less'],
                     estimator=np.nanmean)
        ax.axvline(x=0, linewidth=1, linestyle='--', color='k', zorder=50)
        ax.legend(loc='upper left')
        ax.get_legend().remove()

        ax.set_ylim([0, 600])
        ax.set_xlim([-60, 120])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

        ax.set_title('+STLC')
        ax.set_ylabel('MSD')
        ax.set_xlabel('time [min]')
        pdf.savefig(transparent=True, bbox_inches='tight')


def fig_3(data):
    with PdfPages(parameters.data_dir + 'out/figure3.pdf') as pdf:
        trk_plots = pl.Tracks(data)
        sns.set_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED])
        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        trk_plots.nocodazole(ax, centered=True)
        ax.set_xlim([-150, 60])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        trk_plots.blebbistatin(ax, centered=True)
        ax.set_xlim([-150, 60])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        trk_plots.cytochalsinD(ax, centered=True)
        ax.set_xlim([-150, 60])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        trk_plots.chTog(ax, centered=True)
        ax.set_xlim([-150, 60])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(4, 1, ax=ax)
        trk_plots.faki(ax, centered=True)
        ax.set_xlim([-150, 60])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        msd_plots = pl.MSD(data)
        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
        sp.set_axis_size(6, 6, ax=ax)

        msd_plots.msd_vs_congression(ax)

        ax.legend(loc='upper right')
        ax.set_ylabel('MSD')
        ax.set_xlabel('Congression [%]')

        pdf.savefig(transparent=True)


if __name__ == '__main__':
    logger.info('loading data')
    data = data.Data()

    fig_1(data)
    fig_3(data)
