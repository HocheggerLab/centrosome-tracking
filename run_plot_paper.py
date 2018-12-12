import logging

import coloredlogs
import matplotlib.gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

import mechanics as m
import data as hdata
import harryplots as hp
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


def fig_1(data):
    trk_plots = pl.Tracks(data)
    msd_plots = pl.MSD(data)
    trk_harry = hp.Tracks()
    msd_harry = hp.MSD()

    logger.info('doing figure 1')
    with PdfPages(parameters.data_dir + 'out/figure1.pdf') as pdf:
        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sns.xkcd_rgb["silver"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        trk_harry.for_condition(ax, '+1nmpp1+stlc-washout-STLC')
        trk_harry.format_axes(ax, time_minor=15, time_major=30)

        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        sns.set_palette(['#000000', '#000000'])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        msd_harry.displacement_more_less(ax, '+1nmpp1+stlc-washout-STLC')
        msd_harry.format_axes(ax, time_minor=5, time_major=20, msd_minor=25, msd_major=50)

        ax.set_ylim([0, 200])
        ax.set_xlim([0, 90])
        sp.set_axis_size(3, 3, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        msd_harry.mother_daughter(ax, '+1nmpp1+stlc-washout-STLC', 'sep')
        msd_harry.format_axes(ax, time_minor=5, time_major=20, msd_minor=25, msd_major=50)

        ax.set_ylim([0, 200])
        ax.set_xlim([0, 90])
        sp.set_axis_size(3, 3, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        trk_plots.pSTLC(ax, centered=True)
        trk_plots.format_axes(ax)

        ax.set_xlim([-240, 30])
        ax.set_ylim([0, 30])
        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        sns.set_palette(['#000000', '#000000'])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        msd_plots.pSTLC(ax)
        msd_plots.format_axes(ax, time_minor=5, time_major=20, msd_minor=25, msd_major=50)

        # ax.set_xlim([-240, 30])
        ax.set_ylim([0, 300])
        ax.set_xlim([-90, 0])
        sp.set_axis_size(3, 3, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        msd_plots.pSTLC_mother_dauther(ax)
        msd_plots.format_axes(ax, time_minor=5, time_major=20, msd_minor=25, msd_major=50)

        ax.set_ylim([0, 200])
        ax.set_xlim([0, 90])
        sp.set_axis_size(3, 3, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sns.xkcd_rgb["silver"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        trk_harry.for_condition(ax, 'unperterbed')
        trk_harry.format_axes(ax, time_minor=15, time_major=30)

        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        sns.set_palette(['#000000', '#000000'])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        msd_harry.displacement_more_less(ax, 'unperterbed')
        msd_harry.format_axes(ax, time_minor=5, time_major=20, msd_minor=50, msd_major=150)

        ax.set_ylim([0, 600])
        ax.set_xlim([0, 90])
        sp.set_axis_size(3, 3, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()

        msd_harry.mother_daughter(ax, 'unperterbed', 'sep')
        msd_harry.format_axes(ax, time_minor=5, time_major=20, msd_minor=25, msd_major=50)

        ax.set_ylim([0, 200])
        ax.set_xlim([0, 90])
        sp.set_axis_size(3, 3, ax=ax)
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

        dfs, cond = data.get_condition(['1_P.C.'])
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

        pdf.savefig(transparent=True, bbox_inches='tight')


def fig_3(data):
    with PdfPages(parameters.data_dir + 'out/figure3.pdf') as pdf:
        trk_plots = pl.Tracks(data)
        sns.set_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED])
        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi, clear=True)
        ax = fig.gca()
        trk_plots.nocodazole(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi, clear=True)
        ax = fig.gca()
        trk_plots.blebbistatin(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi, clear=True)
        ax = fig.gca()
        trk_plots.cytochalsinD(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi, clear=True)
        ax = fig.gca()
        trk_plots.chTog(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi, clear=True)
        ax = fig.gca()
        trk_plots.faki(ax, centered=True)
        ax.set_xlim([-100, 0])
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
    # fig_3(data)
