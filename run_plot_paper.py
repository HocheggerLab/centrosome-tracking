import itertools
import logging

import coloredlogs
import matplotlib as mpl
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy.stats

import stats
import parameters
import plot_special_tools as sp
from imagej_pandas import ImagejPandas
import run_plots_eb3 as eb3
import tools.plots as pl
import tools.data as data
import mechanics as m
import data as hdata

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

_dpi = 300


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
        ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(30))
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
        # print(dd)
        # print(dd.columns)

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
                    # FIXME: This test assumes that the populations have identical variances by default.
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
        # ax.set_xlim([-150, 150])
        ax.set_xlim([-60, 120])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

        ax.set_title('+STLC')
        ax.set_ylabel('MSD')
        ax.set_xlabel('time [min]')
        pdf.savefig(transparent=True, bbox_inches='tight')

        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(dpi=_dpi)
        # fig.clf()
        # fig.set_size_inches(2.2, 2.2)
        # ax = fig.gca(projection='3d')
        # sp.ribbon(dfs[dfs['condition'] == '-STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax)
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(dpi=_dpi)
        # fig.clf()
        # fig.set_size_inches(2.2, 2.2)
        # ax = fig.gca(projection='3d')
        # sp.ribbon(dfs[dfs['condition'] == '+STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax)
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # MSD
        # dfs = dfs[dfs['Time'] <= 50]
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        # # ---------------------------
        # #          FIRST PAGE
        # # ---------------------------
        # fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        # fig.clf()
        # ax1 = fig.gca()
        # sp.msd(dfs[dfs['condition'] == names['1_N.C.']], ax1, ylim=[0, 120])
        # ax1.set_ylabel('Mean Square Displacement')
        # ax1.set_xlabel('time delay [min]')
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        # fig.clf()
        # ax2 = fig.gca(sharey=ax1)
        # sp.msd(dfs[dfs['condition'] == names['1_P.C.']], ax2, ylim=[0, 120])
        # ax2.set_ylabel('Mean Square Displacement')
        # ax2.set_xlabel('time delay [min]')
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        # fig.clf()
        # ax3 = fig.gca()
        # sp.msd_indivs(dfs[dfs['condition'] == names['1_N.C.']], ax3, ylim=msd_ylim)
        # ax3.set_ylabel('Mean Square Displacement')
        # ax3.set_xlabel('time delay [min]')
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        # fig.clf()
        # ax4 = fig.gca(sharey=ax3)
        # sp.msd_indivs(dfs[dfs['condition'] == names['1_P.C.']], ax4, ylim=msd_ylim)
        # ax4.set_ylabel('Mean Square Displacement')
        # ax4.set_xlabel('time delay [min]')
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # DISTANCES boxplot
        # df_valid = df.loc[~df['CellBound'].isnull(), :]
        # log.debug(df_valid.set_index(ImagejPandas.NUCLEI_INDIV_INDEX).sort_index().index.unique())
        # log.debug(len(df_valid.set_index(ImagejPandas.NUCLEI_INDIV_INDEX).sort_index().index.unique()))
        #
        # stats = md.gen_dist_data(df[(df['condition'] == 'pc')])
        # order = ['C1 (Away)', 'C2 (Close)', 'Nucleus\nCentroid', 'Cell\nCentroid', 'Cell\n(manual)']
        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        # fig.clf()
        # ax = fig.gca()
        # sns.boxplot(data=stats, y='Dist', x='Type', order=order, width=0.5, linewidth=0.5, fliersize=0, ax=ax)
        # for i, artist in enumerate(ax.artists):
        #     artist.set_facecolor('None')
        #     artist.set_edgecolor('k')
        #     artist.set_zorder(5000)
        # for i, artist in enumerate(ax.lines):
        #     artist.set_color('k')
        #     artist.set_zorder(5000)
        # sns.swarmplot(data=stats, y='Dist', x='Type', order=order, size=3, zorder=100, color=sp.SUSSEX_CORAL_RED,
        #               ax=ax)
        # ax.set_xlabel('')
        # ax.set_ylabel('Distance [um]')
        # # ax2.yaxis.set_major_locator(MultipleLocator(5))
        # pmat = st.p_values(stats, 'Dist', 'Type', filename=parameters.data_dir + 'out/fig1-pv-dist.xls')
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        #
        # # ---------------------------
        # #          NEXT PAGE
        # # ---------------------------
        # fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        # fig.clf()
        # ax = fig.gca()
        # sp.anotated_boxplot(stats, 'Dist', cat='Type', ax=ax)
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_1_selected_track(df, mask):
    df_selected = df[(df['condition'] == 'pc') & (df['run'] == 'run_114') & (df['Nuclei'] == 2)]
    msk_selected = mask[(mask['condition'] == 'pc') & (mask['run'] == 'run_114') & (mask['Nuclei'] == 2)]

    with PdfPages(parameters.data_dir + 'out/fig1-selected.pdf') as pdf:
        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.6, 2.6), dpi=_dpi)
        fig.clf()
        gs = matplotlib.gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[0:2, 0])
        ax2 = plt.subplot(gs[2:4, 0], sharex=ax1)

        with sns.color_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED]):
            time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(df_selected, ImagejPandas.DIST_THRESHOLD)
            sp.distance_to_nuclei_center(df_selected, ax1, mask=msk_selected, time_contact=time_of_c)
            # sp.distance_between_centrosomes(between_df, ax2, mask=mask_c, time_contact=time_of_c)
            sp.distance_to_cell_center(df_selected, ax2, mask=msk_selected, time_contact=time_of_c)

        # change y axis title properties for small plots
        for _ax in [ax1, ax2]:
            # _ax.set_xlabel(_ax.get_xlabel(), rotation='horizontal', ha='center')
            _ax.set_ylabel(_ax.get_ylabel(), rotation='vertical', ha='center')
            _ax.set_xlim(0, _ax.get_xlim()[1])
            _ax.set_ylim(0, _ax.get_ylim()[1])
        ax1.set_ylabel('[um]')
        ax2.set_ylabel('[um]')
        ax1.set_title('distance from nuleus centroid')
        ax2.set_title('distance from cell centroid')
        ax2.set_xlabel('time [min]')
        ax2.xaxis.set_major_locator(MultipleLocator(30))
        plt.setp(ax2.get_xticklabels(), visible=True)
        plt.subplots_adjust(hspace=0.7)

        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_1_mother_daughter(df, dfc):
    # MOTHER-DAUGHTER
    _conds = ['mother-daughter']
    with PdfPages(parameters.data_dir + 'out/fig1-mother.pdf') as pdf:
        dfs, conds, colors = pl.sorted_conditions(df, _conds)

        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax1 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sp.msd(dfs[dfs['condition'] == pl.names['mother-daughter']], ax1, ylim=[0, 120])
        ax1.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax2 = fig.add_subplot(111, sharey=ax1)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            df_msd = ImagejPandas.msd_particles(dfs)
            df_msd.loc[df_msd['CentrLabel'] == 'A', 'CentrLabel'] = 'Mother'
            df_msd.loc[df_msd['CentrLabel'] == 'B', 'CentrLabel'] = 'Daugther'

            sns.tsplot(data=df_msd[df_msd['CentrLabel'] == 'Mother'], color='k', linestyle='-',
                       time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax2)
            sns.tsplot(data=df_msd[df_msd['CentrLabel'] == 'Daugther'], color='k', linestyle='--',
                       time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax2)
            ax2.set_ylim([0, 120])
            ax2.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
            ax2.set_xticks(np.arange(0, dfs['Time'].max(), 20.0))
            ax2.legend(title=None, loc='upper left')
            ax2.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sp.msd_indivs(dfs[dfs['condition'] == pl.names['mother-daughter']], ax3)
        ax3.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax4 = fig.add_subplot(111, sharey=ax3)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sns.tsplot(data=df_msd, lw=3, err_style=['unit_traces'], err_kws=pl._err_kws,
                       time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax4)
        ax4.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
        ax4.set_xticks(np.arange(0, dfs['Time'].max(), 20.0))
        ax4.legend(title=None, loc='upper left')
        ax4.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_2(df, dfc):
    with PdfPages(parameters.data_dir + 'out/fig2.pdf') as pdf:
        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=pl._err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        _conds = ['1_P.C.']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=pl._err_kws)
        _conds = ['1_DIC']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=pl._err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['2_Kines1', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=pl._err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '1_DIC']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(1.3, 1.3), dpi=_dpi)
        fig.clf()
        ax1 = fig.add_subplot(111)
        mua = dfcs.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        with sns.color_palette([pt_color for c in dfc['condition'].unique()]):
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1,
                                xlabels={'+STLC': '+STLC',
                                         'DHC+STLC': 'DHC\n+STLC',
                                         'DIC+STLC': 'DIC\n+STLC'})
        ax1.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '1_ASUND', '1_CENPF']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(1.3, 1.3), dpi=_dpi)
        fig.clf()
        ax1 = fig.add_subplot(111)
        mua = dfcs.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        with sns.color_palette([pt_color for c in dfc['condition'].unique()]):
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1, xlabels=lbl_dict)
        ax1.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        # _conds = ['1_P.C.', '1_Dynei', '1_ASUND', '1_CENPF']
        # dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.8, 1.3), dpi=_dpi)
        fig.clf()
        ax4 = fig.add_subplot(111)
        with sns.color_palette(
                [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE, sp.SUSSEX_FUSCHIA_PINK]):
            sp.congression(df, ax=ax4, order=conds)
            ax4.set_xlabel('time [min]')
            ax4.set_ylabel('% congression')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '2_CDK1_DC', '2_CDK1_DA']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(1.3, 1.3), dpi=_dpi)
        fig.clf()
        ax1 = fig.add_subplot(111)
        mua = dfcs.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        with sns.color_palette([pt_color for c in dfc['condition'].unique()]):
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1, xlabels=lbl_dict)
        ax1.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        # _conds = ['1_P.C.', '1_Dynei', '2_CDK1_DC', '2_CDK1_DA']
        # dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.8, 1.3), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        with sns.color_palette(
                [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE, sp.SUSSEX_FUSCHIA_PINK]):
            sp.congression(df, ax=ax, order=conds)
            ax.set_xlabel('time [min]')
            ax.set_ylabel('% congression')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '1_DIC']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.8, 1.3), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE]):
            sp.congression(df, ax=ax, order=conds)
            ax.set_xlabel('time [min]')
            ax.set_ylabel('% congression')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.3, 1.3), dpi=_dpi)
        fig.clf()
        ax5 = fig.add_subplot(111)
        sp.msd(df[df['condition'] == pl.names['1_DIC']], ax5, ylim=[0, 400])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

    mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    pmat = st.p_values(mua, 'SpeedCentr', 'condition', filename=parameters.data_dir + 'out/pvalues_spd_fig2.xls')


def fig_3hmm(df, dfc):
    with PdfPages(parameters.data_dir + 'out/fig3.pdf') as pdf:
        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        _conds = ['1_Dynei', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=pl._err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '2_Kines1', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette(
                [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE, sp.SUSSEX_FUSCHIA_PINK]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['ci_band'])
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '2_Kines1', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette(
                [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE, sp.SUSSEX_FUSCHIA_PINK]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=pl._err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '2_Kines1', '1_Dynei', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.1, 1.3), dpi=_dpi)
        fig.clf()
        ax2 = fig.add_subplot(111)
        mua = dfcs.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        with sns.color_palette([sp.SUSSEX_COBALT_BLUE for c in dfc['condition'].unique()]):
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax2,
                                xlabels={'+STLC': '+STLC',
                                         'Kinesin1+STLC': 'Kinesin1\n+STLC',
                                         'DHC+STLC': 'DHC\n+STLC',
                                         'DHC&Kinesin1+STLC': 'DHC & Kinesin1\n+STLC'})
        ax2.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '2_Kines1', '2_CDK1_DK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.8, 1.3), dpi=_dpi)
        fig.clf()
        ax4 = fig.add_subplot(111)
        with sns.color_palette(
                [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE, sp.SUSSEX_FUSCHIA_PINK]):
            sp.congression(df, ax=ax4, order=conds)
            ax4.set_xlabel('time [min]')
            ax4.set_ylabel('% congression')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_4(df, dfc, eb3df):
    with PdfPages(parameters.data_dir + 'out/fig4.pdf') as pdf:
        plot_eb3 = False
        if plot_eb3:
            # ---------------------------
            #    PAGE - time colorbar
            # ---------------------------
            fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi / 4)
            fig.clf()
            ax = fig.add_subplot(111)
            ax.axis('off')
            max_time = st.baseround(eb3df[eb3df['condition'] != 'eb3-g2-no-arrest']['time'].max())
            max_frame = eb3df['frame'].max()
            palette = sns.color_palette('cool', n_colors=max_frame)
            norm = mpl.colors.Normalize(vmin=0, vmax=max_time)
            sm = plt.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(palette), norm=norm)
            sm.set_array([])
            ticks = np.linspace(0, max_time, 5, endpoint=True, dtype=np.int)
            logger.debug('ticks: %s' % str(ticks))
            cb1 = plt.colorbar(sm, ax=ax, ticks=ticks, boundaries=np.arange(0, max_time + 1, 1), orientation='vertical')
            cb1 = plt.colorbar(sm, ax=ax, ticks=ticks, boundaries=np.arange(0, max_time + 1, 1),
                               orientation='horizontal')
            cb1.set_label('time [s]')
            pdf.savefig(transparent=True, bbox_inches='tight')

            # ---------------------------
            #    PAGE - tracks control
            # ---------------------------
            eb3fld = parameters.data_dir + 'out/data/lab/eb3'
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi / 4)
            fig.clf()
            ax = fig.add_subplot(111)
            eb3dfs = eb3df[eb3df['tag'] == 'U2OS CDK1as EB3 +1NM on controls only.sld - Capture 4']
            eb3.render_image_track(eb3dfs, ax, folder=eb3fld, point_size=0.1, line_width=0.05, palette=palette,
                                   tracks_to_show=300)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_xlim([0, 70])
            ax.set_ylim([0, 70])
            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.set_title('Control')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

            # ---------------------------
            #    PAGE - tracks MCAK
            # ---------------------------
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi / 4)
            fig.clf()
            ax = fig.add_subplot(111)
            eb3dfs = eb3df[eb3df['tag'] == 'U2OS CDK1as EB3 chTOG RNAi 3days +1NM on.sld - Capture 21']
            eb3.render_image_track(eb3dfs, ax, folder=eb3fld, point_size=0.1, line_width=0.05, palette=palette)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_xlim([0, 70])
            ax.set_ylim([0, 70])
            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.set_title('MCAK')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

            # ---------------------------
            #    PAGE - tracks chTog
            # ---------------------------
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi / 4)
            fig.clf()
            ax = fig.add_subplot(111)
            eb3dfs = eb3df[eb3df['tag'] == 'U2OS CDK1as EB3 chTOG RNAi 3days +1NM on.sld - Capture 6']
            eb3.render_image_track(eb3dfs, ax, folder=eb3fld, point_size=0.1, line_width=0.05, palette=palette)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_xlim([0, 70])
            ax.set_ylim([0, 70])
            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.set_title('chTog')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

            # ---------------------------
            #    PAGE - tracks nocodazole
            # ---------------------------
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi / 4)
            fig.clf()
            ax = fig.add_subplot(111)
            eb3dfs = eb3df[eb3df['tag'] == 'U2OS CDK1as EB3 1NM on +10ng Noco.sld - Capture 1 - 2hrs post 10ng noco']
            eb3.render_image_track(eb3dfs, ax, folder=eb3fld, point_size=0.1, line_width=0.05, palette=palette)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_xlim([0, 70])
            ax.set_ylim([0, 70])
            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.set_title('Nocodazole 10ng')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #    PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        _conds = ['1_P.C.']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=pl._err_kws)
        _conds = ['1_No10+']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=pl._err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.set_ylim([0, 30])
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #    PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        _conds = ['1_P.C.']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=pl._err_kws)
        _conds = ['1_MCAK']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=pl._err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.set_ylim([0, 30])
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #    PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        _conds = ['1_P.C.']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=pl._err_kws)
        _conds = ['1_chTOG']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=pl._err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.set_ylim([0, 30])
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_No10+', '1_MCAK', '1_chTOG']
        dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax = fig.gca()
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_TURQUOISE, sp.SUSSEX_SKY_BLUE])
        sp.congression(dfs, ax=ax, order=conds)
        ax.set_xlabel('time [min]')
        ax.set_ylabel('% congression')
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_No10+', '1_MCAK', '1_chTOG']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax2 = fig.add_subplot(111)
        mua = dfcs.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        with sns.color_palette(list(itertools.repeat(pt_color, 4))):
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax2,
                                xlabels={'+STLC': '+STLC',
                                         'Nocodazole 10ng+STLC': 'Noc 10ng\n+STLC',
                                         'MCAK+STLC': 'MCAK\n+STLC',
                                         'chTog+STLC': 'chTog\n+STLC'})
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        pmat = st.p_values(mua, 'SpeedCentr', 'condition',
                           filename=parameters.data_dir + 'out/pvalues_pc-noc-mcak_spd.xls')

        df = df[df['Time'] <= 50]
        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_No10+']
        dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax5 = fig.add_subplot(111)
        sp.msd(dfs, ax5, ylim=[0, 120])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_MCAK']
        dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax5 = fig.add_subplot(111)
        sp.msd(dfs, ax5, ylim=[0, 120])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        # _conds = ['1_MCAK']
        # dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax5 = fig.add_subplot(111)
        sp.msd_indivs(dfs, ax5, ylim=[0, 120])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_chTOG']
        dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax5 = fig.add_subplot(111)
        sp.msd(dfs, ax5, ylim=[0, 120])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        # _conds = ['1_chTOG']
        # dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax5 = fig.add_subplot(111)
        sp.msd_indivs(dfs, ax5, ylim=[0, 120])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_4_eb3_stats(eb3_stats, filename='fig4_boxplots.pdf', title=None):
    pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]
    with PdfPages(parameters.data_dir + 'out/%s' % filename) as pdf:
        # ---------------------------
        #    PAGE - Eb3 velocity boxplots
        # ---------------------------
        fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        df_stats = eb3_stats[
            (eb3_stats['condition'].isin(['eb3-control', 'eb3-nocodazole', 'eb3-mcak', 'eb3-chtog'])) & (
                    eb3_stats['speed'] > 1e-2)]
        with sns.color_palette([pt_color, pt_color, pt_color, pt_color]):
            sp.anotated_boxplot(df_stats, 'speed', point_size=0.5, ax=ax,
                                order=['eb3-control', 'eb3-nocodazole', 'eb3-mcak', 'eb3-chtog'],
                                xlabels={'eb3-control': '+STLC',
                                         'eb3-nocodazole': 'Noc 10ng\n+STLC',
                                         'eb3-mcak': 'MCAK\n+STLC',
                                         'eb3-chtog': 'chTog\n+STLC'})
            # sp.anotated_boxplot(df_stats, 'speed', point_size=0.5, ax=ax)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Average speed [um/s]')
        ax.set_ylim([0, 0.6])
        ax.set_title(title)
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # # ---------------------------
        # #    PAGE - Eb3 length boxplots
        # # ---------------------------
        # fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        # fig.clf()
        # ax = fig.add_subplot(111)
        # df_stats = eb3_stats[
        #     (eb3_stats['condition'].isin(['eb3-control', 'eb3-nocodazole', 'eb3-mcak', 'eb3-chtog'])) & (
        #             eb3_stats['speed'] > 1e-2)]
        # with sns.color_palette([pt_color, pt_color, pt_color, pt_color]):
        #     sp.anotated_boxplot(df_stats, 'length', point_size=0.5, ax=ax,
        #                         xlabels=['Control', 'Noc 10ng', 'MCAK', 'chTog'])
        # ax.set_xlabel('Condition')
        # ax.set_ylabel('Average length [um]')
        # ax.set_title(title)
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        pmat = st.p_values(df_stats, 'speed', 'condition',
                           filename=parameters.data_dir + 'out/pvalues_%s_ctr-noc-mcak-chtog.xls' % filename)

        # ---------------------------
        #    PAGE - Eb3 velocity boxplots
        # ---------------------------
        fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        fig.clf()
        ax = fig.add_subplot(111)
        df_stats = eb3_stats[(eb3_stats['condition'].isin(['eb3-control', 'eb3-g2-no-arrest'])) & (
                eb3_stats['speed'] > 1e-2)]
        with sns.color_palette([pt_color, pt_color, pt_color]):
            sp.anotated_boxplot(df_stats, 'speed', point_size=0.5, ax=ax,
                                order=['eb3-control', 'eb3-g2-no-arrest'],
                                xlabels={'eb3-control': '+STLC', 'eb3-g2-no-arrest': 'No arrest\n+STLC'})
        ax.set_xlabel('Condition')
        ax.set_ylabel('Average speed [um/s]')
        ax.set_ylim([0, 0.6])
        ax.set_title(title)
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # # ---------------------------
        # #    PAGE - Eb3 length boxplots
        # # ---------------------------
        # fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        # fig.clf()
        # ax = fig.add_subplot(111)
        # df_stats = eb3_stats[(eb3_stats['condition'].isin(['eb3-control', 'eb3-g2-no-arrest'])) & (
        #         eb3_stats['speed'] > 1e-2)]
        # with sns.color_palette([pt_color, pt_color, pt_color]):
        #     sp.anotated_boxplot(df_stats, 'length', point_size=0.5, ax=ax, xlabels=['+STLC', 'No arrest\n+STLC'])
        # ax.set_xlabel('Condition')
        # ax.set_ylabel('Average length [um]')
        # ax.set_title(title)
        # pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        pmat = st.p_values(df_stats, 'speed', 'condition',
                           filename=parameters.data_dir + 'out/pvalues_%s_ctr-g2.xls' % filename)
        # pmat = st.p_values(df_stats, 'length', 'condition', filename=parameters.data_dir+'out/data/lab/pvalues_len.xls')


def fig_5(df, dfc):
    with PdfPages(parameters.data_dir + 'out/fig5.pdf') as pdf:
        # ---------------------------
        #    PAGES - individuals
        # ---------------------------
        for _cond in ['1_FAKI', '1_CyDT', '1_Bleb']:
            dfcs, conds, colors = pl.sorted_conditions(dfc, [_cond])
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
            fig.clf()
            ax = fig.add_subplot(111)
            with sns.color_palette([sp.SUSSEX_COBALT_BLUE]):
                sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                           estimator=np.nanmean, ax=ax, lw=3,
                           err_style=['unit_traces'], err_kws=pl._err_kws)
                ax.set_xlabel('time prior contact [min]')
                ax.set_ylabel('Distance [um]')
                ax.legend(title=None, loc='upper left')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_FAKI', '1_CyDT', '1_Bleb']
        dfs, conds, colors = pl.sorted_conditions(df, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax = fig.gca()
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_TURQUOISE, sp.SUSSEX_SKY_BLUE])
        sp.congression(dfs, ax=ax, order=conds)
        ax.set_xlabel('time [min]')
        ax.set_ylabel('% congression')
        pdf.savefig(transparent=True, bbox_inches='tight')

        pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]
        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_FAKI', '1_CyDT', '1_Bleb']
        dfcs, conds, colors = pl.sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
        fig.clf()
        ax2 = fig.add_subplot(111)
        mua = dfcs.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        with sns.color_palette([pt_color, pt_color, pt_color, pt_color]):
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax2,
                                xlabels={'+STLC': '+STLC',
                                         'FAKi+STLC': 'FAKi\n+STLC',
                                         'Cytochalsin D+STLC': 'Cytochalsin D\n+STLC',
                                         'Blebbistatin+STLC': 'Blebbistatin\n+STLC'})
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)
        pmat = st.p_values(mua, 'SpeedCentr', 'condition',
                           filename=parameters.data_dir + 'out/pvalues_pc-noc-mcak_spd.xls')

        df = df[df['Time'] <= 50]
        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        for _conds in ['1_FAKI', '1_CyDT', '1_Bleb']:
            dfs, conds, colors = pl.sorted_conditions(df, [_conds])
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
            fig.clf()
            ax5 = fig.add_subplot(111)
            sp.msd(dfs, ax5, ylim=[0, 120])
            ax5.set_xlabel('time delay [min]')
            ax5.set_ylabel('MSD')
            ax5.legend(title=None, loc='upper left')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_3(data):
    msd_plots = pl.MSD(data)

    with PdfPages(parameters.data_dir + 'out/figure3.pdf') as pdf:
        # -----------
        # Page 1
        # -----------
        fig = plt.figure(dpi=_dpi, clear=True)
        ax = fig.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
        sp.set_axis_size(6, 6, ax=ax)

        msd_plots.msd_vs_congression(ax)

        ax.legend(loc='upper right')
        ax.set_ylabel('MSD')
        ax.set_xlabel('Congression [%]')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        pdf.savefig(transparent=True)


if __name__ == '__main__':
    logger.info('loading data')
    data = data.Data()

    #
    fig_1(data)
    # fig_1_selected_track(df_m, df_msk)
    # fig_1_mother_daughter(df_m, df_mc)
    # fig_2(df_m, dfcentr)
    # fig_3(df_m, dfcentr)
    #
    # df_eb3_flt = pd.read_pickle(parameters.data_dir + 'eb3filter.pandas')
    # fig_4(df_m, dfcentr, df_eb3_flt)
    # df_eb3_avg = pd.read_pickle(parameters.helfrid_lab_dir + 'eb3-nearest-3px/eb3stats.pandas')
    # fig_4_eb3_stats(df_eb3_avg, filename='fig4_boxplots_nearest3px.pdf',
    #                 title='Nearest velocity (3px) prediction algorithm')
    # df_eb3_avg = pd.read_pickle(parameters.data_dir + 'eb3stats.pandas')
    # fig_4_eb3_stats(df_eb3_avg, filename='fig4_boxplots_drift.pdf', title='Drift prediction algorithm')
    # fig_5(df_m, dfcentr)
    fig_3(data)

    # tom_plots()
