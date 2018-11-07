import itertools
import logging
import os
from collections import OrderedDict

import PIL
import coloredlogs
import matplotlib as mpl
import matplotlib.axes
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

import manual_data as md
import parameters
import plot_special_tools as sp
import stats as st
from imagej_pandas import ImagejPandas
import run_plots_eb3 as eb3

log = logging.getLogger(__name__)
log.info(font_manager.OSXInstalledFonts())
log.info(font_manager.OSXFontDirectories)

plt.style.use('bmh')
print(matplotlib.rcParams.keys())
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

names = OrderedDict([('1_N.C.', '-STLC'),
                     ('1_P.C.', '+STLC'),
                     ('mother-daughter', 'mother-daughter'),
                     ('1_DIC', 'DIC+STLC'),
                     ('1_Dynei', 'DHC+STLC'),
                     ('1_ASUND', 'Asunder+STLC'),
                     ('1_CENPF', 'CenpF+STLC'),
                     ('1_BICD2', 'Bicaudal+STLC'),
                     ('1_MCAK', 'MCAK+STLC'),
                     ('1_chTOG', 'chTog+STLC'),
                     ('2_Kines1', 'Kinesin1+STLC'),
                     ('2_CDK1_DA', 'DHC&Asunder+STLC'),
                     ('2_CDK1_DC', 'DHC&CenpF+STLC'),
                     ('2_CDK1_DK', 'DHC&Kinesin1+STLC'),
                     ('1_No10+', 'Nocodazole 10ng+STLC'),
                     ('1_CyDT', 'Cytochalsin D+STLC'),
                     ('1_Bleb', 'Blebbistatin+STLC'),
                     ('1_FAKI', 'FAKi+STLC'),
                     ('hset', 'Hset+STLC'),
                     ('kif25', 'Kif25+STLC'),
                     ('hset+kif25', 'Hset&Kif25+STLC'),
                     ('mother-daughter', '+STLC mother daughter')])
col_palette = ["#e74c3c", sp.SUSSEX_CORAL_RED,
               "#3498db", sns.xkcd_rgb["teal green"], "#9b59b6", "#2ecc71",
               sns.xkcd_rgb["windows blue"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["medium green"],
               '#91744B', sns.xkcd_rgb["pale red"], sns.xkcd_rgb["amber"], sns.xkcd_rgb["amber"],
               '#91744B', sns.xkcd_rgb["pale red"], sns.xkcd_rgb["amber"], sns.xkcd_rgb["amber"], "#3498db"]
cond_colors = dict(zip(names.keys(), itertools.cycle(col_palette)))
pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]
_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.5, 'lw': 0.1}
msd_ylim = [0, 420]
_dpi = 600
lbl_dict = {'+STLC': '+STLC',
            'mother-daughter': 'mother-daughter',
            'DHC+STLC': 'DHC\n+STLC',
            'Asunder+STLC': 'Asunder\n+STLC',
            'Bicaudal+STLC': 'Bicaudal\n+STLC',
            'MCAK+STLC': 'MCAK\n+STLC',
            'chTog+STLC': 'chTog\n+STLC',
            'DHC&CenpF+STLC': 'DHC&CenpF\n+STLC',
            'DHC&Asunder+STLC': 'DHC&Asunder\n+STLC',
            'CenpF+STLC': 'CenpF\n+STLC'}


def rename_conditions(df):
    for k, n in names.iteritems():
        df.loc[df['condition'] == k, 'condition'] = n
    return df


def sorted_conditions(df, original_conds):
    conditions = [names[c] for c in original_conds] if type(original_conds) == list else names[original_conds]
    _colors = [cond_colors[c] for c in original_conds] if type(original_conds) == list else cond_colors[original_conds]
    dfc = df[df['condition'].isin(conditions)] if type(original_conds) == list else df[df['condition'] == conditions]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc.loc[:, 'cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc = dfc.set_index(['cnd_idx', 'run', 'Nuclei', 'Frame', 'Time']).sort_index().reset_index()

    return dfc, conditions, _colors


def fig_1(df, dfc):
    _conds = ['1_N.C.', '1_P.C.']
    dfs, conds, colors = sorted_conditions(df, _conds)
    dfc, conds, colors = sorted_conditions(dfc, _conds)

    sp.render_tracked_centrosomes(parameters.data_dir + 'out/centrosomes.nexus.hdf5', 'pc', 'run_114', 2)
    img_fnames = [os.path.join(parameters.data_dir + 'out/data', 'run_114_N02_F%03d.png' % f) for f in range(20)]
    images = [PIL.Image.open(path) for path in img_fnames]
    pil_grid = sp.pil_grid(images, max_horiz=5)
    pil_grid.save(parameters.data_dir + 'out/data/fig1_grid.png')

    with PdfPages(parameters.data_dir + 'out/fig1.pdf') as pdf:
        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=_dpi)
        fig.clf()
        fig.set_size_inches(1.3, 1.3)
        ax = fig.gca()
        mua = dfs[dfs['CentrLabel'] == 'A'].groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax)
        pmat = st.p_values(mua, 'SpeedCentr', 'condition')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.text(0.5, 1.3, 'pvalue=%0.2e' % pmat[0, 1], ha='center', size='small')
        ax.set_ylabel('Average speed [um/min]')
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi)
        fig.clf()
        fig.set_size_inches(1.3, 1.3)
        ax = fig.gca()
        sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, lw=3,
                   ax=ax, err_style=['unit_traces'], err_kws=_err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi)
        fig.clf()
        fig.set_size_inches(w=2.8, h=1.3)
        ax = fig.gca()
        sp.congression(dfs, ax=ax, order=conds)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi)
        fig.clf()
        fig.set_size_inches(2.2, 2.2)
        ax = fig.gca(projection='3d')
        sp.ribbon(dfs[dfs['condition'] == '-STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax)
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(dpi=_dpi)
        fig.clf()
        fig.set_size_inches(2.2, 2.2)
        ax = fig.gca(projection='3d')
        sp.ribbon(dfs[dfs['condition'] == '+STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax)
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # MSD
        dfs = dfs[dfs['Time'] <= 50]
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax1 = fig.gca()
        sp.msd(dfs[dfs['condition'] == names['1_N.C.']], ax1, ylim=[0, 120])
        ax1.set_ylabel('Mean Square Displacement')
        ax1.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax2 = fig.gca(sharey=ax1)
        sp.msd(dfs[dfs['condition'] == names['1_P.C.']], ax2, ylim=[0, 120])
        ax2.set_ylabel('Mean Square Displacement')
        ax2.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax3 = fig.gca()
        sp.msd_indivs(dfs[dfs['condition'] == names['1_N.C.']], ax3, ylim=msd_ylim)
        ax3.set_ylabel('Mean Square Displacement')
        ax3.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax4 = fig.gca(sharey=ax3)
        sp.msd_indivs(dfs[dfs['condition'] == names['1_P.C.']], ax4, ylim=msd_ylim)
        ax4.set_ylabel('Mean Square Displacement')
        ax4.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # DISTANCES boxplot
        df_valid = df.loc[~df['CellBound'].isnull(), :]
        log.debug(df_valid.set_index(ImagejPandas.NUCLEI_INDIV_INDEX).sort_index().index.unique())
        log.debug(len(df_valid.set_index(ImagejPandas.NUCLEI_INDIV_INDEX).sort_index().index.unique()))

        stats = md.gen_dist_data(df[(df['condition'] == 'pc')])
        order = ['C1 (Away)', 'C2 (Close)', 'Nucleus\nCentroid', 'Cell\nCentroid', 'Cell\n(manual)']
        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        fig.clf()
        ax = fig.gca()
        sns.boxplot(data=stats, y='Dist', x='Type', order=order, width=0.5, linewidth=0.5, fliersize=0, ax=ax)
        for i, artist in enumerate(ax.artists):
            artist.set_facecolor('None')
            artist.set_edgecolor('k')
            artist.set_zorder(5000)
        for i, artist in enumerate(ax.lines):
            artist.set_color('k')
            artist.set_zorder(5000)
        sns.swarmplot(data=stats, y='Dist', x='Type', order=order, size=3, zorder=100, color=sp.SUSSEX_CORAL_RED,
                      ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Distance [um]')
        # ax2.yaxis.set_major_locator(MultipleLocator(5))
        pmat = st.p_values(stats, 'Dist', 'Type', filename=parameters.data_dir + 'out/fig1-pv-dist.xls')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(2.3, 1.5), dpi=_dpi)
        fig.clf()
        ax = fig.gca()
        sp.anotated_boxplot(stats, 'Dist', cat='Type', ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


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
        dfs, conds, colors = sorted_conditions(df, _conds)

        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax1 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sp.msd(dfs[dfs['condition'] == names['mother-daughter']], ax1, ylim=[0, 120])
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
            sp.msd_indivs(dfs[dfs['condition'] == names['mother-daughter']], ax3)
        ax3.set_xlabel('time delay [min]')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.4, 1.4), dpi=_dpi)
        fig.clf()
        ax4 = fig.add_subplot(111, sharey=ax3)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sns.tsplot(data=df_msd, lw=3, err_style=['unit_traces'], err_kws=_err_kws,
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=_err_kws)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=_err_kws)
        _conds = ['1_DIC']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=_err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['2_Kines1', '2_CDK1_DK']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=_err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '1_DIC']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        # dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        # dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        sp.msd(df[df['condition'] == names['1_DIC']], ax5, ylim=[0, 400])
        ax5.set_xlabel('time delay [min]')
        ax5.set_ylabel('MSD')
        ax5.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

    mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    pmat = st.p_values(mua, 'SpeedCentr', 'condition', filename=parameters.data_dir + 'out/pvalues_spd_fig2.xls')


def fig_3(df, dfc):
    with PdfPages(parameters.data_dir + 'out/fig3.pdf') as pdf:
        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        _conds = ['1_Dynei', '2_CDK1_DK']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=_err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_Dynei', '2_Kines1', '2_CDK1_DK']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        fig = plt.figure(figsize=(2.5, 2.5), dpi=_dpi)
        fig.clf()
        ax3 = fig.add_subplot(111)
        with sns.color_palette(
                [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_SKY_BLUE, sp.SUSSEX_FUSCHIA_PINK]):
            sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                       estimator=np.nanmean, ax=ax3, lw=3,
                       err_style=['unit_traces'], err_kws=_err_kws)
            ax3.set_xlabel('time prior contact [min]')
            ax3.set_ylabel('Distance [um]')
            ax3.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '2_Kines1', '1_Dynei', '2_CDK1_DK']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
            log.debug('ticks: %s' % str(ticks))
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=_err_kws)
        _conds = ['1_No10+']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=_err_kws)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=_err_kws)
        _conds = ['1_MCAK']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=_err_kws)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_COBALT_BLUE,
                   err_style=None, err_kws=_err_kws)
        _conds = ['1_chTOG']
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
        sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                   estimator=np.nanmean, ax=ax, lw=3, color=sp.SUSSEX_CORAL_RED,
                   err_style=['unit_traces'], err_kws=_err_kws)
        ax.set_xlabel('time prior contact [min]')
        ax.set_ylabel('Distance [um]')
        ax.set_ylim([0, 30])
        ax.legend(title=None, loc='upper left')
        pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_No10+', '1_MCAK', '1_chTOG']
        dfs, conds, colors = sorted_conditions(df, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
        dfs, conds, colors = sorted_conditions(df, _conds)
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
        dfs, conds, colors = sorted_conditions(df, _conds)
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
        # dfs, conds, colors = sorted_conditions(df, _conds)
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
        dfs, conds, colors = sorted_conditions(df, _conds)
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
        # dfs, conds, colors = sorted_conditions(df, _conds)
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
                           filename=parameters.data_dir + 'out/data/lab/pvalues_%s_ctr-noc-mcak-chtog.xls' % filename)

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
                           filename=parameters.data_dir + 'out/data/lab/pvalues_%s_ctr-g2.xls' % filename)
        # pmat = st.p_values(df_stats, 'length', 'condition', filename=parameters.data_dir+'out/data/lab/pvalues_len.xls')


def fig_5(df, dfc):
    with PdfPages(parameters.data_dir + 'out/fig5.pdf') as pdf:
        # ---------------------------
        #    PAGES - individuals
        # ---------------------------
        for _cond in ['1_FAKI', '1_CyDT', '1_Bleb']:
            dfcs, conds, colors = sorted_conditions(dfc, (_cond))
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
            fig.clf()
            ax = fig.add_subplot(111)
            with sns.color_palette([sp.SUSSEX_COBALT_BLUE]):
                sns.tsplot(data=dfcs, time='Time', value='DistCentr', unit='indv', condition='condition',
                           estimator=np.nanmean, ax=ax, lw=3,
                           err_style=['unit_traces'], err_kws=_err_kws)
                ax.set_xlabel('time prior contact [min]')
                ax.set_ylabel('Distance [um]')
                ax.legend(title=None, loc='upper left')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        _conds = ['1_P.C.', '1_FAKI', '1_CyDT', '1_Bleb']
        dfs, conds, colors = sorted_conditions(df, _conds)
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
        dfcs, conds, colors = sorted_conditions(dfc, _conds)
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
            dfs, conds, colors = sorted_conditions(df, _conds)
            fig = plt.figure(figsize=(1.8, 1.8), dpi=_dpi)
            fig.clf()
            ax5 = fig.add_subplot(111)
            sp.msd(dfs, ax5, ylim=[0, 120])
            ax5.set_xlabel('time delay [min]')
            ax5.set_ylabel('MSD')
            ax5.legend(title=None, loc='upper left')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


def fig_6(df):
    _conds = ['1_N.C.', '1_P.C.',
              '2_Kines1', '2_CDK1_DK', '1_DIC', '1_Dynei', '1_CENPF', '1_BICD2',
              '1_No10+', '1_MCAK', '1_chTOG',
              '1_CyDT', '1_FAKI', '1_ASUND']

    markers = ['o', 'o',
               's', 'X', 'v', '^', '<', '>',
               'p', 'P', 'X',
               'p', 'P', 'X']
    df, conds, colors = sorted_conditions(df, _conds)
    colors = [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]
    colors.extend([sp.SUSSEX_FLINT] * 6)
    colors.extend([sp.SUSSEX_FUSCHIA_PINK] * 3)
    colors.extend([sp.SUSSEX_TURQUOISE] * 3)
    colortuple = dict(zip(conds, colors))

    with PdfPages(parameters.data_dir + 'out/fig6.pdf') as pdf:
        # -----------
        # Page 1
        # -----------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(9.3, 9.3)
        ax = plt.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

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
        for c, m in zip(_conds, markers):
            cnd = names[c]
            p = df_msd_final[df_msd_final['condition'] == cnd]
            ax.scatter(p['cgr'], p['msd'], c=colortuple[cnd], s=200, label=cnd, marker=m, zorder=1000)

        ax.legend(loc='upper right')
        ax.set_ylabel('MSD $[\mu m^2]$')
        ax.set_xlabel('Congression [%]')

        # fig.patch.set_alpha(0.0)
        pdf.savefig(transparent=True)


def tom_plots():
    pass


def color_keys(df, dfc):
    cldf, conds, colors = sorted_conditions(df, names.keys())
    coldf, conds, colors = sorted_conditions(dfc, names.keys())
    with PdfPages(parameters.data_dir + 'out/colors.pdf') as pdf:
        with sns.color_palette(colors):
            # ---------------------------
            #          FIRST PAGE
            # ---------------------------
            fig = plt.figure(figsize=(20, 12.4), dpi=_dpi)
            fig.clf()
            mua = coldf.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds)
            fig.gca().set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)

            # ---------------------------
            #          NEXT PAGE
            # ---------------------------
            fig = plt.figure(figsize=(20, 12.4), dpi=_dpi)
            fig.clf()
            mua = cldf.groupby(ImagejPandas.CENTROSOME_INDIV_INDEX).mean().reset_index()
            sp.anotated_boxplot(mua, 'v_s', order=conds)
            pdf.savefig(transparent=True, bbox_inches='tight', pad_inches=0.3)


if __name__ == '__main__':
    df_m = pd.read_pickle(parameters.data_dir + 'merge.pandas')
    df_msk = pd.read_pickle(parameters.data_dir + 'mask.pandas')
    df_mc = pd.read_pickle(parameters.data_dir + 'merge_centered.pandas')

    df_m = df_m.loc[df_m['Time'] >= 0, :]
    df_m = df_m.loc[df_m['Time'] <= 100, :]
    df_mc = df_mc.loc[df_mc['Time'] <= 0, :]
    df_mc = df_mc.loc[df_mc['Time'] >= -100, :]

    # filter original dataframe to get just data between centrosomes
    dfcentr = df_mc[df_mc['CentrLabel'] == 'A']
    dfcentr['indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)
    dfcentr.drop(
        ['CentrLabel', 'Centrosome', 'NuclBound', 'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY', 'Speed', 'Acc'],
        axis=1, inplace=True)

    df_m.loc[:, 'indv'] = df_m['condition'] + '-' + df_m['run'] + '-' + df_m['Nuclei'].map(int).map(str) + '-' + \
                          df_m['Centrosome'].map(int).map(str)
    df_mc.loc[:, 'indv'] = df_mc['condition'] + '-' + df_mc['run'] + '-' + df_mc['Nuclei'].map(int).map(str) + '-' + \
                           df_mc['Centrosome']

    for id, dfc in df_m.groupby(['condition']):
        log.info('condition %s: %d tracks' % (id, len(dfc['indv'].unique()) / 2.0))
    df_m = rename_conditions(df_m)
    df_mc = rename_conditions(df_mc)
    dfcentr = rename_conditions(dfcentr)

    # filter starting distances greater than a threshold
    # indivs_filter = dfcentr.set_index(['Time', 'indv']).unstack('indv')['DistCentr'].fillna(method='bfill').iloc[0]
    # indivs_filter = indivs_filter[indivs_filter > 5].index.values
    # dfcentr = dfcentr[dfcentr['indv'].isin(indivs_filter)]

    print(df_m['indv'].unique().size)
    # df_m = m.get_trk_length(df_m, x='CentX', y='CentY', time='Time', frame='Frame',
    #                         group=ImagejPandas.CENTROSOME_INDIV_INDEX)
    # print df_m['s']
    color_keys(df_m, dfcentr)

    fig_1(df_m, dfcentr)
    fig_1_selected_track(df_m, df_msk)
    fig_1_mother_daughter(df_m, df_mc)
    fig_2(df_m, dfcentr)
    fig_3(df_m, dfcentr)

    df_eb3_flt = pd.read_pickle(parameters.data_dir + 'eb3filter.pandas')
    fig_4(df_m, dfcentr, df_eb3_flt)
    df_eb3_avg = pd.read_pickle(parameters.helfrid_lab_dir + 'eb3-nearest-3px/eb3stats.pandas')
    fig_4_eb3_stats(df_eb3_avg, filename='fig4_boxplots_nearest3px.pdf',
                    title='Nearest velocity (3px) prediction algorithm')
    df_eb3_avg = pd.read_pickle(parameters.data_dir + 'eb3stats.pandas')
    fig_4_eb3_stats(df_eb3_avg, filename='fig4_boxplots_drift.pdf', title='Drift prediction algorithm')
    fig_5(df_m, dfcentr)
    fig_6(df_m)

    # tom_plots()
