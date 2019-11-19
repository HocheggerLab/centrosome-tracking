import os
import logging

import numpy as np
import pandas as pd
from tools.matplotlib_essentials import plt
from matplotlib.backends.backend_pdf import PdfPages
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import seaborn as sns

import tools.image as tim
import mechanics as m
import data as hdata
import tools.data as data
import harryplots as hp
import parameters as p
import tools.plot_tools as sp
import plots._plots as pl
from tools.draggable import DraggableRectangle
from plots import merge, montage
from eb3._stats import Tracks
from eb3._plots import MSD
import tools.stats as st

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.width', 320)


def fig_1(data):
    trk_plots = pl.Tracks(data)
    msd_plots = pl.MSD(data)
    trk_harry = hp.Tracks()
    msd_harry = hp.MSD()

    logger.info('doing figure 1')
    with PdfPages(os.path.join(p.out_dir, 'figure1.pdf')) as pdf:
        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sns.xkcd_rgb["silver"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()

        trk_harry.for_condition(ax, '+1nmpp1+stlc-washout-STLC')
        trk_harry.format_axes(ax, time_minor=15, time_major=30)

        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        # sns.set_palette(['#000000', '#000000'])
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()

        msd_harry.displacement_more_less(ax, '+1nmpp1+stlc-washout-STLC')
        msd_harry.format_axes(ax, time_minor=15, time_major=30, msd_minor=25, msd_major=50)

        ax.set_ylim([0, 200])
        # ax.set_xlim([0, 90])
        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=p.dpi, clear=True)
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
        fig = plt.figure(dpi=p.dpi, clear=True)
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
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        # sns.set_palette(['#000000', '#000000'])
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()

        msd_plots.pSTLC(ax)
        msd_plots.format_axes(ax, time_minor=15, time_major=30, msd_minor=25, msd_major=50)

        # ax.set_xlim([-240, 30])
        ax.set_ylim([0, 300])
        ax.set_xlim([-240, 30])
        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=p.dpi, clear=True)
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
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()

        trk_harry.for_condition(ax, 'unperterbed')
        trk_harry.format_axes(ax, time_minor=15, time_major=30)

        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        # sns.set_palette(['#000000', '#000000'])
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()

        msd_harry.displacement_more_less(ax, 'unperterbed')
        msd_harry.format_axes(ax, time_minor=15, time_major=30, msd_minor=50, msd_major=150)

        ax.set_ylim([0, 600])
        # ax.set_xlim([0, 90])
        sp.set_axis_size(5, 2, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        # sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()

        msd_harry.mother_daughter(ax, 'unperterbed', 'sep')
        msd_harry.format_axes(ax, time_minor=5, time_major=20, msd_minor=25, msd_major=50)

        ax.set_ylim([0, 200])
        ax.set_xlim([0, 30])
        sp.set_axis_size(3, 3, ax=ax)
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sns.xkcd_rgb["grey"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()
        sp.set_axis_size(3.02, 1.5, ax=ax)

        dfh = hdata.harry_tidy_format()
        dfh = dfh[dfh['condition'].isin(['unperterbed', '+1nmpp1+stlc-washout-STLC'])]
        dfh = dfh[dfh['state'].isin(['sep'])]
        dd = dfh.groupby(['condition', 'indiv']).apply(m.avg_speed).rename('avg_speed').reset_index()
        dd = dd.dropna()

        dfs, cond = data.get_condition(['1_P.C.'])
        dfs.drop(
            columns=['NuclX', 'NuclY', 'Dist', 'Speed', 'Acc',
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


def fig_1sup(data):
    # trk_plots = pl.Tracks(data)
    msd_plots = pl.MSD(data)
    # trk_harry = hp.Tracks()
    msd_harry = hp.MSD()

    logger.info('doing supplementary of figure 1')
    with PdfPages(os.path.join(p.out_dir, 'figure1_sup.pdf')) as pdf:
        # ---------------------------
        #          PAGE
        # ---------------------------
        sns.set_palette([sp.SUSSEX_CORAL_RED, sns.xkcd_rgb["grey"], sp.SUSSEX_COBALT_BLUE])
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, )
        ax = fig.gca()
        for ((cond, state), msd), yl in zip(msd_harry.dft.groupby(['condition', 'state']),
                                            [50, 50, 50, 200, 100, 200]):
            ax.cla()

            msd_harry.mother_daughter(ax, cond, state)
            msd_harry.format_axes(ax)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(yl / 2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(yl / 4))

            ax.set_ylim([0, yl])
            ax.set_xlim([0, 30])

            pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          NEXT PAGE - MSD to compare nuclei wr to centrosome
        # ---------------------------
        sns.set_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED])
        ax.cla()

        msd_plots.pSTLC_nuclei_vs_slowc(ax)
        msd_plots.format_axes(ax, time_minor=15, time_major=30, msd_minor=15, msd_major=30)

        ax.set_ylim([0, 60])
        ax.set_xlim([0, 60])

        pdf.savefig(transparent=True, bbox_inches='tight')


def fig2_eb3_speed_boxplots():
    plt.style.use('bmh')

    algorithm = "eb3-drift-prediction"
    # algorithm = "eb3-nearest-3px"

    df = pd.read_pickle(os.path.join(p.compiled_data_dir, algorithm, "eb3filter.pandas"))
    # df = df[df['particle'] <= 10]
    print(df.columns)
    stats = Tracks(df, track_group=["condition", "tag", "particle"], cell_group=["condition", "tag"])
    cell = stats.per_cell()
    cell.to_csv(os.path.join(p.compiled_data_dir, algorithm, "cell.csv"))
    # cell = pd.read_csv(os.path.join(p.compiled_data_dir, algorithm, "cell.csv"))
    print(cell)

    cond_order = ['eb3-control', 'eb3-nocodazole', 'eb3-mcak', 'eb3-chtog']
    cell = cell[cell['condition'].isin(cond_order)]

    fig = plt.figure(figsize=(2.5, 1.5))
    fig.clf()
    ax = fig.gca()
    sns.set_palette([sp.SUSSEX_COBALT_BLUE] * 4)
    sp.anotated_boxplot(cell, 'speed|mean', group='condition', order=cond_order,
                        point_size=2.5, fontsize=7, stars=True,
                        xlabels={'eb3-control': '+STLC',
                                 'eb3-nocodazole': 'Noc 10ng\n+STLC',
                                 'eb3-mcak': 'MCAK\n+STLC',
                                 'eb3-chtog': 'chTog\n+STLC'},
                        ax=ax)
    ax.set_ylim([0, 0.3])
    figname = os.path.join(p.out_dir, "%s.speed.pdf" % algorithm)
    fig.savefig(figname, transparent=True, bbox_inches='tight', pad_inches=0.3)
    st.p_values(cell, 'speed|mean', 'condition',
                filename=os.path.join(p.out_dir, "%s.pvals.xls" % algorithm))


def fig2_eb3_msd():
    plt.style.use('bmh')

    algorithm = "eb3-drift-prediction"
    # algorithm = "eb3-nearest-3px"

    palette = [
        sp.colors.sussex_cobalt_blue,
        sp.colors.sussex_coral_red,
        sp.colors.sussex_turquoise,
        # sp.colors.sussex_sunshine_yellow,
        sp.colors.sussex_deep_aquamarine,
        # sp.colors.sussex_grape,
        sns.xkcd_rgb["grey"]
    ]
    sns.set_palette(palette)

    df = pd.read_pickle(os.path.join(p.compiled_data_dir, algorithm, "eb3filter.pandas"))
    # df = df[df['particle'] <= 10]
    # convert categorical data to int
    df.loc[:, 'tag'] = df['tag'].astype('category')
    df.loc[:, 'file_id'] = df['tag'].cat.codes
    df.loc[:, 'indv'] = df['condition'] + '|' + df['file_id'].map(str) + '|' + df['particle'].map(str)

    cond_order = df['condition'].unique()
    cond_order = ['eb3-control', 'eb3-nocodazole', 'eb3-mcak', 'eb3-chtog']
    df.loc[:, :] = df[df['condition'].isin(cond_order)]

    print(df.head(10))
    print(df.columns)
    print(cond_order)
    # exit(0)

    logging.info('making msd plots')
    msd = MSD(df, track_group=["condition", "tag", "particle"], cell_group=["condition", "tag"], uneven_times=True)
    # print(sorted(msd.timeseries['time'].unique()))

    fig = plt.figure(dpi=p.dpi, clear=True)
    sns.set_palette(palette)
    fig.set_size_inches((1.8, 1.8))
    ax = plt.gca()
    msd.track_each(ax, order=cond_order)
    fig.savefig(p.out_dir + 'msd_idv.pdf')

    fig = plt.figure(dpi=p.dpi, clear=True)
    sns.set_palette(palette)
    fig.set_size_inches((1.8, 1.8))
    ax = plt.gca()
    msd.track_average(ax, order=cond_order)
    fig.savefig(p.out_dir + 'msd_avg.pdf')


def fig_3(data):
    with PdfPages(os.path.join(p.out_dir, 'figure3.pdf')) as pdf:
        trk_plots = pl.Tracks(data)
        sns.set_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED])
        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.nocodazole(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.chTog(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.MCAK(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.blebbistatin(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.cytochalsinD(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(1.8, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.faki(ax, centered=True)
        ax.set_xlim([-100, 0])
        ax.set_ylim([0, 30])
        pdf.savefig(transparent=True, bbox_inches='tight')

        msd_plots = pl.MSD(data)
        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(dpi=p.dpi, clear=True)
        ax = fig.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
        sp.set_axis_size(6, 6, ax=ax)

        msd_plots.msd_vs_congression(ax)

        ax.legend(loc='upper right')
        ax.set_ylabel('MSD')
        ax.set_xlabel('Congression [%]')

        pdf.savefig(transparent=True)


def fig_4(data):
    trk_plots = pl.Tracks(data)
    with PdfPages(os.path.join(p.out_dir, 'figure4.pdf')) as pdf:
        sns.set_palette([sp.SUSSEX_CORAL_RED] * 5)
        # ---------------------------
        #          PAGE
        # ---------------------------
        fig = plt.figure(figsize=(3, 1.8), dpi=p.dpi, clear=True)
        ax = fig.gca()
        trk_plots.dist_toc(ax)
        pdf.savefig(transparent=True, bbox_inches='tight')


def ring_in_cell_lines_and_conditions():
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190924 - HCT106 (dynein eg5 phall)/MAX_actrng-hct_2019_09_24__15_32_15.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190924 - HCT106 (dynein eg5 phall)/MAX_actrng-hct_2019_09_24__16_10_36.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190924 - u2os -arr - cyto 2ng (dynein eg5 phall)/MAX_actrng-u2os-arr-cyto_2019_09_24__14_46_02.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190924 - u2os -arr - cyto 2ng (dynein eg5 phall)/MAX_actrng-u2os-arr-cyto_2019_09_24__15_09_14.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190924 - u2os -arr (dynein eg5 phall)/MAX_actrng-u2os-arrest_2019_09_24__14_13_01.tif"
    f = "/Volumes/Kidbeat/data/lab/airy-ring/20190921 - actrng - u2os - arrest/actrng-u2os-arrest_2019_09_21__12_14_31.czi"
    select_and_make_montage(f)


def dynein_eb3():
    f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - ctr- (dynein eg5)/MAX_actrng-ctr-_2019_09_27__18_42_05.tif"
    # f = "/Volumes/Kidbeat/data/lab/airy-dynein-eg5/20190927 - ctr- (dynein eg5)/MAX_actrng-ctr-_2019_09_27__18_34_14.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - ctr+ (dynein eg5)/MAX_actrng-ctr+_2019_09_27__22_57_33.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - ctr+ (dynein eg5)/MAX_actrng-ctr+_2019_09_27__23_07_07.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod- (dynein eg5)/MAX_actrng-fhod-_2019_09_27__23_19_56.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod- (dynein eg5)/MAX_actrng-fhod-_2019_09_27__23_32_17.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod- (dynein eg5)/MAX_actrng-fhod-_2019_09_27__23_41_34.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod- (dynein eg5)/MAX_actrng-fhod-_2019_09_27__23_52_29.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod+ (dynein eg5)/MAX_actrng-fhod+_2019_09_28__00_02_40.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod+ (dynein eg5)/MAX_actrng-fhod+_2019_09_28__00_13_31.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - fhod+ (dynein eg5)/MAX_actrng-fhod+_2019_09_28__00_21_20.tif"
    # f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - n2g+ (dynein eg5)/MAX_actrng-n2g+_2019_09_28__00_29_50.tif"
    f = "/Users/Fabio/data/lab/airy-dynein-eg5/20190927 - n2g+ (dynein eg5)/MAX_actrng-n2g+_2019_09_28__00_37_05.tif"
    # f = "/Volumes/Kidbeat/data/lab/airy-dynein-eg5/20190924 - u2os -arr - cyto 2ng (dynein eg5 phall)/MAX_actrng-u2os-arr-cyto_2019_09_24__14_46_02.tif"
    # f = "/Volumes/Kidbeat/data/lab/airy-dynein-eg5/20190924 - HCT106 (dynein eg5 phall)/MAX_actrng-hct_2019_09_24__15_32_15.tif"
    # f = "/Volumes/Kidbeat/data/lab/airy-dynein-eg5/20190924 - HCT106 (dynein eg5 phall)/MAX_actrng-hct_2019_09_24__16_10_36.tif"
    # f = ""

    select_and_make_montage(f)


def select_and_make_montage(f):
    # import numpy as np
    global x, y, winsize
    image, pix_per_um, dt, n_frames, n_channels, _ = tim.load_tiff(f)
    # image, pix_per_um, dt, n_frames, n_channels, _ = tim.load_zeiss(f)

    # cmaps = ['uscope_green', 'uscope_blue', 'uscope_magenta', 'uscope_red']
    # names = ['Phalloidin', 'DAPI', 'Eg5', 'DIC2']

    names = ['Eg5', 'DAPI', 'DIC2']
    cmaps = ['uscope_green', 'uscope_blue', 'uscope_red']
    order = [1, 0, 2]

    # names = ['Phalloidin', 'Eg5', 'DAPI', 'DIC2']
    # cmaps = ['uscope_green', 'uscope_green', 'uscope_blue', 'uscope_red']
    # order = [2, 1, 3]

    # image = np.expand_dims(image[6:9], axis=0)
    print(image[0].shape)

    # get portion of interest
    fig = plt.figure(figsize=(4, 4), dpi=p.dpi, clear=True)
    ax = fig.gca()
    merge(image[0], ax, um_per_pix=1 / pix_per_um,
          cmaps=cmaps, order=order, merge=order)
    # ---------------------------
    winsize = 15
    rect = patches.Rectangle((0, 0), 2 * winsize, 2 * winsize,
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    dr = DraggableRectangle(rect)
    dr.connect()
    # ---------------------------
    plt.show()
    logger.info("Making montage based on selection...")
    x, y = dr.rect.xy
    if x == 0 and y == 0: return
    x, y = dr.x0 + winsize, dr.y0 + winsize

    # save selection
    fig.savefig(os.path.join(p.out_dir, "%s.selection.pdf" % os.path.basename(f)))

    # zero portions of image that are not in the ROI for intensity rescale purposes
    _f, _c, h, w = image.shape
    c0 = int((x - winsize) * pix_per_um)
    c1 = int((x + winsize) * pix_per_um)
    r0 = int((y - winsize) * pix_per_um)
    r1 = int((y + winsize) * pix_per_um)
    image[:, :, :, 0:c0] = 0
    image[:, :, :, c1:w] = 0
    image[:, :, 0:r0, :] = 0
    image[:, :, r1:h, :] = 0

    g = montage(image[0], um_per_pix=1 / pix_per_um,
                xlim_um=[x - winsize, x + winsize], ylim_um=[y - winsize, y + winsize],
                cmaps=cmaps, order=order, ch_names=names,
                merge=order[-2:])
    g.savefig(os.path.join(p.out_dir, "%s.montage.pdf" % os.path.basename(f)), dpi=p.dpi)


def helfrid_plot():
    path = "/Volumes/Kidbeat/data/lab/compiled/Congression tracks.csv"
    df = pd.read_csv(path)
    df = pd.melt(df, id_vars='Time', value_name='Dist')
    v = df["variable"].str.split("_", n=1, expand=True)
    df.loc[:, "condition"] = v[0]
    df.loc[:, "run"] = v[1]
    df.drop(columns=["variable"], inplace=True)
    df.loc[:, "indv"] = df['condition'] + '|' + df['run']
    print(df)

    palette = [
        sp.colors.sussex_cobalt_blue,
        sp.colors.sussex_coral_red,
        sp.colors.sussex_turquoise,
        # sp.colors.sussex_sunshine_yellow,
        sp.colors.sussex_deep_aquamarine,
        # sp.colors.sussex_grape,
        sns.xkcd_rgb["grey"]
    ]
    cond_order = ['arrest', 'noc20ng', 'cyto 2ug', 'fhod1', 'n2g']
    sns.set_palette(palette)

    fig = plt.figure()
    fig.set_size_inches((3.5, 3.5))
    ax = plt.gca()
    sns.lineplot(data=df,
                 x='Time', y='Dist',
                 hue='condition',
                 estimator=np.nanmean, err_style=None,
                 lw=2, alpha=1, ax=ax)
    sns.lineplot(data=df,
                 x='Time', y='Dist',
                 hue='condition',
                 units='indv', estimator=None,
                 lw=0.2, alpha=1, ax=ax)
    # ax.legend(title=None, loc='upper left')
    ax.set_ylim([0, 35])
    ax.get_legend().remove()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Distance [um]')

    fig.savefig(os.path.join(p.out_dir, "congression.pdf"))


if __name__ == '__main__':
    data = data.Data()

    fig_1(data)
    fig_1sup(data)
    fig_3(data)
    fig_4(data)

    ring_in_cell_lines_and_conditions()
    dynein_eb3()
    helfrid_plot()
