import logging
import os
import random
import sys

import coloredlogs
import matplotlib as mpl
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import tools.image as image
import parameters as p
import mechanics as m
import tools.plot_tools as sp
from tools import stats as st

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
coloredlogs.install()
pd.set_option('display.width', 320)
indiv_idx = ['condition', 'tag', 'particle']


def df_filter(df, k=10, f=-1):
    logging.info('%d tracks before filter' % df.set_index(indiv_idx).index.unique().size)
    # filter dataframe for tracks having more than K points
    filtered_ix = df.set_index('frame').sort_index().groupby(indiv_idx).apply(lambda t: len(t.index) > k)
    df = df.set_index(indiv_idx)[filtered_ix].reset_index()
    logging.info('filtered %d tracks after selecting tracks with more that k=%d points' % (
        df.set_index(indiv_idx).index.unique().size, k))

    if f > 0:
        logging.info('filtering tracks on starting time.')
        # filter dataframe for tracks starting before frame f
        filtered_ix = df.set_index('frame').sort_index().groupby(indiv_idx).apply(lambda t: t.index[0] < f)
        df = df.set_index(indiv_idx)[filtered_ix].reset_index()
        logging.info('filtered %d tracks after selecting tracks starting before frame %d' % (
            df.set_index(indiv_idx).index.unique().size, f))

    return df


def indiv_plots(dff, df_stat, pdf_fname='eb3_indv.pdf'):
    with PdfPages(p.out_dir + '%s' % pdf_fname) as pdf:
        flatui = ['#9b59b6', '#3498db', '#95a5a6', '#e74c3c', '#34495e', '#2ecc71']
        palette = sns.color_palette(flatui)

        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(p.size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1: plt.Axes = plt.subplot(gs[0, 0])
        ax2: plt.Axes = plt.subplot(gs[0, 1])
        ax3: plt.Axes = plt.subplot(gs[1, 0])
        ax4: plt.Axes = plt.subplot(gs[1, 1])
        ax5: plt.Axes = plt.subplot(gs[2, 0])
        ax6: plt.Axes = plt.subplot(gs[2, 1])

        a = 1.0

        for (_id, fdf), _color in zip(dff.groupby('tag'), palette):
            fdf = fdf.groupby('particle')
            fdf.plot(x='time', y='dist', c=_color, lw=1, alpha=a, legend=False, ax=ax1)
            fdf.plot(x='time', y='dist_i', c=_color, lw=1, alpha=a, legend=False, ax=ax2)

            fdf.plot(x='time_i', y='dist', c=_color, lw=1, alpha=a, legend=False, ax=ax3)
            fdf.plot(x='time_i', y='dist_i', c=_color, lw=1, alpha=a, legend=False, ax=ax4)
        fig.suptitle('Distance')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            #     ax.legend([])
            ax.set_xlabel('Time $[s]$')

        ax1.set_title('Absolute distance over time')
        ax2.set_title('Incremental distance over time')

        pdf.savefig()
        plt.close()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(p.size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        dff.loc[:, 'speed'] = dff['speed'].abs()
        a = 0.2
        df = dff.reset_index()
        with sns.color_palette(flatui):
            sns.lineplot(data=df, x='time', y='speed', units='particle', hue='tag',
                         estimator=None, lw=0.2, alpha=1, legend=False, ax=ax1)
            sns.lineplot(data=df, x='time', y='speed', hue='tag',
                         estimator=np.nanmean, legend=False, ax=ax2)

            sns.lineplot(data=df, x='time_i', y='speed', units='particle', hue='tag',
                         estimator=None, lw=0.2, alpha=1, legend=False, ax=ax3)
            sns.lineplot(data=df, x='time_i', y='speed', hue='tag',
                         estimator=np.nanmean, legend=False, ax=ax4)

        for (_id, adf), _color in zip(df_stat.groupby('tag'), palette):
            adf.plot.scatter(x='time', y='speed', color=_color, alpha=a, ax=ax5)
            df_stat.plot.scatter(x='time', y='n_points', color=_color, alpha=a, ax=ax6)

            try:
                kde_avgspd = stats.gaussian_kde(adf['speed'])
                kde_trklen = stats.gaussian_kde(adf['n_points'])
                y_avgspd = np.linspace(adf['speed'].min(), adf['speed'].max(), 100)
                y_trklen = np.linspace(adf['n_points'].min(), adf['n_points'].max(), 100)
                x_avgspd = kde_avgspd(y_avgspd) * 5.0
                x_trklen = kde_trklen(y_trklen) * 100
                ax5.plot(x_avgspd, y_trklen, color=_color)  # gaussian kde
                ax6.plot(x_trklen, y_trklen, color=_color)  # gaussian kde
            except Exception:
                pass

        ax1.set(yscale='log')
        ax3.set(yscale='log')

        fig.suptitle('Speed')
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xlabel('Time $[s]$')
        ax5.set_ylim(0, df_stat['speed'].max())
        ax6.set_ylim(0, df_stat['n_points'].max())

        pdf.savefig()
        plt.close()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(p.size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        print(df_stat['tag'].unique())
        for (_id, adf), _color in zip(df_stat.groupby('tag'), palette):
            adf['speed'].plot.hist(20, color=_color, ax=ax3)
            adf['n_points'].plot.hist(20, color=_color, ax=ax4)

            try:
                kde_avgspd = stats.gaussian_kde(adf['speed'])
                kde_trklen = stats.gaussian_kde(adf['n_points'])
                x_avgspd = np.linspace(adf['speed'].min(), adf['speed'].max(), 100)
                x_trklen = np.linspace(adf['n_points'].min(), adf['n_points'].max(), 100)
                ax1.plot(x_avgspd, kde_avgspd(x_avgspd), color=_color)
                ax2.plot(x_trklen, kde_trklen(x_trklen), color=_color)  # gaussian kde
            except Exception:
                pass

        sns.distplot(df_stat['speed'].dropna(), ax=ax5)
        sns.distplot(df_stat['n_points'].dropna(), ax=ax6)

        ax1.set_title('Avg speed per track')
        ax2.set_title('Track length')
        for ax in [ax1, ax3, ax5]:
            ax.set_xlabel('Avg speed $[\mu m/s]$')
        for ax in [ax2, ax4, ax6]:
            ax.set_xlabel('N frames')

        pdf.savefig()
        plt.close()


def stats_plots(df, df_stats):
    with PdfPages(p.out_dir + 'boxplot_spd.pdf') as pdf:
        flatui = ['#9b59b6', '#3498db', '#95a5a6', '#e74c3c', '#34495e', '#2ecc71']
        palette = sns.color_palette(flatui)

        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches((10, 6.18))
        ax1 = plt.gca()

        df_stats = df_stats[df_stats['speed'] > 1e-2]
        with sns.color_palette([sp.SUSSEX_CORAL_RED, ] * 6):
            sp.anotated_boxplot(df_stats, 'speed', swarm=False, stars=True, point_size=1.5, ax=ax1)
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('Average Eb1 speed per particle $[\mu m \cdot s^{-1}]$')

        pdf.savefig()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches((10, 10))
        gs = matplotlib.gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])

        ptsize = 0.5
        with sns.color_palette(['grey', ] * 6):
            sp.anotated_boxplot(df_stats, 'speed', swarm=False, point_size=ptsize, ax=ax1)
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('Average Eb1 speed per particle track $[\mu m \cdot s^{-1}]$')

            sp.anotated_boxplot(df_stats, 'length', swarm=False, point_size=ptsize, ax=ax2)
            ax2.set_xlabel('Condition')
            ax2.set_ylabel('Average Eb1 length $[\mu m]$')

        with palette:
            for i, d in df_stats.groupby('condition'):
                sns.distplot(d['speed'], label=i, ax=ax3)
                sns.distplot(d['length'], label=i, ax=ax4)

        ax4.legend()
        for ax in [ax1, ax2]:
            for tick in ax.get_xticklabels():
                tick.set_label(tick.get_label()[4:])
                tick.set_rotation('vertical')

        pmat = st.p_values(df_stats, 'speed', 'condition', filename=p.out_dir + 'pvalues_spd.xls')
        pmat = st.p_values(df_stats, 'length', 'condition', filename=p.out_dir + 'pvalues_len.xls')

        pdf.savefig()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches((10, 10))
        gs = matplotlib.gridspec.GridSpec(2, 2)
        ax1: plt.Axes = plt.subplot(gs[0, 0])
        ax2: plt.Axes = plt.subplot(gs[0, 1])
        ax3: plt.Axes = plt.subplot(gs[1, 0])
        ax4: plt.Axes = plt.subplot(gs[1, 1])

        dfi = df.set_index(['condition', 'tag', 'particle', 'frame']).sort_index()
        totals = dfi.groupby('condition')['speed'].count()
        spd_gt = dfi[dfi['speed'] > 0.1].groupby('condition').count()['speed'] / totals * 100

        # plt.bar(range(4), [100] * 4, color='r', edgecolor='white', width=0.85)
        # plt.bar(range(4), spd_gt, color='#b5ffb9', edgecolor='white', width=0.85)
        # plt.xticks(range(4), ['chtog', 'control', 'mcak', 'noc'], rotation='vertical')
        ptsize = 0.5
        with sns.color_palette(['grey', 'grey', 'grey', 'grey', 'grey', 'grey']):
            sp.anotated_boxplot(df_stats, 'n_points', swarm=False, point_size=ptsize, ax=ax1)
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('N of points per particle track')

            df_stats.loc[:, 'norm_len'] = df_stats['length'] / df_stats['n_points']
            sp.anotated_boxplot(df_stats, 'norm_len', swarm=False, point_size=ptsize, ax=ax2)
            ax2.set_xlabel('Condition')
            ax2.set_ylabel('Normalized Eb1 length $[\mu m]$')

        sns.distplot(df_stats['speed'], ax=ax3)
        sns.distplot(df_stats['length'], ax=ax4)

        ax3.set_title('Avg speed per track')
        ax3.set_xlabel('Avg speed $[\mu m/s]$')
        ax4.set_title('Track length')
        ax4.set_xlabel('Avg length $[\mu m]$')
        for ax in [ax1, ax2]:
            ax.set_xlabel('N frames')

        # bugfix: rotate xticks for last subplot
        for tick in ax4.get_xticklabels():
            tick.set_rotation('horizontal')

        pdf.savefig()
        plt.close()


def msd_plots(df):
    dfn = df.reset_index()
    with PdfPages(p.out_dir + 'eb3_msd.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(p.size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1: plt.Axes = plt.subplot(gs[0:2, :])
        ax3: plt.Axes = plt.subplot(gs[2, 0])
        ax4: plt.Axes = plt.subplot(gs[2, 1])

        # plot of each eb3 track
        for _id, _df in df.groupby('particle'):
            ax1.scatter(x='x', y='y', c='frame', cmap='copper_r', data=_df)
            # dfi = _df.set_index('frame').sort_index()
            # ax1.text(dfi['x'].iloc[0], dfi['y'].iloc[0], '%d - %0.1f' % (_id, _df['msd'].iloc[-1]), fontsize=5)

        last_pt = [dm.iloc[-1] for _id, dm in df.groupby('particle')]
        msd_df = pd.DataFrame(last_pt)

        sns.stripplot(x=msd_df['msd'], jitter=True, ax=ax3)
        ax3.set_title('Distribution of %d individuals of $MSD(t_n)$' % len(msd_df.index))
        ax3.set_ylabel('Population')
        ax3.set_xlabel('Last MSD value $[\mu m]$')

        # plot of MSD for each track
        ax = ax4
        dfn.loc[:, 'indv'] = dfn['condition'] + '-' + dfn['tag'] + '-' + dfn['particle'].map(int).map(str)
        sns.lineplot(data=dfn, x='frame', y='msd',
                     units='indv', estimator=None, hue='condition',
                     lw=0.2, alpha=1, ax=ax)
        ax.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
        ax.legend(title=None, loc='upper left')
        ax.set_xticks(np.arange(0, dfn['frame'].max(), 5))
        ax.set_xlabel('Time delay $[frames]$')
        ax.set_xticks(range(0, dfn['frame'].max(), 5))
        ax.set_xlim([0, dfn['frame'].max()])

        pdf.savefig()
        plt.close()


def render_image_track(df, ax, folder, point_size=5, line_width=1, palette=None, colorbar=False,
                       tracks_to_show=np.infty):
    iname = df['tag'].iloc[0] + '.tif'
    logging.debug('reading %s' % iname)
    img, res, dt = image.find_image(iname, folder)
    max_time = df['time'].max()
    _, height, width = img.shape

    ax.cla()
    palgreys = sns.color_palette('Greys_r', n_colors=127)
    ax.imshow(img[1], extent=[0, width / res, height / res, 0], cmap=mpl.colors.ListedColormap(palgreys))

    if palette is None:
        palette = sns.color_palette('cool', n_colors=df['frame'].max())
        cmap = mpl.colors.ListedColormap(palette)
        norm = mpl.colors.Normalize(vmin=0, vmax=max_time)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if colorbar:
            cb1 = plt.colorbar(sm, ax=ax,
                               ticks=np.linspace(0, max_time, 5, endpoint=True, dtype=np.int),
                               boundaries=np.arange(0, max_time + 1, 1), orientation='horizontal')
            cb1.set_label('time [s]')

    if tracks_to_show < np.infty:
        df_flt = df[df['particle'].isin(random.sample(df['particle'].unique(), tracks_to_show))]
    else:
        df_flt = df
    for _id, df in df_flt.groupby('particle'):
        df.plot.scatter(x='x', y='y', c=palette, ax=ax, s=point_size)
        df.plot(x='x', y='y', c='y', legend=False, ax=ax, lw=line_width)
        # last_pt = df.set_index('frame').sort_index().iloc[-1]
        # ax.text(last_pt['x'], last_pt['y'], '%0.2f' % last_pt['s'], color='white', fontsize=7)

    ax.set_xlabel('X $[\mu m]$')
    ax.set_ylabel('Y $[\mu m]$')

    return ax


def render_image_tracks(df_total, folder='.'):
    for tag, dff in df_total.groupby('tag'):
        logging.info('rendering %s' % tag)
        try:
            fig = matplotlib.pyplot.gcf()
            fig.clf()
            fig.set_size_inches((10, 10))
            ax = fig.gca()
            render_image_track(dff, ax, folder)
            fig.savefig(os.path.abspath(os.path.join(folder, 'py-renders', tag + '-render.png')))
            return ax
        except Exception as e:
            logging.critical('couldn\'t do the plot. %s' % (e))


def batch_filter(df):
    logging.info('%d tracks prior to apply filters' % df.set_index(indiv_idx).index.unique().size)

    flt = df_filter(df, k=3)
    flt = m.get_msd(flt, group=indiv_idx)

    # filter dataframe based on track's displacement
    msd_thr = 5
    filtered_ix = flt.set_index('frame').sort_index().groupby(indiv_idx).apply(
        lambda t: t['msd'].iloc[-1] > msd_thr)
    flt = flt.set_index(indiv_idx)[filtered_ix].reset_index()
    logging.info('filtered %d tracks after MSD filter with msd_thr=%0.1f' % (
        flt.set_index(indiv_idx).index.unique().size, msd_thr))

    logging.info('computing speed, acceleration, length')
    flt = m.get_speed_acc(flt, group=indiv_idx)
    flt = m.get_center_df(flt, group=indiv_idx)
    flt = m.get_trk_length(flt, group=indiv_idx)

    # construct average track speed and track length
    dfi = flt.set_index('frame').sort_index()
    dfi['speed'] = dfi['speed'].abs()
    avg = dfi.groupby(indiv_idx)['time', 'speed'].mean()
    avg.loc[:, 'time'] = dfi.groupby(indiv_idx)['time'].first()
    avg.loc[:, 'n_points'] = dfi.groupby(indiv_idx)['x'].count()
    avg.loc[:, 'length'] = dfi.groupby(indiv_idx)['s'].agg(np.sum)
    avg = avg.reset_index()

    # # speed filter
    # print (dfi['speed'].describe())
    # speed_ix = dfi.groupby(indiv_idx).apply(lambda t: t['speed'].max() < 0.4)
    # df_flt = df_flt.set_index(indiv_idx)[speed_ix].reset_index()

    return flt, avg


if __name__ == '__main__':
    do_filter_stats = True
    # do_filter_stats = False

    if do_filter_stats:
        _df = pd.read_pickle(p.compiled_data_dir + 'eb3.pandas')
        df_flt, df_avg = batch_filter(_df)
        df_avg = df_avg.replace([np.inf, -np.inf], np.nan).dropna()
        df_flt.to_pickle(p.compiled_data_dir + 'eb3filter.pandas')
        df_avg.to_pickle(p.compiled_data_dir + 'eb3stats.pandas')
    else:
        if os.path.exists(p.compiled_data_dir + 'eb3_selected.pandas'):
            logging.info('Loading GUI selected features instead of filtered particles!')
            df_flt = pd.read_pickle(p.compiled_data_dir + 'eb3_selected.pandas')
            df_avg = pd.read_pickle(p.compiled_data_dir + 'eb3stats_sel.pandas')
        else:
            df_flt = pd.read_pickle(p.compiled_data_dir + 'eb3filter.pandas')
            df_avg = pd.read_pickle(p.compiled_data_dir + 'eb3stats.pandas')
        logging.info('Loaded %d tracks after filters' % df_flt.set_index(indiv_idx).index.unique().size)

    # # # logging.info('rendering images.')
    # # # render_image_tracks(df_flt, folder=p.experiments_dir + 'eb3')

    logging.info('making indiv plots')
    df_flt['time'] = df_flt['time'].apply(np.round, decimals=3)
    df_flt['time_i'] = df_flt['time_i'].apply(np.round, decimals=3)
    for id, _dff in df_flt.groupby('condition'):
        dfavg = df_avg[df_avg['condition'] == id]
        logging.info('Plotting individuals for %s group.' % id)
        indiv_plots(_dff, dfavg, pdf_fname='eb3_indv-%s.pdf' % id)

    logging.info('making stat plots')
    stats_plots(df_flt, df_avg)

    logging.info('making msd plots')
    msd_plots(df_flt)
