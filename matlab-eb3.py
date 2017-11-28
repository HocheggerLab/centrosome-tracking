import logging
import os
import sys
import time

import cv2
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import tifffile as tf
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn import linear_model

import mechanics as m
import special_plots as sp
import stats as st

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
pd.set_option('display.width', 320)
indiv_idx = ['condition', 'tag', 'trk']


def import_eb3_matlab(gen_filename, trk_filename, tag=None, limit=None):
    g = sio.loadmat(gen_filename, squeeze_me=True)
    x = sio.loadmat(trk_filename, squeeze_me=True)
    time_interval = g['timeInterval']
    tracks = x['tracksFinal']
    tracked_particles = tracks['tracksCoordAmpCG']
    trk_events = tracks['seqOfEvents']
    num_tracks = tracked_particles.shape[0]
    df_out = pd.DataFrame()
    logging.info('tracked_particles.shape=%s  trk_events.shape=%s' % (tracked_particles.shape, trk_events.shape))

    for ti, (_trk, _ev) in enumerate(zip(tracked_particles, trk_events)):
        if ti == limit:
            logging.debug('breaking on limit')
            break
        if _ev.shape == (2, 4):
            ini_frame = _ev[0, 0] if _ev[0, 1] == 1 else None
            end_frame = _ev[1, 0] + 1 if _ev[1, 1] == 2 else None
            if ini_frame is not None and end_frame is not None:
                trk = np.reshape(_trk, [len(_trk) / 8, 8])
                _df = pd.DataFrame(data=trk[:, 0:2], columns=['x', 'y'])
                _df.loc[:, 'trk'] = ti
                _df.loc[:, 'frame'] = np.arange(ini_frame, end_frame, 1, np.int16)
                _df.loc[:, 'time'] = _df['frame'] * time_interval

                df_out = df_out.append(_df)
            else:
                raise Exception('Invalid event format.')

    if tag is not None:
        df_out['tag'] = tag

    return df_out.reset_index().drop('index', axis=1)


def df_filter(df, k=10, msd_thr=50.0):
    logging.info('%d tracks before filter' % (df.set_index(indiv_idx).index.unique().size))
    # filter dataframe for tracks having more than K points
    filtered_ix = df.set_index('frame').sort_index().groupby(indiv_idx).apply(lambda t: len(t.index) > k)
    df = df.set_index(indiv_idx)[filtered_ix].reset_index()
    logging.info('filtered %d tracks after selecting tracks with more that k=%d points' % (
        df.set_index(indiv_idx).index.unique().size, k))

    # filter dataframe based on track's mobility
    filtered_ix = df.set_index('frame').sort_index().groupby(indiv_idx).apply(lambda t: t['msd'].iloc[-1] > msd_thr)
    df = df.set_index(indiv_idx)[filtered_ix].reset_index()
    logging.info('filtered %d tracks after MSD filter with msd_thr=%0.1f' % (
        df.set_index(indiv_idx).index.unique().size, msd_thr))

    return df


def indiv_plots(dff, df_stat, pdf_fname='eb3_indv.pdf'):
    with PdfPages('/Users/Fabio/data/lab/%s' % pdf_fname) as pdf:
        _err_kws = {'alpha': 0.3, 'lw': 1}
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        palette = sns.color_palette(flatui)

        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        a = 1.0

        for (id, fdf), _color in zip(dff.groupby('tag'), palette):
            fdf = fdf.groupby('trk')
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
        fig.set_size_inches(_fig_size_A3)
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
            sns.tsplot(data=df, time='time', value='speed', unit='trk', condition='tag',
                       estimator=np.nanmean, err_style=['unit_traces'], err_kws=_err_kws, ax=ax1)
            sns.tsplot(data=df, time='time', value='speed', unit='trk', condition='tag', estimator=np.nanmean,
                       legend=False, ax=ax2)

            sns.tsplot(data=df, time='time_i', value='speed', unit='trk', condition='tag',
                       estimator=np.nanmean, err_style=['unit_traces'], err_kws=_err_kws, ax=ax3)
            sns.tsplot(data=df, time='time_i', value='speed', unit='trk', condition='tag', estimator=np.nanmean,
                       legend=False, ax=ax4)

        for (id, adf), _color in zip(df_stat.groupby('tag'), palette):
            adf.plot.scatter(x='time', y='speed', color=_color, alpha=a, ax=ax5)
            df_stat.plot.scatter(x='time', y='trk_len', color=_color, alpha=a, ax=ax6)

            try:
                kde_avgspd = stats.gaussian_kde(adf['speed'])
                kde_trklen = stats.gaussian_kde(adf['trk_len'])
                y_avgspd = np.linspace(adf['speed'].min(), adf['speed'].max(), 100)
                y_trklen = np.linspace(adf['trk_len'].min(), adf['trk_len'].max(), 100)
                x_avgspd = kde_avgspd(y_avgspd) * 5.0
                x_trklen = kde_trklen(y_trklen) * 100
                ax5.plot(x_avgspd, y_trklen, color=_color)  # gaussian kde
                ax6.plot(x_trklen, y_trklen, color=_color)  # gaussian kde
            except:
                pass

        ax1.set(yscale='log')
        ax3.set(yscale='log')

        fig.suptitle('Speed')
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xlabel('Time $[s]$')
        ax5.set_ylim(0, df_stat['speed'].max())
        ax6.set_ylim(0, df_stat['trk_len'].max())

        pdf.savefig()
        plt.close()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        print df_stat['tag'].unique()
        for (id, adf), _color in zip(df_stat.groupby('tag'), palette):
            adf['speed'].plot.hist(20, color=_color, ax=ax3)
            adf['trk_len'].plot.hist(20, color=_color, ax=ax4)

            try:
                kde_avgspd = stats.gaussian_kde(adf['speed'])
                kde_trklen = stats.gaussian_kde(adf['trk_len'])
                x_avgspd = np.linspace(adf['speed'].min(), adf['speed'].max(), 100)
                x_trklen = np.linspace(adf['trk_len'].min(), adf['trk_len'].max(), 100)
                ax1.plot(x_avgspd, kde_avgspd(x_avgspd), color=_color)
                ax2.plot(x_trklen, kde_trklen(x_trklen), color=_color)  # gaussian kde
            except:
                pass

        sns.distplot(df_stat['speed'], ax=ax5)
        sns.distplot(df_stat['trk_len'], ax=ax6)

        ax1.set_title('Avg speed per track')
        ax2.set_title('Track length')
        for ax in [ax1, ax3, ax5]:
            ax.set_xlabel('Avg speed $[\mu m/s]$')
        for ax in [ax2, ax4, ax6]:
            ax.set_xlabel('N frames')

        pdf.savefig()
        plt.close()


def stats_plots(df, df_stats, res, img_file=None):
    with PdfPages('/Users/Fabio/data/lab/boxplot_spd.pdf') as pdf:
        _err_kws = {'alpha': 0.3, 'lw': 1}
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        palette = sns.color_palette(flatui)

        # ---------------------------
        #          FIRST PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches((10, 10))
        gs = matplotlib.gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        # ax4 = plt.subplot(gs[1, 1])

        ptsize = 0.5
        with sns.color_palette(['grey', 'grey', 'grey', 'grey']):
            sp.anotated_boxplot(df_stats, 'speed', swarm=False, point_size=ptsize, ax=ax1)
            pmat = st.p_values(df_stats, 'speed', 'condition', filename='/Users/Fabio/data/lab/pvalues_spd.xls')
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('Average Eb1 speed $[\mu m \cdot s^{-1}]$')

            sp.anotated_boxplot(df_stats, 'length', swarm=False, point_size=ptsize, ax=ax2)
            ax2.set_xlabel('Condition')
            ax2.set_ylabel('Average Eb1 length $[\mu m]$')

            sp.anotated_boxplot(df_stats, 'trk_len', swarm=False, point_size=ptsize, ax=ax3)
            ax3.set_xlabel('Condition')
            ax3.set_ylabel('Eb1 track length $[a.u.]$')

        pdf.savefig()
        plt.close()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches((10, 10))

        dfi = df.set_index(['condition', 'tag', 'trk', 'frame']).sort_index()
        totals = dfi.groupby('condition')['speed'].count()
        spd_gt = dfi[dfi['speed'] > 0.1].groupby('condition').count()['speed'] / totals * 100

        plt.bar(range(4), [100] * 4, color='r', edgecolor='white', width=0.85)
        plt.bar(range(4), spd_gt, color='#b5ffb9', edgecolor='white', width=0.85)
        plt.xticks(range(4), ['chtog', 'control', 'mcak', 'noc'], rotation='vertical')

        plt.xlabel('Condition')
        plt.ylabel('Percentage of speeds higher than a thr (green)')

        pdf.savefig()
        plt.close()

        with PdfPages('/Users/Fabio/data/lab/eb3_stats.pdf') as pdf:
            fig = matplotlib.pyplot.gcf()
            fig.clf()
            fig.set_size_inches(_fig_size_A3)
            gs = matplotlib.gridspec.GridSpec(3, 2)
            ax1 = plt.subplot(gs[0:2, :])
            ax3 = plt.subplot(gs[2, 0])
            ax4 = plt.subplot(gs[2, 1])
            max_frame = df['frame'].max()
            cmap = sns.color_palette('cool', n_colors=max_frame)

            # plot of each eb3 track
            _ax = ax1
            if img_file is not None:
                image = cv2.imread(img_file)
                _ax.imshow(image, extent=[0, 512 / res, 512 / res, 0])
            for _id, df in df.groupby('trk'):
                df.plot.scatter(x='x', y='y', c=cmap, ax=_ax, s=5)

            _ax.spines['top'].set_visible(False)
            _ax.spines['right'].set_visible(False)
            _ax.spines['bottom'].set_visible(False)
            _ax.spines['left'].set_visible(False)
            _ax.set_xlabel('X $[\mu m]$')
            _ax.set_ylabel('Y $[\mu m]$')

            # plot MSD sum distribution on semilog space
            msd_bins = np.logspace(-2, 4, 100)
            last_pt = [dm.iloc[-1] for _id, dm in df.groupby('trk')]
            # sns.distplot(last_msd, rug=True, bins=msd_bins, ax=ax3)
            msd_df = pd.DataFrame(last_pt)
            msd_df['condition'] = 'dummy'
            sns.stripplot(x=msd_df['msd'], jitter=True, ax=ax3)
            ax3.set_title('Distribution of $MSD(t_n)$')
            ax3.set_ylabel('Frequency')
            ax3.set_xlabel('$\sum MSD$ $[\mu m]$')
            # ax3.set_xticks(range(0,20,5).extend(range(40,100,20)))
            # ax3.set_xscale('log')

            # sns.distplot(df_stats['d'], rug=True, ax= ax4)
            ax4.set_title('Distribution of track lengths')
            ax4.set_ylabel('Frequency')
            ax4.set_xlabel('Length $[\mu m]$')

            pdf.savefig(transparent=True)
            plt.close()

            # ---------------------------
            #          NEXT PAGE
            # ---------------------------
            fig = matplotlib.pyplot.gcf()
            fig.clf()
            fig.set_size_inches(_fig_size_A3)
            gs = matplotlib.gridspec.GridSpec(3, 2)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[1, 0])
            ax4 = plt.subplot(gs[1, 1])

            for _id, df in df.groupby('trk'):
                df.plot(x='time', y='dist', c=cmap, ax=ax1, lw=1, alpha=0.3, color='gr')
                df.plot(x='time', y='speed', c=cmap, ax=ax2, lw=1, alpha=0.3, color='gr')

            pdf.savefig()
            plt.close()


def msd_plots(df):
    _df = df.reset_index()
    with PdfPages('/Users/Fabio/eb3_msd.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0:2, :])
        ax3 = plt.subplot(gs[2, 0])
        ax4 = plt.subplot(gs[2, 1])

        # plot of each eb3 track
        max_frame = _df['frame'].max()
        cmap = sns.color_palette('copper_r', n_colors=max_frame)
        for _id, _df in _df.groupby('trk'):
            _df.plot.scatter(x='x', y='y', c=cmap, ax=ax1)
            dfi = _df.set_index('frame').sort_index()
            ax1.text(dfi['x'].iloc[0], dfi['y'].iloc[0], '%d - %0.1f' % (_id, _df['msd'].iloc[-1]), fontsize=5)

        last_pt = [dm.iloc[-1] for _id, dm in df.groupby('trk')]
        msd_df = pd.DataFrame(last_pt)
        msd_df['condition'] = 'dummy'
        sns.stripplot(x=msd_df['msd'], jitter=True, ax=ax3)
        ax3.set_title('Distribution of %d individuals of $MSD(t_n)$' % len(msd_df.index))
        ax3.set_ylabel('Population')
        ax3.set_xlabel('Last MSD value $[\mu m]$')

        # plot of MSD for each track
        _ax = ax4
        sns.tsplot(data=_df, lw=3,
                   err_style=['unit_traces'], err_kws=_err_kws,
                   time='frame', value='msd', unit='trk', estimator=np.nanmean, ax=_ax)
        _ax.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
        _ax.set_xticks(np.arange(0, _df['frame'].max(), 5))
        _ax.legend(title=None, loc='upper left')
        _ax.set_xlabel('Time delay $[frames]$')
        _ax.set_xticks(range(0, _df['frame'].max(), 5))
        _ax.set_xlim([0, _df['frame'].max()])

        pdf.savefig()
        plt.close()


def est_lines(df, n, res, ax=None, dray=10.0):
    """
        Plots linear regression of the line constructed by the first n points
    """
    t0 = time.time()
    trk_lr = pd.DataFrame()
    for k, (_id, _df) in enumerate(df.groupby('trk')):
        # do linear regression of both tracks to see which has higher slope
        x = _df['x'].iloc[0:n]
        y = _df['y'].iloc[0:n]
        x = x.values.reshape(n, 1)
        y = y.values.reshape(n, 1)
        if np.isnan(y).any() or np.isnan(x).any():
            logging.warn('Some (x,y) points of track tag %d contains NaNs.' % _id)
        else:
            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            a = regr.coef_[0][0]
            b = regr.intercept_[0]
            alpha = np.arctan(a)
            if x[0] < x[1]:
                xf = x[0][0] - dray * np.cos(alpha)
            else:
                xf = x[0][0] + dray * np.cos(alpha)
            yf = a * xf + b

            trklfit = pd.DataFrame([{'_id': k, 'trk': _id, 'a': a, 'b': b,
                                     'xs': x[0][0], 'ys': y[0][0],
                                     'xf': xf, 'yf': yf}])
            trk_lr = trk_lr.append(trklfit)
    t1 = time.time()
    logging.info('Linear regressions done in %0.2f seconds.' % (t1 - t0))
    trk_lr.set_index('_id', inplace=True)

    xi = list()
    yi = list()
    if ax is not None:
        for row in trk_lr.iterrows():
            _df = row[1]
            ax.plot([_df['xs'], _df['xf']], [_df['ys'], _df['yf']], lw=0.5, c='k', alpha=0.1)

    # setting trk column as index and computing upper right matrix of line intersections
    # consider a square matrix of M x M, M=len(trk_lr['trk'].unique())
    # then, following the convention of numbering each cell from 0 to M^2-1, incrementing first columns, then rows
    # then we have that Cell(Ln,Lm) = (n-1)M + (m-1)
    M = trk_lr.index.size
    logging.info('M^2=%d' % M ** 2)
    t0 = time.time()
    for n in range(0, M):
        for m in range(n + 1, M):
            logging.debug('(%d,%d) -> (%d,%d) %d' % (n, m, trk_lr.iloc[n]['trk'], trk_lr.iloc[m]['trk'], n * M + m))
            l1 = trk_lr.iloc[n]
            l2 = trk_lr.iloc[m]
            # xsi = (l2['b'] - l1['b']) / (l1['a'] - l2['a'])
            # ysi = xsi * l1['a'] + l1['b']
            # xs = np.array([xsi, ysi])
            a = np.array([[-l1['a'], 1], [-l2['a'], 1]])
            b = np.array([l1['b'], l2['b']])
            xs = np.linalg.solve(a, b)
            xsi = xs[0]
            ysi = xs[1]

            in_range = [0 < v and v < 512 / res for v in [l1['xs'], l1['xf'], l2['xs'], l2['xf']]]
            in_l1x1 = l1['xs'] < xsi < l1['xf']
            in_l1x2 = l1['xf'] < xsi < l1['xs']
            in_l2x1 = l2['xs'] < xsi < l2['xf']
            in_l2x2 = l2['xf'] < xsi < l2['xs']
            in_l1y1 = l1['ys'] < ysi < l1['yf']
            in_l1y2 = l1['yf'] < ysi < l1['ys']
            in_l2y1 = l2['ys'] < ysi < l2['yf']
            in_l2y2 = l2['yf'] < ysi < l2['ys']
            if in_range and sum([in_l1x1, in_l1x2, in_l2x1, in_l2x2]) == 2 and \
                            sum([in_l1y1, in_l1y2, in_l2y1, in_l2y2]) == 2:
                xi.append(xsi)
                yi.append(ysi)
    t1 = time.time()

    logging.info('Line intersections done in %0.2f seconds.' % (t1 - t0))

    ax.scatter(xi, yi, s=20, c='k', lw=0)
    ax.scatter(trk_lr['xs'], trk_lr['ys'], s=10, c='b', lw=0, alpha=1)
    ax.scatter(trk_lr['xf'], trk_lr['yf'], s=5, c='r', lw=0, alpha=1)
    ax.set_aspect('equal', 'datalim')

    return trk_lr, np.array([xi, yi]).T


def est_plots(df_matlab, res):
    with PdfPages('/Users/Fabio/eb3_estimation.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0:2, :])
        # ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[2, 0])
        # ax4 = plt.subplot(gs[2, 1])

        max_frame = df_matlab.reset_index()['frame'].max()
        cmap = sns.color_palette('GnBu_r', n_colors=max_frame)

        # plot of each eb3 track and estimated lines
        _ax = ax1
        for _id, df in df_matlab.groupby('trk'):
            df.plot.scatter(x='x', y='y', c=cmap, ax=_ax)
            _ax.text(df['x'].iloc[0], df['y'].iloc[0], _id, fontsize=5)
        trk_lr, xi = est_lines(df_matlab, 4, res, ax=_ax)
        np.savetxt('/Users/Fabio/intersection.csv', xi, delimiter=',')
        _ax.set_xlabel('x $[\mu m]$')
        _ax.set_ylabel('y $[\mu m]$')

        # plot one track and fit model
        _ax = ax3
        trk__id = 3615
        df = df_matlab[df_matlab['trk'] == trk__id]
        df.plot.scatter(x='x', y='y', c=cmap, ax=_ax)
        _ax.set_title(trk__id)
        _ax.set_xlabel('x $[\mu m]$')
        _ax.set_ylabel('y $[\mu m]$')

        pdf.savefig()
        plt.close()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)

        sns.jointplot(xi[:, 0], xi[:, 1], kind='kde')
        fig.gca().set_aspect('equal', 'datalim')

        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    do_compute = do_filter_stats = True
    # do_compute = do_filter_stats = False
    # do_compute, do_filter_stats = False, True

    _fig_size_A3 = (11.7, 16.5)
    _err_kws = {'alpha': 0.3, 'lw': 1}

    if do_compute:
        dir_base = '/Users/Fabio/data/lab/eb3'
        logging.info('importing data from %s' % dir_base)
        df_matlab = pd.DataFrame()

        # gathering data from u-track plugin
        # 1st level = conditions
        citems = [d for d in os.listdir(dir_base) if d[0] != '.']
        for cit in citems:
            cpath = os.path.join(dir_base, cit)
            run_i = 1
            if os.path.isdir(cpath):
                # 2nd level = dates
                ditems = [d for d in os.listdir(cpath) if d[0] != '.']
                for dit in ditems:
                    dpath = os.path.join(cpath, dit)
                    if os.path.isdir(dpath):
                        # 3rd level = results
                        ritems = [d for d in os.listdir(dpath) if d[0] != '.']
                        for rit in ritems:
                            rpath = os.path.join(dpath, rit)
                            if os.path.isdir(rpath):
                                # 4rd level = matlab result file
                                mitems = [d for d in os.listdir(rpath) if d[0] != '.']
                                for mit in mitems:
                                    mpath = os.path.join(rpath, mit)
                                    if os.path.isfile(mpath) and mit != 'time.mat':
                                        logging.info('importing %s' % mpath)

                                        # Import
                                        try:
                                            dir_data = rpath
                                            genfname = os.path.join(dir_data, 'time.mat')
                                            trkfname = dir_data + '/TrackingPackage/tracks/Channel_1_tracking_result.mat'
                                            imgname = os.path.join(dpath, mit[:-4] + '.tif')
                                            with tf.TiffFile(imgname, fastij=True) as tif:
                                                if tif.is_imagej is not None:
                                                    res = 'n/a'
                                                    if tif.pages[0].resolution_unit == 'centimeter':
                                                        # asuming square pixels
                                                        xr = tif.pages[0].x_resolution
                                                        res = float(xr[0]) / float(xr[1])  # pixels per cm
                                                        res = res / 1e4  # pixels per um
                                                    elif tif.pages[0].imagej_tags.unit == 'micron':
                                                        # asuming square pixels
                                                        xr = tif.pages[0].x_resolution
                                                        res = float(xr[0]) / float(xr[1])  # pixels per um
                                            df_mtlb = import_eb3_matlab(genfname, trkfname, tag='%s-%d' % (cit, run_i))
                                            df_mtlb.loc[:, 'condition'] = cit
                                            run_i += 1
                                            # Process
                                            df_mtlb['x'] /= res
                                            df_mtlb['y'] /= res
                                            df_matlab = df_matlab.append(df_mtlb)
                                        except IOError as ioe:
                                            logging.warning('could not import due to IO error: %s' % ioe)
        # compute speed and msd
        df_matlab = m.get_speed_acc(df_matlab, group=indiv_idx)
        df_matlab = m.get_msd(df_matlab, group=indiv_idx)

        df_matlab.to_pickle('/Users/Fabio/data/lab/eb3.pandas')
    else:
        df_matlab = pd.read_pickle('/Users/Fabio/data/lab/eb3.pandas')

    if do_filter_stats:
        df_flt = df_filter(df_matlab, k=1, msd_thr=3)
        df_flt = m.get_center_df(df_flt, group=indiv_idx)
        df_flt = m.get_trk_length(df_flt, group=indiv_idx)
        df_flt.to_pickle('/Users/Fabio/data/lab/eb3filter.pandas')

        # construct average track speed and track length
        dfi = df_flt.set_index('frame').sort_index()
        dfi['speed'] = dfi['speed'].abs()
        _df_avg = dfi.groupby(indiv_idx)['time', 'speed'].mean()
        _df_avg.loc[:, 'time'] = dfi.groupby(indiv_idx)['time'].first()
        _df_avg.loc[:, 'trk_len'] = dfi.groupby(indiv_idx)['x'].count()
        _df_avg.loc[:, 'length'] = df_matlab.groupby(indiv_idx).apply(m.agg_trk_length)
        _df_avg = _df_avg.reset_index()
        _df_avg.to_pickle('/Users/Fabio/data/lab/eb3stats.pandas')

        # write csv eb3 track data
        filename = '/Users/Fabio/data/lab/eb3_tracks.csv'
        with open(filename, 'w') as of:
            of.write('condition,run,id,frame,xm,ym\n')

        for _id, df in df_flt.reset_index().groupby('trk'):
            with open(filename, 'a') as f:
                df[['condition', 'tag', 'trk', 'frame', 'x', 'y']].to_csv(f, header=False, index=False)

    else:
        _df_avg = pd.read_pickle('/Users/Fabio/data/lab/eb3stats.pandas').reset_index()
        df_flt = pd.read_pickle('/Users/Fabio/data/lab/eb3filter.pandas')
        # msd_lreg = pd.read_pickle('/Users/Fabio/eb3reg.pandas')
        logging.info('Loaded %d tracks after filters' % df_flt['trk'].unique().size)

    # print m.get_trk_length(df_matlab.iloc[0:1000], group=['condition', 'tag', 'trk'])
    logging.info('making indiv plots')
    for id, dff in df_flt.groupby('condition'):
        dfavg = _df_avg[_df_avg['condition'] == id]
        logging.info('Plotting individuals for %s group.' % id)
        indiv_plots(dff, dfavg, pdf_fname='eb3_indv-%s.pdf' % id)

    logging.info('making stat plots')
    res = 1
    stats_plots(df_matlab, _df_avg, res, img_file=None)

    logging.info('making msd plots')
    msd_plots(df_flt)

    logging.info('making estimation plots')
    est_plots(df_flt, res)
