import logging
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
from sklearn import linear_model

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
pd.set_option('display.width', 320)


def import_eb3_matlab(filename, tag=None):
    x = sio.loadmat(filename, squeeze_me=True)
    tracks = x['tracksFinal']
    tracked_particles = tracks['tracksCoordAmpCG']
    trk_events = tracks['seqOfEvents']
    num_tracks = tracked_particles.shape[0]
    df_out = pd.DataFrame()
    logging.info('tracked_particles.shape=%s  trk_events.shape=%s' % (tracked_particles.shape, trk_events.shape))

    for ti, (_trk, _ev) in enumerate(zip(tracked_particles, trk_events)):
        if _ev.shape == (2, 4):
            ini_frame = _ev[0, 0] if _ev[0, 1] == 1 else None
            end_frame = _ev[1, 0] + 1 if _ev[1, 1] == 2 else None
            if ini_frame is not None and end_frame is not None:
                trk = np.reshape(_trk, [len(_trk) / 8, 8])
                _df = pd.DataFrame(data=trk[:, 0:2], columns=['x', 'y'])
                _df['trk'] = ti
                _df['frame'] = np.arange(ini_frame, end_frame, 1, np.int16)

                df_out = df_out.append(_df)
            else:
                raise Exception('Invalid event format.')

    if tag is not None:
        df_out['tag'] = tag

    return df_out


def df_filter(df, k=10, msd_thr=50.0):
    logging.info('%d tracks before filter' % (df['trk'].unique().size))
    # filter dataframe for tracks having more than K points
    filtered_df = pd.DataFrame()
    for _tid, _tdf in df.groupby('trk'):
        if len(_tdf.index) > k:
            filtered_df = filtered_df.append(_tdf)
    df = filtered_df
    logging.info('filtered %d tracks after selecting tracks with more that k=%d points' % (df['trk'].unique().size, k))

    # filter dataframe based on track's mobility
    filtered_df = pd.DataFrame()
    for _tid, _tdf in df.groupby('trk'):
        if _tdf['msd'].iloc[-1] > msd_thr:
            filtered_df = filtered_df.append(_tdf)
    df = filtered_df
    logging.info('filtered %d tracks after MSD filter with msd_thr=%0.1f' % (df['trk'].unique().size, msd_thr))

    return df


def msd(dfi):
    """
        Computes Mean Square Displacement as defined by:

        {\rm {MSD}}\equiv \langle (x-x_{0})^{2}\rangle ={\frac {1}{N}}\sum _{n=1}^{N}(x_{n}(t)-x_{n}(0))^{2}
    """
    dfout = pd.DataFrame()
    for _id, _df in dfi.groupby('trk'):
        x0, y0 = _df['x'].iloc[0], _df['y'].iloc[0]
        _msdx = _df.loc[:, 'x'].apply(lambda x: (x - x0) ** 2)
        _msdy = _df.loc[:, 'y'].apply(lambda y: (y - y0) ** 2)
        _df.loc[:, 'msd'] = _msdx + _msdy
        dfout = dfout.append(_df)

    return dfout


def msd_lreg(df):
    """
        Computes a linear regression of the Mean Square Displacement
    """
    msd_lr = pd.DataFrame()
    for _id, _df in df.groupby('trk'):
        # do linear regression of both tracks to see which has higher slope
        x = _df.index.values
        y = _df['msd'].values
        length = len(x)
        x = x.reshape(length, 1)
        y = y.reshape(length, 1)
        if np.isnan(y).any():
            logging.warn('MSD of track tag %d contains NaNs.' % _id)
        else:
            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            msdlr = pd.DataFrame()
            msdlr['trk'] = _id
            msdlr['msd_lfit_a'] = regr.coef_[0][0]
            msdlr['msd_lfit_b'] = regr.intercept_[0]

            msd_lr = msd_lr.append(msdlr)

    return msd_lr


def trk_length(df):
    """
        Computes path length for each track
    """
    dfout = pd.DataFrame()
    for _id, _df in df.groupby('trk'):
        _dx2 = _df.loc[:, 'x'].diff().apply(lambda x: x ** 2)
        _dy2 = _df.loc[:, 'y'].diff().apply(lambda y: y ** 2)
        dst = pd.DataFrame([{'trk': _id, 'd': np.sum(np.sqrt(_dx2 + _dy2))}])
        dfout = dfout.append(dst)

    return dfout.set_index('trk')


def stats_plots(df, df_stats, res, img_file=None):
    _df = df.reset_index()
    with PdfPages('/Users/Fabio/eb3_stats.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0:2, :])
        ax3 = plt.subplot(gs[2, 0])
        ax4 = plt.subplot(gs[2, 1])
        max_frame = _df['frame'].max()
        cmap = sns.color_palette('cool', n_colors=max_frame)

        # plot of each eb3 track
        _ax = ax1
        if img_file is not None:
            image = cv2.imread(img_file)
            _ax.imshow(image, extent=[0, 512 / res, 512 / res, 0])
        for _id, df in _df.groupby('trk'):
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

        sns.distplot(df_stats['d'], rug=True, ax=ax4)
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

        # # g = sns.JointGr_id(x='d', y='msd_sum', data=df_stat, space=0)
        # # g = g.plot_joint(sns.kdeplot, cmap='Blues_d')
        # g = g.plot_joint(plt.scatter, color='.5', edgecolor='white')
        # # g = g.plot_marginals(sns.kdeplot, shade=True,bins=msd_bins)
        # ax = g.ax_joint
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # # g.ax_marg_x.set_xscale('log')
        # # g.ax_marg_y.set_yscale('log')

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

        # plot one track and fit model
        _ax = ax3
        trk__id = 1041
        df = df_matlab[df_matlab['trk'] == trk__id]
        df.plot.scatter(x='x', y='y', c=cmap, ax=_ax)
        _ax.set_title(trk__id)

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
    # do_compute, do_filter_stats = True, False

    _fig_size_A3 = (11.7, 16.5)
    _err_kws = {'alpha': 0.3, 'lw': 1}

    fname = '/Users/Fabio/data/lab/eb3-control/data/Result of U2OS CDK1as EB3 +1NM on controls only.sld - Capture 1/TrackingPackage/tracks/Channel_1_tracking_result.mat'
    imgname = '/Users/Fabio/data/lab/eb3-control/input/U2OS CDK1as EB3 +1NM on controls only.sld - Capture 1.tif'
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

    if do_compute:
        df_matlab = import_eb3_matlab(fname).set_index('frame').sort_index()
        df_matlab['x'] /= res
        df_matlab['y'] /= res
        df_matlab = msd(df_matlab)
        # TODO: compute speed and acceleration
        df_matlab.to_pickle('/Users/Fabio/eb3.pandas')
    else:
        df_matlab = pd.read_pickle('/Users/Fabio/eb3.pandas')

    if do_filter_stats:
        df_stat = trk_length(df_matlab)
        df_stat.to_pickle('/Users/Fabio/eb3stats.pandas')

        df_flt = df_filter(df_matlab, k=10, msd_thr=5)
        df_flt.to_pickle('/Users/Fabio/eb3filter.pandas')

        # msd_lreg = msd_lreg(df_flt)
        # msd_lreg.to_pickle('/Users/Fabio/eb3reg.pandas')

        # write csv eb3 track data
        filename = '/Users/Fabio/eb3_tracks.csv'
        with open(filename, 'w') as of:
            of.write('id,frame,xm,ym\n')

        for _id, df in df_flt.reset_index().groupby('trk'):
            with open(filename, 'a') as f:
                df[['trk', 'frame', 'x', 'y']].to_csv(f, header=False, index=False)

    else:
        df_stat = pd.read_pickle('/Users/Fabio/eb3stats.pandas')
        df_flt = pd.read_pickle('/Users/Fabio/eb3filter.pandas')
        # msd_lreg = pd.read_pickle('/Users/Fabio/eb3reg.pandas')
        logging.info('Loaded %d tracks after filters' % df_flt['trk'].unique().size)

    logging.info('making stat plots')
    stats_plots(df_matlab, df_stat, res, img_file=imgname)

    logging.info('making msd plots')
    msd_plots(df_flt)

    logging.info('making estimation plots')
    est_plots(df_flt)
