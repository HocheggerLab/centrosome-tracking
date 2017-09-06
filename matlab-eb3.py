# from sklearn.preprocessing import Imputer
import warnings

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import linear_model

pd.set_option('display.width', 320)


def import_eb3_matlab(fname, tag=None):
    x = sio.loadmat(fname, squeeze_me=True)
    tracks = x['tracksFinal']
    tracked_particles = tracks['tracksCoordAmpCG']
    trk_events = tracks['seqOfEvents']
    num_tracks = tracked_particles.shape[0]
    df_matlab = pd.DataFrame()
    print tracked_particles.shape, trk_events.shape

    for ti, (_trk, _ev) in enumerate(zip(tracked_particles, trk_events)):
        if _ev.shape == (2, 4):
            ini_frame = _ev[0, 0] if _ev[0, 1] == 1 else None
            end_frame = _ev[1, 0] + 1 if _ev[1, 1] == 2 else None
            if ini_frame is not None and end_frame is not None:
                trk = np.reshape(_trk, [len(_trk) / 8, 8])
                _df = pd.DataFrame(data=trk[:, 0:2], columns=['x', 'y'])
                _df['trk'] = ti
                # print _df, _df.shape
                # print np.arange(ini_frame, end_frame, 1, np.int16), np.arange(ini_frame, end_frame, 1, np.int16).shape
                _df['frame'] = np.arange(ini_frame, end_frame, 1, np.int16)

                df_matlab = df_matlab.append(_df)
            else:
                raise Exception('Invalid event format.')

    if tag is not None:
        df_matlab['tag'] = tag

    return df_matlab


def df_filter(df, df_stat, K=20, msd_thr=200.0):
    # filter dataframe for tracks having more than K points
    filtered_df = pd.DataFrame()
    for _tid, _tdf in df.groupby('trk'):
        if _tdf.size > K:
            filtered_df = filtered_df.append(_tdf)
    # df = filtered_df.reset_index()
    print '%d tracks after K points filter' % df['trk'].unique().size

    # filter dataframe based on track's mobility
    df = df[df['trk'].isin(df_stat[df_stat['msd_sum'] > msd_thr].index)]
    print '%d tracks after MSD filter' % df['trk'].unique().size

    return df


def msd(dfi):
    """
        Computes Mean Square Displacement as defined by:

        {\rm {MSD}}\equiv \langle (x-x_{0})^{2}\rangle ={\frac {1}{N}}\sum _{n=1}^{N}(x_{n}(t)-x_{n}(0))^{2}
    """
    dfout = pd.DataFrame()
    # for id, _df in dfi.groupby(['tag','trk']):
    for id, _df in dfi.groupby(['trk']):
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
    # for id, _df in dfi.groupby(['tag','trk']):
    for id, _df in df.groupby(['trk']):
        # do linear regression of both tracks to see which has higher slope
        x = _df.index.values
        y = _df['msd'].values
        length = len(x)
        x = x.reshape(length, 1)
        y = y.reshape(length, 1)
        if np.isnan(y).any():
            # raise ValueError('MSD of track tag %s %d contains NaNs.' % ('', id))
            warnings.warn('MSD of track tag %s %d contains NaNs.' % ('', id))
        else:
            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            msdlr = pd.DataFrame()
            msdlr['trk'] = id
            msdlr['msd_lfit_a'] = regr.coef_[0][0]
            msdlr['msd_lfit_b'] = regr.intercept_[0]

            msd_lr = msd_lr.append(msdlr)

    return msd_lr


def trk_length(df):
    """
        Computes path length for each track
    """
    dfout = pd.DataFrame()
    # for id, _df in dfi.groupby(['tag','trk']):
    for id, _df in df.groupby(['trk']):
        _dx2 = _df.loc[:, 'x'].diff().apply(lambda x: x ** 2)
        _dy2 = _df.loc[:, 'y'].diff().apply(lambda y: y ** 2)
        dst = pd.DataFrame([{'trk': id, 'd': np.sum(np.sqrt(_dx2 + _dy2))}])
        dfout = dfout.append(dst)

    return dfout.set_index('trk')


def stats_plots(_df, df_stat):
    _df = _df.reset_index()
    with PdfPages('/Users/Fabio/eb3.pdf') as pdf:
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
        for id, df in _df.groupby('trk'):
            df.plot.scatter(x='x', y='y', c=cmap, ax=ax1)

        # plot MSD sum distribution on semilog space
        msd_bins = np.logspace(-2, 4, 100)
        sns.distplot(df_stat['msd_sum'], rug=False, kde=False, bins=msd_bins, ax=ax3)
        # sns.kdeplot(df_stat['msd_sum'], gridsize=10e4, ax=ax3)
        # ax4.set_title('Distribution of track lengths for track id %s'%df_dist['trk'])
        ax3.set_title('Distribution of $\sum MSD$')
        ax3.set_ylabel('Frequency')
        ax3.set_xlabel('$\sum MSD$ $[\mu m]$')
        ax3.set_xscale('log')

        sns.distplot(df_stat['d'], rug=True, ax=ax4)
        # ax4.set_title('Distribution of track lengths for track id %s'%df_dist['trk'])
        ax4.set_title('Distribution of track lengths')
        ax4.set_ylabel('Frequency')
        ax4.set_xlabel('Length $[\mu m]$')

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
        ax3 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        # plot MSD sum distribution on semilog space
        msd_bins = np.logspace(-2, 2, 100)
        sns.distplot(df_stat['r1'], rug=False, kde=False, bins=msd_bins, ax=ax1)
        # ax4.set_title('Distribution of track lengths for track id %s'%df_dist['trk'])
        ax1.set_title('Distribution of $r_1$')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Ratio $r_1$ $[\mu m^{-1}]$')
        ax1.set_xscale('log')

        # plot of MSD for each track
        sns.tsplot(data=_df, lw=3,
                   err_style=['unit_traces'], err_kws=_err_kws,
                   time='frame', value='msd', unit='trk', estimator=np.nanmean, ax=ax6)
        ax6.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
        ax6.set_xticks(np.arange(0, _df['frame'].max(), 5))
        ax6.legend(title=None, loc='upper left')
        ax6.set_xlabel('Time delay $[frames]$')
        ax6.set_xticks(range(0, _df['frame'].max(), 5))
        ax6.set_xlim([0, _df['frame'].max()])

        pdf.savefig()
        plt.close()

        # ---------------------------
        #          NEXT PAGE
        # ---------------------------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)

        g = sns.JointGrid(x='d', y='msd_sum', data=df_stat, space=0)
        # g = g.plot_joint(sns.kdeplot, cmap='Blues_d')
        g = g.plot_joint(plt.scatter, color='.5', edgecolor='white')
        # g = g.plot_marginals(sns.kdeplot, shade=True,bins=msd_bins)
        ax = g.ax_joint
        ax.set_xscale('log')
        ax.set_yscale('log')
        # g.ax_marg_x.set_xscale('log')
        # g.ax_marg_y.set_yscale('log')

        pdf.savefig()
        plt.close()


def msd_plots(df_matlab):
    df_matlab = df_matlab.reset_index()
    with PdfPages('/Users/Fabio/eb3_identification.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0:2, :])
        ax3 = plt.subplot(gs[2, 0])
        ax4 = plt.subplot(gs[2, 1])

        # plot of each eb3 track
        max_frame = df_matlab['frame'].max()
        cmap = sns.color_palette('copper_r', n_colors=max_frame)
        for id, df in df_matlab.groupby('trk'):
            df.plot.scatter(x='x', y='y', c=cmap, ax=ax1)
            dfi = df.set_index('frame').sort_index()
            # ax1.text(dfi['x'].iloc[0], dfi['y'].iloc[0], id, fontsize=5)
            ax1.text(dfi['x'].iloc[0], dfi['y'].iloc[0], '%0.1f' % df_stat[df_stat.index == id]['msd_sum'], fontsize=5)

        # plot one track and fit model
        trk_id = 2351
        df = df_matlab[df_matlab['trk'] == trk_id]
        df.plot.scatter(x='x', y='y', c=cmap, ax=ax4)
        ax4.set_title(trk_id)
        # ax4.set_xlim(ax1.get_xlim())
        # ax4.set_ylim(ax1.get_ylim())

        # plt.savefig('/Users/Fabio/eb3_identification.png')
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    do_compute = False
    do_filter_stats = False
    _fig_size_A3 = (11.7, 16.5)
    _err_kws = {'alpha': 0.3, 'lw': 1}

    if do_compute:
        fname = '/Users/Fabio/data/lab/eb3-control/data/Result of U2OS CDK1as EB3 +1NM on controls only.sld - Capture 1/TrackingPackage/tracks/Channel_1_tracking_result.mat'
        # fname = '/Users/Fabio/data/lab/eb3-control/input/Result of U2OS CDK1as EB3 +1NM on controls only.sld - Capture 1/TrackingPackage/tracks/Channel_1_tracking_result.mat'
        df_matlab = import_eb3_matlab(fname).set_index('frame').sort_index()
        df_matlab = msd(df_matlab)
        # TODO: compute speed and acceleration
        df_matlab.to_pickle('/Users/Fabio/eb3.pandas')
    else:
        df_matlab = pd.read_pickle('/Users/Fabio/eb3.pandas')

    if do_filter_stats:
        df_stat = trk_length(df_matlab)
        df_stat['msd_sum'] = df_matlab.groupby('trk')['msd'].sum()
        df_stat['r1'] = df_stat['d'] / df_stat['msd_sum']
        df_stat.to_pickle('/Users/Fabio/eb3stats.pandas')

        df_flt = df_filter(df_matlab, df_stat)
        df_flt.to_pickle('/Users/Fabio/eb3filter.pandas')

        msd_lreg = msd_lreg(df_flt)
        msd_lreg.to_pickle('/Users/Fabio/eb3reg.pandas')
    else:
        df_stat = pd.read_pickle('/Users/Fabio/eb3stats.pandas')
        df_flt = pd.read_pickle('/Users/Fabio/eb3filter.pandas')
        msd_lreg = pd.read_pickle('/Users/Fabio/eb3reg.pandas')

    print 'stat plots'
    stats_plots(df_matlab, df_stat)

    print 'msd plots'
    msd_plots(df_flt)
