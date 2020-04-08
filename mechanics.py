"""
Computes classic and statistical mechanics

All the formulas intended for using with the pandas apply funcion on a groupby.

When dataframe is expressed as df, then implies a normal dataframe. On th other hand, when dfi is mentioned,
it means that the Dataframe must be frame indexed.

"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def speed_acc(df, x='x', y='y', time='time', frame='frame'):
    df = df.set_index(frame).sort_index()
    df.loc[:, 'dist'] = (df[x] ** 2 + df[y] ** 2).map(np.sqrt)
    d = df.loc[:, [x, y, 'dist', time]].diff().rename(
        columns={x: 'dx', y: 'dy', 'dist': 'dD', time: 'dt'})
    df.loc[:, 'speed'] = d.dD / d.dt
    df.loc[:, 'acc'] = d.dD.diff() / d.dt
    return df.reset_index()


def velocity(df, x='x', y='y', time='time', frame='frame'):
    df = df.set_index(frame).sort_index()
    df = df.loc[[x, y, time]].diff().rename(columns={x: 'Vx', y: 'Vy', time: 'dt'})
    df.loc[:, 'Vx'] = df['Vx'] / df['dt']
    df.loc[:, 'Vy'] = df['Vx'] / df['dt']
    return df.reset_index()


def avg_speed(df, frame='frame', time='time', centrosome_label_col='centrosome'):
    df = df.set_index(frame).sort_index()
    dfa = df[df[centrosome_label_col] == 'A']
    dfb = df[df[centrosome_label_col] == 'B']
    if not dfa.empty and not dfb.empty:
        ddi = np.sqrt((dfb['x'].iloc[0] - dfa['y'].iloc[0]) ** 2 + (dfb['y'].iloc[0] - dfa['y'].iloc[0]) ** 2)
        ddf = np.sqrt((dfb['x'].iloc[-1] - dfa['y'].iloc[-1]) ** 2 + (dfb['y'].iloc[-1] - dfa['y'].iloc[-1]) ** 2)
        dd = np.sqrt((ddf - ddi) ** 2)
        dt = df[time].iloc[-1] - df[time].iloc[0]
        return dd / dt


def get_speed_acc(df, x='x', y='y', time='time', frame='frame', group=None):
    if df.empty:
        raise Exception('df is empty')
    kwargs = {'x': x, 'y': y, 'time': time, 'frame': frame}
    dout = df.groupby(group).apply(speed_acc, **kwargs).reset_index(drop=True)
    return dout


def get_speed_acc_rel_to(df, x='x', y='y', rx='rx', ry='ry', time='time', frame='frame', group=None):
    if df.empty:
        raise Exception('df is empty')
    kwargs = {'x': x, 'y': y, 'time': time, 'frame': frame}
    dout = df.groupby(group).apply(speed_acc, **kwargs).reset_index(drop=True)
    return dout


def dist_vel_acc_centrosomes(df, cell_unit_idx=[],
                             time_col='time', frame_col='frame',
                             x_col='x', y_col='y',
                             centrosome_label_col='centrosome'):
    def dist_between(df):
        dfu = df.set_index([frame_col, centrosome_label_col]).sort_index().unstack(centrosome_label_col)
        ddx = dfu[x_col]['A'] - dfu[x_col]['B']
        ddy = dfu[y_col]['A'] - dfu[y_col]['B']

        dist = (ddx ** 2 + ddy ** 2).map(np.sqrt)
        time = dfu[time_col].max(axis=1)
        dt = time.diff()
        dfu.loc[:, ('DistCentr', 'A')] = dist
        dfu.loc[:, ('DistCentr', 'B')] = dist
        dfu.loc[:, ('SpeedCentr', 'A')] = dist.diff() / dt
        dfu.loc[:, ('SpeedCentr', 'B')] = dist.diff() / dt
        dfu.loc[:, ('AccCentr', 'A')] = dist.diff().diff() / dt
        dfu.loc[:, ('AccCentr', 'B')] = dist.diff().diff() / dt
        return dfu.stack().reset_index()

    df = df.groupby(cell_unit_idx).apply(dist_between)
    return df.reset_index(drop=True)


def center_df(df):
    df = df.set_index('frame').sort_index()
    time_ini = df['time'].iloc[0]
    dist_ini = df['dist'].iloc[0]
    df.loc[:, 'time_i'] = df['time'] - time_ini
    df.loc[:, 'dist_i'] = df['dist'] - dist_ini
    return df.reset_index()


def get_center_df(df, time='time', dist='dist', frame='frame', group=None):
    df = df.rename(columns={dist: 'dist', frame: 'frame', time: 'time'})
    dout = df.groupby(group).apply(center_df)
    return dout.reset_index(drop=True)


def get_msd(df, x='x', y='y', time='time', frame='frame', group=None):
    logger.debug('computing msd')

    def msd(df):
        """
            Computes Mean Square Displacement as defined by:

            {\rm {MSD}}\equiv \langle (x-x_{0})^{2}\rangle ={\frac {1}{N}}\sum _{n=1}^{N}(x_{n}(t)-x_{n}(0))^{2}
        """
        df = df.set_index(frame).sort_index()
        x0, y0 = df[x].iloc[0], df[y].iloc[0]
        _msdx = df.loc[:, x].apply(lambda x: (x - x0) ** 2)
        _msdy = df.loc[:, y].apply(lambda y: (y - y0) ** 2)
        df.loc[:, 'msd'] = _msdx + _msdy
        return df.reset_index()

    dfout = df.groupby(group).apply(msd)
    return dfout.reset_index(drop=True)


def _msd_tag(df, time='time', centrosome_label='centrosome'):
    logger.debug('classifying msd')
    pd.set_option('mode.chained_assignment', None)
    # with pd.set_option('mode.chained_assignment', 'raise')
    dfreg = msd_lreg(df.set_index(time).sort_index(), centrosome_label=centrosome_label)
    mvtag = pd.DataFrame()
    for id, _df in df.groupby('indiv'):
        c_a = dfreg[(dfreg['indiv'] == id) & (dfreg['centrosome'] == 'A')]['msd_slope'].reset_index(drop=True)
        c_b = dfreg[(dfreg['indiv'] == id) & (dfreg['centrosome'] == 'B')]['msd_slope'].reset_index(drop=True)
        if len(c_a) == 0 or len(c_b) == 0: continue
        c_a = c_a[0]
        c_b = c_b[0]
        if c_a > c_b:
            _df.loc[_df[centrosome_label] == 'A', 'msd_cat'] = 'displacing more'
            _df.loc[_df[centrosome_label] == 'B', 'msd_cat'] = 'displacing less'
        else:
            _df.loc[_df[centrosome_label] == 'B', 'msd_cat'] = 'displacing more'
            _df.loc[_df[centrosome_label] == 'A', 'msd_cat'] = 'displacing less'
        mvtag = mvtag.append(_df)
    pd.set_option('mode.chained_assignment', 'warn')
    return mvtag


def msd_lreg(df, centrosome_label='centrosome'):
    """
        Computes a linear regression of the Mean Square Displacement
    """
    from sklearn import linear_model
    msd_lr = pd.DataFrame()
    for _id, _df in df.groupby('trk'):
        # do linear regression of both tracks to see which has higher slope
        x = _df.index.values
        y = _df['msd'].values
        length = len(x)
        x = x.reshape(length, 1)
        y = y.reshape(length, 1)
        if np.isnan(y).any():
            logging.warning('MSD of track tag %d contains NaNs.' % _id)
        else:
            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            msdlr = pd.DataFrame(data={'indiv': _df['indiv'].iloc[0],
                                       'condition': _df['condition'].iloc[0],
                                       'centrosome': _df[centrosome_label].iloc[0],
                                       'msd_slope': [regr.coef_[0][0]],
                                       'msd_intercept': [regr.intercept_[0]]})

            msd_lr = msd_lr.append(msdlr)

    return msd_lr


def agg_trk_length(df):
    """
        Computes path length
    """
    df = df.set_index('frame').sort_index()
    _dx2 = df.loc[:, 'x'].diff().apply(lambda x: x ** 2)
    _dy2 = df.loc[:, 'y'].diff().apply(lambda y: y ** 2)
    return np.sum((_dx2 + _dy2).apply(np.sqrt))


def trk_length(df):
    """
        Computes path length
    """
    df = df.set_index('frame').sort_index()
    _dx2 = df.loc[:, 'x'].diff().apply(lambda x: x ** 2)
    _dy2 = df.loc[:, 'y'].diff().apply(lambda y: y ** 2)
    df.loc[:, 's'] = np.sqrt(_dx2 + _dy2)
    df.at[0, 's'] = 0
    return df.reset_index()
    # return np.sum((_dx2 + _dy2).apply(np.sqrt))


def get_trk_length(df, x='x', y='y', time='time', frame='frame', group=None):
    """
        Computes path length for each group
    """
    df = df.rename(columns={x: 'x', y: 'y', frame: 'frame', time: 'time'})
    dfout = df.groupby(group).apply(trk_length)
    return dfout.reset_index(drop=True)
