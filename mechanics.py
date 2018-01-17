"""
Computes classic and statistical mechanics

All the formulas intended for using with the pandas apply funcion on a groupby.

When dataframe is expressed as df, then implies a normal dataframe. On th other hand, when dfi is mentioned,
it means that the Dataframe must be frame indexed.

"""
import logging

import numpy as np
import pandas as pd
from sklearn import linear_model


def speed_acc(df):
    df = df.set_index('frame').sort_index()
    df.loc[:, 'dist'] = (df['_x'] ** 2 + df['_y'] ** 2).map(np.sqrt)
    d = df.loc[:, ['_x', '_y', 'dist', 'time']].diff().rename(
        columns={'_x': 'dx', '_y': 'dy', 'dist': 'dD', 'time': 'dt'})
    df.loc[:, 'speed'] = d.dD / d.dt
    df.loc[:, 'acc'] = d.dD.diff() / d.dt
    return df.reset_index()


def get_speed_acc(df, x='x', y='y', time='time', frame='frame', group=None):
    if df.empty:
        raise Exception('df is empty')
    df = df.rename(columns={frame: 'frame', time: 'time'})
    df.loc[:, '_x'] = df.loc[:, x]
    df.loc[:, '_y'] = df.loc[:, y]
    dout = df.groupby(group).apply(speed_acc)
    dout = dout.rename(columns={'frame': frame, 'time': time})
    return dout.drop(['_x', '_y'], axis=1).reset_index(drop=True)


def get_speed_acc_rel_to(df, x='x', y='y', rx='rx', ry='ry', time='time', frame='frame', group=None):
    if df.empty:
        raise Exception('df is empty')
    df = df.rename(columns={frame: 'frame', time: 'time'})
    df.loc[:, '_x'] = df[rx] - df[x]
    df.loc[:, '_y'] = df[ry] - df[y]
    dout = df.groupby(group).apply(speed_acc)
    dout = dout.rename(columns={'frame': frame, 'time': time})
    return dout.reset_index(drop=True)


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


def msd(df):
    """
        Computes Mean Square Displacement as defined by:

        {\rm {MSD}}\equiv \langle (x-x_{0})^{2}\rangle ={\frac {1}{N}}\sum _{n=1}^{N}(x_{n}(t)-x_{n}(0))^{2}
    """
    df = df.set_index('frame').sort_index()
    x0, y0 = df['x'].iloc[0], df['y'].iloc[0]
    _msdx = df.loc[:, 'x'].apply(lambda x: (x - x0) ** 2)
    _msdy = df.loc[:, 'y'].apply(lambda y: (y - y0) ** 2)
    df.loc[:, 'msd'] = _msdx + _msdy
    return df.reset_index()


def get_msd(df, x='x', y='y', time='time', frame='frame', group=None):
    df = df.rename(columns={x: 'x', y: 'y', frame: 'frame', time: 'time'})
    dfout = df.groupby(group).apply(msd)
    return dfout.reset_index(drop=True)


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
    return df.reset_index()
    # return np.sum((_dx2 + _dy2).apply(np.sqrt))


def get_trk_length(df, x='x', y='y', time='time', frame='frame', group=None):
    """
        Computes path length for each group
    """
    df = df.rename(columns={x: 'x', y: 'y', frame: 'frame', time: 'time'})
    dfout = df.groupby(group).apply(trk_length)
    return dfout.reset_index(drop=True)
