import logging
import os
import re

import numpy as np
import pandas as pd
from sklearn import linear_model

import mechanics as m


class ImagejPandas(object):
    DIST_THRESHOLD = 0.5  # um before 1 frame of contact
    TIME_BEFORE_CONTACT = 30
    MASK_INDEX = ['Frame', 'Time', 'Nuclei', 'CentrLabel']
    NUCLEI_INDIV_INDEX = ['condition', 'run', 'Nuclei']
    CENTROSOME_INDIV_INDEX = NUCLEI_INDIV_INDEX + ['CentrLabel']

    def __init__(self, filename):
        self.path_csv = filename
        self.df_centrosome = pd.read_csv(self.path_csv)
        base_path = os.path.dirname(filename)
        bname = os.path.basename(filename)
        self.fname = re.search('(.+)-table.csv$', bname).group(1)
        self.path_nuclei = '%s/%s-nuclei.csv' % (base_path, self.fname)

        self.df_nuclei = pd.read_csv(self.path_nuclei)
        self.centrosome_replacements = dict()

        self.df_centrosome.loc[self.df_centrosome.index, 'Time'] /= 60.0  # time in minutes
        self.df_centrosome[['Frame', 'Nuclei', 'Centrosome']].astype(np.int64, inplace=True)
        self.df_nuclei[['Frame', 'Nuclei']].astype(np.int64, inplace=True)

        # merge with nuclei data
        self.merged_df = self.df_centrosome.merge(self.df_nuclei)
        self.merged_df = self.merged_df.drop(['ValidCentroid'], axis=1)

    @staticmethod
    def get_contact_time(df, distance_threshold):
        if df.set_index(ImagejPandas.NUCLEI_INDIV_INDEX).index.unique().size > 1:
            raise Exception('this function accepts just one track pair per analysis.')
        # get all distances less than a threshold, order them by time and pick the earlier one
        time, frame, dist = None, None, None
        cent_list = df['CentrLabel'].unique()
        if len(cent_list) == 2:
            dsf = ImagejPandas.dist_vel_acc_centrosomes(df) if 'DistCentr' not in df else df
            dsr = dsf[dsf['DistCentr'] <= distance_threshold]
            tail_analysis = False
            if dsr.size > 0:
                zeros_df = dsr[dsr['DistCentr'] == 0]
                if zeros_df.size > 0:
                    dsr = zeros_df.set_index('Frame').sort_index()
                    frame = dsr.index[0]
                    time = dsr.loc[dsr.index == frame, 'Time'].iloc[0]
                    dist = dsr.loc[dsr.index == frame, 'DistCentr'].iloc[0]
                else:
                    dsd = dsr.reset_index().set_index('DistCentr').sort_index()
                    frame = dsd.iloc[0]['Frame']
                    time = dsd.iloc[0]['Time']
                    dist = dsd.index[0]
                    # most of the times, when centrosomes come together we see that one track is lost
                    tail_analysis = True
            else:
                tail_analysis = True

            if tail_analysis:
                logging.debug('doing tail analysis for %s %s %s' % (
                    df.iloc[0]['condition'], df.iloc[0]['run'], df.iloc[0]['Nuclei']))
                dsu = dsf.set_index(['Frame', 'CentrLabel']).sort_index().unstack('CentrLabel')
                # most of the times, when centrosomes come together we see that one track is lost
                dsu = dsu.fillna(method='bfill')
                filtered = dsu[~dsu['DistCentr'].notna().all(axis=1)].stack().reset_index()
                # check that only one centrosome (eg 'A') is present
                if not filtered.empty and len(filtered['CentrLabel'].unique()) == 1:
                    q = filtered.iloc[0]
                    return q['Time'], q['Frame'], 0

        return time, frame, dist

    @staticmethod
    def vel_acc_nuclei(df):
        df = df.rename(columns={'CNx': '_x', 'CNy': '_y', 'Dist': 'dist', 'Speed': 'speed', 'Acc': 'acc'})
        df = m.get_speed_acc_rel_to(df, x='CentX', y='CentY', rx='NuclX', ry='NuclY', time='Time', frame='Frame',
                                    group=ImagejPandas.CENTROSOME_INDIV_INDEX)
        df = df.rename(columns={'_x': 'CNx', '_y': 'CNy', 'dist': 'Dist', 'speed': 'Speed', 'acc': 'Acc'})
        return df

    @staticmethod
    def dist_vel_acc_centrosomes(df):
        def dist_between(df):
            dfu = df.set_index(['Frame', 'CentrLabel']).sort_index().unstack('CentrLabel')
            ddx = dfu['CentX']['A'] - dfu['CentX']['B']
            ddy = dfu['CentY']['A'] - dfu['CentY']['B']

            dist = (ddx ** 2 + ddy ** 2).map(np.sqrt)
            time = dfu['Time'].max(axis=1)
            dfu.loc[:, ('DistCentr', 'A')] = dist
            dfu.loc[:, ('DistCentr', 'B')] = dist
            dfu.loc[:, ('SpeedCentr', 'A')] = dist.diff() / time.diff()
            dfu.loc[:, ('SpeedCentr', 'B')] = dist.diff() / time.diff()
            dfu.loc[:, ('AccCentr', 'A')] = dist.diff().diff() / time.diff()
            dfu.loc[:, ('AccCentr', 'B')] = dist.diff().diff() / time.diff()
            return dfu.stack().reset_index()

        df = df.groupby(ImagejPandas.NUCLEI_INDIV_INDEX).apply(dist_between)
        return df.reset_index(drop=True)

    @staticmethod
    def msd_particles(df, particle_x='CentX', particle_y='CentY'):
        """
            Computes Mean Square Displacement as defined by:

            {\rm {MSD}}\equiv \langle (x-x_{0})^{2}\rangle ={\frac {1}{N}}\sum _{n=1}^{N}(x_{n}(t)-x_{n}(0))^{2}
        """
        dfout = pd.DataFrame()
        for id, _df in df.groupby(ImagejPandas.CENTROSOME_INDIV_INDEX):
            _df = _df.set_index('Time').sort_index()
            x0, y0 = _df[particle_x].iloc[0], _df[particle_y].iloc[0]
            _msdx = _df.loc[:, particle_x].apply(lambda x: (x - x0) ** 2)
            _msdy = _df.loc[:, particle_y].apply(lambda y: (y - y0) ** 2)
            _df.loc[:, 'msd'] = _msdx + _msdy

            # do linear regression of both tracks to see which has higher slope
            x = _df.index.values
            y = _df['msd'].values
            length = len(x)
            x = x.reshape(length, 1)
            y = y.reshape(length, 1)
            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            _df.loc[:, 'msd_lfit_a'] = regr.coef_[0][0]

            dfout = dfout.append(_df.reset_index())
        return dfout

    @staticmethod
    def interpolate_data(df):
        if df.groupby(ImagejPandas.MASK_INDEX).size().max() > 1:
            raise LookupError('this function accepts just 1 value per (frame,centrosome)')

        def interpolate(df):
            s = df.set_index(['Frame', 'Time', 'Nuclei', 'CentrLabel']).sort_index()
            u = s.unstack('CentrLabel')
            u.fillna(value=np.nan, inplace=True)
            u = u.interpolate(limit=30, limit_direction='backward')
            idx = u['Centrosome'].notna().all(axis=1)

            u.loc[idx, ('condition', 'A')] = u.loc[idx, ('condition', 'A')].fillna(u.loc[idx, ('condition', 'B')])
            u.loc[idx, ('run', 'A')] = u.loc[idx, ('run', 'A')].fillna(u.loc[idx, ('run', 'B')])
            u.loc[idx, ('NuclBound', 'A')] = u.loc[idx, ('NuclBound', 'A')].fillna(u.loc[idx, ('NuclBound', 'B')])

            u.loc[idx, ('condition', 'B')] = u.loc[idx, ('condition', 'B')].fillna(u.loc[idx, ('condition', 'A')])
            u.loc[idx, ('run', 'B')] = u.loc[idx, ('run', 'B')].fillna(u.loc[idx, ('run', 'A')])
            u.loc[idx, ('NuclBound', 'B')] = u.loc[idx, ('NuclBound', 'B')].fillna(u.loc[idx, ('NuclBound', 'A')])
            return u.stack().reset_index()

        def mask(df):
            s = df.set_index(['Frame', 'Time', 'Nuclei', 'CentrLabel']).sort_index()
            u = s.unstack('CentrLabel')
            u.fillna(value=np.nan, inplace=True)
            umask = u.notna()  # false for interpolated values
            umask['condition'] = u['condition']
            umask['run'] = u['run']
            umask['Centrosome'] = u['Centrosome']
            return umask.stack().reset_index()

        dfout = df.groupby(ImagejPandas.NUCLEI_INDIV_INDEX).apply(interpolate)
        mask = df.groupby(ImagejPandas.NUCLEI_INDIV_INDEX).apply(mask)
        return dfout.reset_index(drop=True), mask.reset_index(drop=True)
