import os
import re

import numpy as np
import pandas as pd
from sklearn import linear_model


class ImagejPandas(object):
    DIST_THRESHOLD = 0.5  # um before 1 frame of contact
    TIME_BEFORE_CONTACT = 30
    MASK_INDEX = ['Frame', 'Time', 'Nuclei', 'Centrosome']
    NUCLEI_INDIV_INDEX = ['condition', 'run', 'Nuclei']
    CENTROSOME_INDIV_INDEX = NUCLEI_INDIV_INDEX + ['Centrosome']

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
        # get all distances less than a threshold, order them by time and pick the earlier one
        cent_list = df.groupby('Centrosome').size().index
        if len(cent_list) == 2:
            dsf = ImagejPandas.dist_vel_acc_centrosomes(df)
            dsr = dsf[dsf['DistCentr'] <= distance_threshold]

            if dsr.size > 0:
                zeros_df = dsr[dsr['DistCentr'] == 0]
                if zeros_df.size > 0:
                    dsr = zeros_df.set_index('Frame').sort_index()
                    frame = dsr.index[0]
                    time = list(dsr[dsr.index == frame]['Time'])[0]
                    dist = list(dsr[dsr.index == frame]['DistCentr'])[0]
                else:
                    dsr = dsr.set_index('DistCentr').sort_index()
                    frame = list(dsr['Frame'])[0]
                    time = list(dsr['Time'])[0]
                    dist = list(dsf[dsf['Frame'] == frame]['DistCentr'])[0]

                return time, frame, dist
        return None, None, None

    @staticmethod
    def vel_acc_nuclei(df):
        df = df.set_index('Frame').sort_index()
        df.loc[:, 'CNx'] = df['NuclX'] - df['CentX']
        df.loc[:, 'CNy'] = df['NuclY'] - df['CentY']
        for c_i, dc in df.groupby('Centrosome'):
            dc.loc[:, 'Dist'] = np.sqrt(dc.CNx ** 2 + dc.CNy ** 2)  # relative to nuclei centroid
            d = dc.loc[:, ['CNx', 'CNy', 'Dist', 'Time']].diff()

            df.loc[df['Centrosome'] == c_i, 'Dist'] = np.sqrt(dc.CNx ** 2 + dc.CNy ** 2)  # relative to nuclei centroid
            df.loc[df['Centrosome'] == c_i, 'Speed'] = d.Dist / d.Time
            df.loc[df['Centrosome'] == c_i, 'Acc'] = d.Dist.diff() / d.Time

        return df.reset_index()

    @staticmethod
    def dist_vel_acc_centrosomes(df):
        cent_list = df.groupby('Centrosome').size().index
        if (len(cent_list) != 2) | (df.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1):
            # we accept just 1 value per (frame,centrosome)
            ds = pd.DataFrame()
            ds['Frame'] = np.NaN
            ds['Time'] = np.NaN
            ds['DistCentr'] = np.NaN
            ds['SpeedCentr'] = np.NaN
            ds['AccCentr'] = np.NaN
            return ds

        dc = df.set_index(['Frame', 'Time', 'Centrosome']).unstack()
        ddx = dc['CNx'][cent_list[0]] - dc['CNx'][cent_list[1]]
        ddy = dc['CNy'][cent_list[0]] - dc['CNy'][cent_list[1]]
        ds = pd.DataFrame()
        ds.loc[:, 'DistCentr'] = np.sqrt(ddx ** 2 + ddy ** 2)

        ds = ds.reset_index().set_index('Frame')
        d = ds.diff()
        ds.loc[:, 'SpeedCentr'] = d.DistCentr / d.Time
        ds.loc[:, 'AccCentr'] = d.DistCentr.diff() / d.Time

        return ds.reset_index()

    @staticmethod
    def msd_centrosomes(df):
        """
            Computes Mean Square Displacement as defined by:

            {\rm {MSD}}\equiv \langle (x-x_{0})^{2}\rangle ={\frac {1}{N}}\sum _{n=1}^{N}(x_{n}(t)-x_{n}(0))^{2}
        """
        dfout = pd.DataFrame()
        for id, _df in df.groupby(ImagejPandas.CENTROSOME_INDIV_INDEX):
            _df = _df.set_index('Time').sort_index()
            x0, y0 = _df['CentX'].iloc[0], _df['CentY'].iloc[0]
            _msdx = _df.loc[:, 'CentX'].apply(lambda x: (x - x0) ** 2)
            _msdy = _df.loc[:, 'CentY'].apply(lambda y: (y - y0) ** 2)
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
        if df.groupby(['Frame', 'Nuclei', 'Centrosome']).size().max() > 1:
            raise LookupError('this function accepts just 1 value per (frame,centrosome)')

        s = df.set_index(ImagejPandas.MASK_INDEX).sort_index()
        u = s.unstack('Centrosome')
        umask = u.isnull()  # true for interpolated values
        u = u.interpolate(limit=30, limit_direction='backward')

        return u.stack().reset_index(), umask.stack().reset_index()

    @staticmethod
    def join_tracks(df, cn, cm):
        u = df[df['Centrosome'].isin([cn, cm])]
        # search for sup{lny}
        supdn = u.groupby('Centrosome')['Time'].max()
        # get the time of the minimum of the two values
        tc = supdn[supdn == supdn.min()].values[0]

        s = u.set_index(ImagejPandas.MASK_INDEX).sort_index().unstack('Centrosome')
        # where are the NaN's?
        nans_are_in = s['Dist'].transpose().isnull().any(axis=1)
        values_where_nans_are_in = nans_are_in[nans_are_in].keys().values

        mask = s.isnull().stack().reset_index()

        if values_where_nans_are_in.size == 1:
            _timeidx = s.index.levels[1]
            if values_where_nans_are_in[0] == cm:
                g = s[_timeidx > tc].transpose().fillna(method='ffill').transpose()
            else:
                g = s[_timeidx > tc].transpose().fillna(method='bfill').transpose()
            s[_timeidx > tc] = g
        u = s.stack()

        return u.reset_index(), mask

    @staticmethod
    def process_dataframe(df, nuclei_list=None, centrosome_inclusion_dict=None, max_time_dict=None):
        # filter non wanted centrosomes
        keep_centrosomes = list()
        for _cnuc in centrosome_inclusion_dict: keep_centrosomes.extend(centrosome_inclusion_dict[_cnuc])
        df = df[df['Centrosome'].isin(keep_centrosomes)]
        df = ImagejPandas.vel_acc_nuclei(df)  # call to make distance & acceleration columns appear

        # assign wanted centrosomes to nuclei
        if centrosome_inclusion_dict is not None:
            for nuId in centrosome_inclusion_dict.keys():
                for centId in centrosome_inclusion_dict[nuId]:
                    df.loc[df['Centrosome'] == centId, 'Nuclei'] = nuId

        df.dropna(how='all', inplace=True)
        df_filtered_nucs, df_masks = pd.DataFrame(), pd.DataFrame()
        for (nucleus_id), filtered_nuc_df in df.groupby(['Nuclei']):
            if nucleus_id in nuclei_list:
                if (max_time_dict is not None) and (nucleus_id in max_time_dict):
                    filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time'] <= max_time_dict[nucleus_id]]

                try:
                    filtered_nuc_df, imask = ImagejPandas.interpolate_data(filtered_nuc_df)

                    # process interpolated data mask
                    im = imask.set_index(['Frame', 'Time', 'Nuclei', 'Centrosome'])
                    mask = (~im).reset_index()

                    # compute velocity with interpolated data
                    filtered_nuc_df = ImagejPandas.vel_acc_nuclei(filtered_nuc_df)
                    df_filtered_nucs = df_filtered_nucs.append(filtered_nuc_df)
                    df_masks = df_masks.append(mask)
                except LookupError as le:
                    print('%s. Check raw input data for nuclei=N%d' % (le, nucleus_id))

        return df_filtered_nucs, df_masks
